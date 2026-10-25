"""GPU state during a run: P-state, clocks, the PCIe link and the system memory the GPU maps.

    with GpuWatch(csv_path) as gw:
        t0 = time.time(); subprocess.run(...)
    w = gen_window(log_text, row.get("pp_ms"))          # generation, in seconds after the process started
    row.update(gw.result(*(t0 + x for x in w)) if w else gw.result())

    python bench/gpu_watch.py [CSV]     # self-test: 3 s of samples into CSV (default gpu_watch_test.csv), prints the summary

nvidia-smi writes a sample every PERIOD_MS into csv_path: P-state, SM and memory clocks, PCIe link generation and
width, power, temperature, utilization. The adapter counter "\\GPU Adapter Memory(*)\\Shared Usage" (system memory
mapped for the GPU; the pinned RAM tier counts there) is read through PDH once a second, by its English name (so it
works on a localized Windows); the largest adapter instance is kept. result() summarizes the samples inside a window
(the whole run by default): P-state and link generation as shares, SM clock p5 / p50 / max, power mean / max, the
largest shared usage - the caller puts them into its result row. ORACLE_GPUWATCH=0 in the environment turns the
watch off. wait_shared_free() waits for the driver to hand back the previous run's pinned memory.
Windows only (PDH through ctypes).
"""
import csv
import ctypes
import os
import subprocess
import threading
import time
from ctypes import wintypes

PERIOD_MS = 250
FIELDS = ("timestamp,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw,"
          "temperature.gpu,utilization.gpu")

PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2


class _FMT_VALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _FMT_ITEM(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FMT_VALUE)]


class _SharedUsage:
    """Largest "\\GPU Adapter Memory(*)\\Shared Usage" over the adapters, in GiB."""

    def __init__(self):
        pdh = self.pdh = ctypes.WinDLL("pdh")
        pdh.PdhOpenQueryW.restype = pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
        pdh.PdhCollectQueryData.restype = pdh.PdhGetFormattedCounterArrayW.restype = wintypes.DWORD
        pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.q, self.c = wintypes.HANDLE(), wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.q)) != 0:
            raise OSError("PdhOpenQuery")
        if pdh.PdhAddEnglishCounterW(self.q, "\\GPU Adapter Memory(*)\\Shared Usage", 0, ctypes.byref(self.c)) != 0:
            raise OSError("PdhAddEnglishCounter")

    def max_gb(self):
        self.pdh.PdhCollectQueryData(self.q)
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        r = self.pdh.PdhGetFormattedCounterArrayW(self.c, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if r != PDH_MORE_DATA or not size.value:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(self.c, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf):
            return None
        items = ctypes.cast(buf, ctypes.POINTER(_FMT_ITEM))
        return max((items[i].FmtValue.largeValue for i in range(count.value)), default=0) / 2**30

    def close(self):
        self.pdh.PdhCloseQuery(self.q)


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def gen_window(log, pp_ms=None):
    """(start, end) of generation in seconds after the process started, from llama's log timestamps
    (M.SS.mmm.uuu): the line "generate:" plus the prompt's time, and the first common_perf_print line."""
    import re
    ts = lambda m: int(m[0]) * 60 + int(m[1]) + int(m[2]) / 1e3 + int(m[3]) / 1e6   # noqa: E731
    g = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I generate:", log, re.M)
    e = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I common_perf_print:", log, re.M)
    if not (g and e):
        return None
    return ts(g.groups()) + (pp_ms or 0) / 1000.0, ts(e.groups())


class GpuWatch:
    def __init__(self, csv_path, period_s=1.0):
        self.csv = str(csv_path)
        self.period = period_s
        self.on = os.environ.get("ORACLE_GPUWATCH", "1") != "0"
        self.shared = []          # (time, GiB)
        self._stop = threading.Event()

    def _poll(self):
        while not self._stop.wait(self.period):
            v = self.su.max_gb()
            if v is not None:
                self.shared.append((time.time(), v))

    def __enter__(self):
        if not self.on:
            return self
        # to stdout, not -f: with -f nvidia-smi keeps the samples buffered and a terminated loop leaves the file empty
        self._fo = open(self.csv, "w", encoding="utf-8")
        self.smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                                     f"--loop-ms={PERIOD_MS}"], stdin=subprocess.DEVNULL,
                                    stdout=self._fo, stderr=subprocess.DEVNULL)
        try:
            self.su = _SharedUsage()
            self.su.max_gb()      # the first collection primes the query
            self._t = threading.Thread(target=self._poll, daemon=True)
            self._t.start()
        except OSError:
            self.su = None
        return self

    def __exit__(self, *exc):
        if not self.on:
            return False
        self.smi.terminate()
        self.smi.wait()
        self._fo.close()
        if self.su:
            self._stop.set()
            self._t.join()
            self.su.close()
        return False

    def samples(self):
        out = []
        try:
            with open(self.csv, encoding="utf-8", errors="replace") as f:
                for r in csv.reader(f):
                    if len(r) < 9:
                        continue
                    try:
                        t = time.mktime(time.strptime(r[0].strip()[:19], "%Y/%m/%d %H:%M:%S")) + \
                            float("0" + r[0].strip()[19:])
                        out.append({"t": t, "pstate": r[1].strip(), "sm": float(r[2]), "mem": float(r[3]),
                                    "gen": r[4].strip(), "width": r[5].strip(), "power": float(r[6]),
                                    "temp": float(r[7]), "util": float(r[8])})
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def result(self, t_from=None, t_to=None):
        if not self.on:
            return {}
        inside = lambda t: (t_from is None or t >= t_from) and (t_to is None or t <= t_to)   # noqa: E731
        s = [x for x in self.samples() if inside(x["t"])]
        r = {}
        if s:
            share = lambda key: {k: round(sum(1 for x in s if x[key] == k) / len(s), 3)   # noqa: E731
                                 for k in sorted({x[key] for x in s})}
            sm = [x["sm"] for x in s]
            r.update({"gpu_n": len(s), "gpu_pstate": share("pstate"), "gpu_pcie_gen": share("gen"),
                      "gpu_sm_mhz": [_pct(sm, 0.05), _pct(sm, 0.5), max(sm)],
                      "gpu_mem_mhz": _pct([x["mem"] for x in s], 0.5),
                      "gpu_power_w": [round(sum(x["power"] for x in s) / len(s), 1), max(x["power"] for x in s)],
                      "gpu_temp_max": max(x["temp"] for x in s),
                      "gpu_util_mean": round(sum(x["util"] for x in s) / len(s), 1)})
        sh = [v for t, v in self.shared if inside(t)]
        if sh:
            r["gpu_shared_max_gib"] = round(max(sh), 2)
        return r


def wait_shared_free(limit_gib=2.0, timeout_s=90.0):
    """Wait until the adapters' Shared Usage falls under limit_gib: the driver hands back the pinned RAM tier of the
    previous run only after that process has exited, and a 46 GiB tier started before that runs past the WDDM
    budget (CUDA out of memory at load). Returns (seconds waited, last GiB)."""
    try:
        su = _SharedUsage()
    except OSError:
        return 0.0, None
    t0 = time.time()
    v = su.max_gb()
    try:
        while v is not None and v > limit_gib and time.time() - t0 < timeout_s:
            time.sleep(1.0)
            v = su.max_gb()
    finally:
        su.close()
    return round(time.time() - t0, 1), (round(v, 2) if v is not None else None)


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "gpu_watch_test.csv"
    with GpuWatch(path) as gw:
        time.sleep(3)
    print(json.dumps(gw.result(), ensure_ascii=False))
