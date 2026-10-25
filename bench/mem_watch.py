"""System RAM during a run (written for RAM tiers above 32 GiB): the least free memory and the hard-fault page reads.

    with MemWatch() as mw:
        subprocess.run(...)
    row.update(mw.result())   # mem_avail0_gb, mem_avail_min_gb, pages_in (4 KiB pages read to resolve hard faults)

    python bench/mem_watch.py       # self-test: watches 3 s and prints the summary

Free memory is GlobalMemoryStatusEx's ullAvailPhys (free + standby), polled every second in a thread. Pages in is the
system-wide cumulative counter "\\Memory\\Pages Input/sec" read raw through PDH by its English name (so it works on a
localized Windows). It counts the whole machine: the run's own Engram rows from the mapped files, plus whatever
Windows pages back in for other processes when the pinned RAM tier squeezes them. proc_times() gives a process's CPU
time. The caller puts the figures into its result row. Windows only (kernel32 and PDH through ctypes).
"""
import ctypes
import threading
from ctypes import wintypes


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PDH_RAW(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("TimeStamp", wintypes.FILETIME),
                ("FirstValue", ctypes.c_longlong), ("SecondValue", ctypes.c_longlong),
                ("MultiCount", wintypes.DWORD)]


def proc_times(handle):
    """-> (user_s, kernel_s) CPU time of a finished or running process, by its Win32 handle (Popen._handle)."""
    ft = [wintypes.FILETIME() for _ in range(4)]
    if not ctypes.windll.kernel32.GetProcessTimes(int(handle), *[ctypes.byref(f) for f in ft]):
        return None
    s = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7   # noqa: E731
    return s(ft[3]), s(ft[2])


def avail_gb():
    st = _MEMSTAT()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2**30


class _PagesIn:
    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh")
        self.q, self.c = wintypes.HANDLE(), wintypes.HANDLE()
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.q)) != 0:
            raise OSError("PdhOpenQuery")
        if self.pdh.PdhAddEnglishCounterW(self.q, "\\Memory\\Pages Input/sec", 0, ctypes.byref(self.c)) != 0:
            raise OSError("PdhAddEnglishCounter")

    def count(self):
        self.pdh.PdhCollectQueryData(self.q)
        raw = _PDH_RAW()
        self.pdh.PdhGetRawCounterValue(self.c, None, ctypes.byref(raw))
        return raw.FirstValue

    def close(self):
        self.pdh.PdhCloseQuery(self.q)


class MemWatch:
    def __init__(self, period_s=1.0):
        self.period = period_s
        self.avail0 = self.avail_min = None
        self.p0 = self.p1 = None
        self._stop = threading.Event()

    def _poll(self):
        while not self._stop.wait(self.period):
            self.avail_min = min(self.avail_min, avail_gb())

    def __enter__(self):
        try:
            self.pages = _PagesIn()
            self.p0 = self.pages.count()
        except OSError:
            self.pages = None
        self.avail0 = self.avail_min = avail_gb()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        if self.pages:
            self.p1 = self.pages.count()
            self.pages.close()
        return False

    def result(self):
        r = {"mem_avail0_gb": round(self.avail0, 2), "mem_avail_min_gb": round(self.avail_min, 2)}
        if self.p0 is not None and self.p1 is not None:
            r["pages_in"] = self.p1 - self.p0
        return r


if __name__ == "__main__":
    import time
    with MemWatch() as mw:
        time.sleep(3)
    print(mw.result())
