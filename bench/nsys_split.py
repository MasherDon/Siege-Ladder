"""Analysis of an Nsight Systems window with the fork's NVTX ranges: where a decode token's time goes.

Reads the report's SQLite export (`nsys stats` / `nsys export` put it next to the .nsys-rep). Token bounds are the
starts of the remap ranges of the lowest layer on the graph thread; only whole tokens of the window count. Per token
(mean and median):

  GPU           compute (kernels and graphs), H2D expert copies (1 MiB and up) and small ones, D2H, compute overlapping
                copies, idle (neither compute nor copies);
  graph thread  remap (within it: stall - waiting for loads, speculate - queueing read-ahead, victim - waiting for a
                free slot, stage - staging into VRAM for the next calls, predict - staging a guess; remap_wait - the
                second half of the remap with runahead), CUDA calls by name (syncs, graph launches, copies), the rest -
                CPU work outside CUDA (graph build, scheduler, sampling);
  I/O           l2 (looking up and waiting for a slab in the RAM tier), read (a demand read from the drive),
                upload.wait (the queue for a copy), upload (a copy with its sync), spec (a read-ahead read) - thread
                time and count;
  waits         what each stall consists of: a copy was running; no copy, but a demand read was; only a RAM-tier
                lookup; nothing visible (wake-ups, locks); and the tail - from the end of the last copy to leaving the
                wait.
  copy start    (does WDDM hold a copy back in a batch?) - for each expert copy, from the end of its cudaMemcpyAsync
                call (or from the end of the previous copy on the same CUDA stream, if that was still running) to its
                start on the GPU; and the queue - how long the copy waited for earlier copies on its stream.

    python bench/nsys_split.py REPORT.sqlite [--json OUT.json]

Prints the table; --json also writes it with the per-token values (bench/oracle_nsys.py keeps them as split.txt and
split.json next to the report). Standard library only.
"""

import json
import sqlite3
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict

# remap_wait - the second half of the remap with runahead (G2: between the remap's halves the next layer's experts are
# predicted from an approximate pass; without this range its time, the wait for loads included, would count as "CPU
# outside CUDA and remap"); predict - staging a guess (the model's own prediction A' / runahead)
MAIN_RANGES = ("remap", "remap_wait", "stall", "speculate", "victim", "stage", "predict")
WORK_RANGES = ("l2", "read", "upload.wait", "upload", "spec")
BIG = 1 << 20   # an expert copy is 3.0 or 3.9 MB; the small ones are indices, inputs, outputs


class Ivs:
    """A union of intervals [s, e) and the length of its intersection with a segment."""

    def __init__(self, iv):
        m = []
        for s, e in sorted(iv):
            if e <= s:
                continue
            if m and s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        self.m = m
        self.starts = [x[0] for x in m]

    def __or__(self, other):
        return Ivs(self.m + other.m)

    def __and__(self, other):
        out, i, j = [], 0, 0
        a, b = self.m, other.m
        while i < len(a) and j < len(b):
            lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            if hi > lo:
                out.append((lo, hi))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return Ivs(out)

    def length(self, a, b):
        i = max(bisect_right(self.starts, a) - 1, 0)
        tot = 0
        while i < len(self.m) and self.m[i][0] < b:
            lo, hi = max(self.m[i][0], a), min(self.m[i][1], b)
            if hi > lo:
                tot += hi - lo
            i += 1
        return tot


def clipped_sum(rows, a, b):
    """The sum of lengths clipped to [a, b) (no union: thread time) and the number of intervals that began in [a, b)."""
    tot, n = 0, 0
    for s, e in rows:
        lo, hi = max(s, a), min(e, b)
        if hi > lo:
            tot += hi - lo
        if a <= s < b:
            n += 1
    return tot, n


def copy_calls(cur):
    """H2D expert copies and their calls: [(start on the GPU, end on the GPU, CUDA stream, end of the call)]."""
    return list(cur.execute(
        "select m.start, m.end, m.streamId, r.end from CUPTI_ACTIVITY_KIND_MEMCPY m "
        "join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = m.correlationId "
        f"where m.copyKind = 1 and m.bytes >= {BIG}"))


def copy_start(calls, lo, hi, n_units, unit="token"):
    """The start of the copies that began on the GPU in [lo, hi): from the end of the call (or of the previous copy on
    the same CUDA stream, if that was still running) to the start on the GPU, and the queue behind the copies of its own
    stream. Prints and returns a summary."""
    prev_end, lags, queues = {}, [], []
    for s, e, stream, call_end in sorted(calls):
        before = prev_end.get(stream, 0)
        if lo <= s < hi:
            lags.append(max(s - max(call_end, before), 0))
            queues.append(max(before - call_end, 0))
        prev_end[stream] = max(before, e)
    if len(lags) < 2:
        return None
    q = statistics.quantiles(lags, n=100)
    out = {"copies": len(lags), "streams": len(prev_end), "lag_us_p50": statistics.median(lags) / 1e3,
           "lag_us_p90": q[89] / 1e3, "lag_us_p99": q[98] / 1e3, "lag_ms_per_unit": sum(lags) / 1e6 / n_units,
           "queue_us_p50": statistics.median(queues) / 1e3, "queue_ms_per_unit": sum(queues) / 1e6 / n_units}
    print(f"\nexpert copy start ({out['copies']} copies in the window, CUDA streams with copies: {out['streams']})")
    print(f"  from the call to the start on the GPU, us: median {out['lag_us_p50']:.1f}, p90 {out['lag_us_p90']:.1f}, "
          f"p99 {out['lag_us_p99']:.1f}; per {unit} {out['lag_ms_per_unit']:.2f} ms")
    print(f"  queue behind the copies of its own stream: median {out['queue_us_p50']:.1f} us, "
          f"per {unit} {out['queue_ms_per_unit']:.2f} ms")
    return out


def load(path):
    db = sqlite3.connect(path)
    cur = db.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    strings = dict(cur.execute("select id, value from StringIds"))

    cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
    pay = [c for c in ("int32Value", "int64Value", "uint32Value", "uint64Value") if c in cols]
    nvtx = defaultdict(list)   # name -> [(start, end, tid, layer)]
    rows = [] if "NVTX_EVENTS" not in tables else cur.execute(
        f"select start, end, globalTid, text, textId{''.join(', ' + c for c in pay)} "
        f"from NVTX_EVENTS where end is not null and end > start")
    for row in rows:
        s, e, tid, text, text_id = row[:5]
        name = text if text is not None else strings.get(text_id)
        if name in MAIN_RANGES or name in WORK_RANGES:
            nvtx[name].append((s, e, tid, next((v for v in row[5:] if v is not None), None)))

    # with --cuda-graph-trace=graph the kernels inside graphs show only as graph launches; the kernel table may then be
    # missing
    comp = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables:
            comp += list(cur.execute(f"select start, end from {t}"))
    h2d_big, h2d_small, d2h, big_bytes = [], [], [], []
    for s, e, nb, kind in cur.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        if kind == 1:
            (h2d_big if nb >= BIG else h2d_small).append((s, e))
            if nb >= BIG:
                big_bytes.append((s, nb))
        elif kind == 2:
            d2h.append((s, e))

    api = defaultdict(list)    # thread -> [(start, end, name)]
    for s, e, tid, nid in cur.execute("select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[tid].append((s, e, strings.get(nid, "?").split("_v")[0]))
    return nvtx, comp, h2d_big, h2d_small, d2h, big_bytes, api, copy_calls(cur)


def main():
    path = sys.argv[1]
    json_out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    nvtx, comp, h2d_big, h2d_small, d2h, big_bytes, api, big_calls = load(path)
    if not nvtx.get("remap"):
        sys.exit("no NVTX remap ranges in the report: the fork was built without LLAMA_MOE_STREAM_NVTX, or the window "
                 "is not on generation")

    by_tid = defaultdict(int)
    for s, e, tid, il in nvtx["remap"]:
        by_tid[tid] += 1
    main_tid = max(by_tid, key=by_tid.get)
    remaps = sorted((s, e, il) for s, e, tid, il in nvtx["remap"] if tid == main_tid)
    il0 = min(il for _, _, il in remaps)
    bounds = [s for s, _, il in remaps if il == il0]
    if len(bounds) < 3:
        sys.exit(f"fewer than two whole tokens in the window ({len(bounds)} starts)")
    n_layers = len({il for _, _, il in remaps})

    g_comp, g_big, g_small, g_d2h = Ivs(comp), Ivs(h2d_big), Ivs(h2d_small), Ivs(d2h)
    g_any = g_comp | g_big | g_small | g_d2h
    g_ovl = g_comp & (g_big | g_small)
    main_rows = {k: [(s, e) for s, e, tid, _ in nvtx.get(k, []) if tid == main_tid] for k in MAIN_RANGES}
    work_rows = {k: [(s, e) for s, e, tid, _ in nvtx.get(k, []) if tid != main_tid] for k in WORK_RANGES}
    w_upload, w_read, w_l2 = (Ivs(work_rows[k]) for k in ("upload", "read", "l2"))
    w_ur = w_upload | w_read
    w_url = w_ur | w_l2
    upload_ends = sorted(e for _, e in work_rows["upload"])
    main_api = sorted(api.get(main_tid, []))
    worker_tids = {tid for k in WORK_RANGES for _, _, tid, _ in nvtx.get(k, []) if tid != main_tid}

    toks = []
    for a, b in zip(bounds, bounds[1:]):
        t = {"token": b - a}
        t["gpu.compute"] = g_comp.length(a, b)
        t["gpu.h2d_experts"] = g_big.length(a, b)
        t["gpu.h2d_small"] = g_small.length(a, b)
        t["gpu.d2h"] = g_d2h.length(a, b)
        t["gpu.compute&copy"] = g_ovl.length(a, b)
        t["gpu.idle"] = (b - a) - g_any.length(a, b)
        t["h2d_GB"] = sum(nb for s, nb in big_bytes if a <= s < b) / 1e9
        for k in MAIN_RANGES:
            t[f"main.{k}"], t[f"n.{k}"] = clipped_sum(main_rows[k], a, b)
        api_sum = defaultdict(int)
        for s, e, name in main_api:
            if e > a and s < b:
                api_sum[name] += min(e, b) - max(s, a)
        t["main.api"] = dict(api_sum)
        t["main.cpu_other"] = (b - a) - t["main.remap"] - t["main.remap_wait"] - sum(api_sum.values())
        for k in WORK_RANGES:
            t[f"io.{k}"], t[f"n.{k}"] = clipped_sum(work_rows[k], a, b)
        # what this token's waits consist of
        an = defaultdict(int)
        for s, e in main_rows["stall"]:
            if not (a <= s < b):
                continue
            u, ur, url = w_upload.length(s, e), w_ur.length(s, e), w_url.length(s, e)
            an["copy"] += u
            an["read_no_copy"] += ur - u
            an["l2_only"] += url - ur
            an["nothing"] += (e - s) - url
            i = bisect_right(upload_ends, e) - 1
            if i >= 0 and upload_ends[i] >= s:
                an["tail"] += e - upload_ends[i]
        t["stall"] = dict(an)
        toks.append(t)

    def agg(get):
        v = [get(t) for t in toks]
        return statistics.mean(v) / 1e6, statistics.median(v) / 1e6

    n = len(toks)
    print(f"{path}\ntokens in the window: {n}; layers with a remap: {n_layers}; graph thread {main_tid}; "
          f"I/O threads with ranges: {len(worker_tids)}")
    mt, md = agg(lambda t: t["token"])
    print(f"token: {mt:.1f} ms (median {md:.1f}) -> {1000 / mt:.2f} tok/s under the profiler\n")
    rows = []

    def line(label, get, indent=2, count=None):
        m, d = agg(get)
        extra = f"   ×{statistics.mean([t[count] for t in toks]):.1f}" if count else ""
        print(f"{' ' * indent}{label:34s}{m:8.2f}{d:8.2f}{extra}")
        rows.append({"what": label.strip(), "mean_ms": round(m, 3), "median_ms": round(d, 3)})

    print(f"{'GPU':36s}{'ms/tok':>8s}{'median':>8s}")
    line("compute (kernels, graphs)", lambda t: t["gpu.compute"])
    line("expert copies H2D", lambda t: t["gpu.h2d_experts"])
    line("small H2D", lambda t: t["gpu.h2d_small"])
    line("D2H", lambda t: t["gpu.d2h"])
    line("compute and copies at once", lambda t: t["gpu.compute&copy"])
    line("idle", lambda t: t["gpu.idle"])
    gb = statistics.mean([t["h2d_GB"] for t in toks])
    copy_ms = agg(lambda t: t["gpu.h2d_experts"])[0]
    up_ms, n_up = agg(lambda t: t["io.upload"])[0], statistics.mean([t["n.upload"] for t in toks])
    print(f"  expert copies per token: {gb:.3f} GB, {1000 * gb / max(copy_ms, 1e-9):.1f} GB/s while copying; "
          f"{n_up:.0f} copies of {1000 * copy_ms / max(n_up, 1):.0f} us on the GPU and "
          f"{1000 * up_ms / max(n_up, 1):.0f} us on the thread (call + sync)")
    print(f"\n{'graph thread':36s}")
    line("remap", lambda t: t["main.remap"], count="n.remap")
    if any(t["n.remap_wait"] for t in toks):
        line("remap_wait (second half, runahead)", lambda t: t["main.remap_wait"], count="n.remap_wait")
    line("of which stall (waiting for loads)", lambda t: t["main.stall"], 4, count="n.stall")
    line("of which speculate", lambda t: t["main.speculate"], 4)
    line("of which victim", lambda t: t["main.victim"], 4, count="n.victim")
    line("of which stage (staging into VRAM)", lambda t: t["main.stage"], 4)
    line("of which predict (staging a guess)", lambda t: t["main.predict"], 4, count="n.predict")
    line("of which other work under the lock", lambda t: t["main.remap"] + t["main.remap_wait"] - t["main.stall"]
         - t["main.speculate"] - t["main.victim"] - t["main.stage"] - t["main.predict"], 4)
    names = defaultdict(int)
    for t in toks:
        for k, v in t["main.api"].items():
            names[k] += v
    for k in sorted(names, key=names.get, reverse=True)[:8]:
        line(f"CUDA: {k}", lambda t, k=k: t["main.api"].get(k, 0))
    line("outside CUDA and remap (CPU)", lambda t: t["main.cpu_other"])
    print(f"\n{'I/O (thread time)':36s}")
    for k in WORK_RANGES:
        line(k, lambda t, k=k: t[f"io.{k}"], count=f"n.{k}")
    print(f"\n{'what the waits consist of (per token)':36s}")
    for k, label in (("copy", "a copy was running"), ("read_no_copy", "no copy, a drive read was running"),
                     ("l2_only", "only a RAM-tier lookup"), ("nothing", "nothing visible"),
                     ("tail", "tail: end of the copy -> exit")):
        line(label, lambda t, k=k: t["stall"].get(k, 0))

    start = copy_start(big_calls, bounds[0], bounds[-1], n)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"report": path, "tokens": n, "layers": n_layers, "rows": rows, "copy_start": start,
                       "per_token": toks}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
