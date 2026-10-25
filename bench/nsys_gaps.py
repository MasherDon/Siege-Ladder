"""GPU idle time in an Nsight window, split by what bounds each gap.

The window is whole tokens, as in nsys_split.py (the remap starts of the lowest layer on the graph thread). GPU work:
C - compute (kernels, graph launches, memset), B - an H2D expert copy (1 MiB and up), S - a small H2D (inputs, slot
numbers), D - D2H. Every stretch where the GPU does nothing is assigned to the pair "the work that ended -> the work
that began" and to what the graph thread was doing meanwhile (the NVTX stall range of the remap, the fork's ranges of
the llama_decode phases, or CUDA calls by name). This shows where the idle time is the drive (B -> B and C -> B during
a stall) and where it is the hand-off between the CPU and the GPU: B -> S -> S -> C (the scheduler waits for the copies
with a sync, copies the slot numbers and launches the next graph) and D -> S - the token boundary (sampling and building
the next token's graph on the CPU).

    python bench/nsys_gaps.py REPORT.sqlite [REPORT2.sqlite ...]

Prints a table per report. Standard library only.
"""

import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict

BIG = 1 << 20   # an expert copy is 3.0 or 3.9 MB; the small ones are indices, inputs, outputs
# NVTX ranges of the graph thread the gaps are attributed to: stall - waiting for loads in the remap; the rest - the
# phases of llama_decode marked by the fork (graph build, the scheduler's allocation, inputs, launching the compute)
# and Engram's row reads
MAIN_NAMES = ("stall", "decode", "graph_build", "graph_alloc", "set_inputs", "graph_compute",
              "engram", "engram_prefetch", "engram_read", "engram_wait")


def main(path):
    db = sqlite3.connect(path)
    cur = db.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    strings = dict(cur.execute("select id, value from StringIds"))

    cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
    pay = [c for c in ("int32Value", "int64Value", "uint32Value", "uint64Value") if c in cols]
    remap, ranges = [], []
    for row in cur.execute(f"select start, end, globalTid, text, textId{''.join(', ' + c for c in pay)} "
                           f"from NVTX_EVENTS where end is not null and end > start"):
        s, e, tid, text, text_id = row[:5]
        name = text if text is not None else strings.get(text_id)
        il = next((v for v in row[5:] if v is not None), None)
        if name == "remap":
            remap.append((s, e, tid, il))
        elif name in MAIN_NAMES:
            ranges.append((s, e, tid, name))
    if not remap:
        sys.exit("no NVTX remap ranges in the report: the fork was built without LLAMA_MOE_STREAM_NVTX, or the window "
                 "is not on generation")
    by_tid = defaultdict(int)
    for _, _, tid, _ in remap:
        by_tid[tid] += 1
    main_tid = max(by_tid, key=by_tid.get)
    rm = sorted((s, e, il) for s, e, tid, il in remap if tid == main_tid)
    il0 = min(il for _, _, il in rm)
    bounds = [s for s, _, il in rm if il == il0]
    if len(bounds) < 3:
        sys.exit(f"fewer than two whole tokens in the window ({len(bounds)} starts)")
    lo, hi = bounds[0], bounds[-1]
    n_tok = len(bounds) - 1

    ev = []  # (start, end, kind)
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables:
            ev += [(s, e, "C") for s, e in cur.execute(f"select start, end from {t}")]
    for s, e, nb, kind in cur.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        if kind == 1:
            ev.append((s, e, "B" if nb >= BIG else "S"))
        elif kind == 2:
            ev.append((s, e, "D"))
    ev = sorted(x for x in ev if x[1] > lo and x[0] < hi)

    api = [(s, e, strings.get(nid, "?").split("_v")[0])
           for s, e, tid, nid in cur.execute("select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")
           if tid == main_tid and e > lo and s < hi]
    st = [(s, e, "stall (remap)" if name == "stall" else f"[{name}]")
          for s, e, tid, name in ranges if tid == main_tid and e > lo and s < hi]

    # rows clipped to the window and sorted by start; the longest row decides where a lookup has to start
    def make_index(rows):
        rows = sorted((max(r[0], lo), min(r[1], hi), r[2]) for r in rows)
        return rows, [r[0] for r in rows], max((r[1] - r[0] for r in rows), default=0)

    def overlap(idx, a, b):
        rows, starts, longest = idx
        out = defaultdict(int)
        i = max(0, bisect_left(starts, a - longest))
        while i < len(rows) and rows[i][0] < b:
            if rows[i][1] > a:
                out[rows[i][2]] += min(rows[i][1], b) - max(rows[i][0], a)
            i += 1
        return out

    api_idx, st_idx = make_index(api), make_index(st)

    # walk the union of busy time: each gap gets the kind of the work that ended last and the kind of the work that
    # closed the gap
    gaps = defaultdict(lambda: [0, 0])          # (before, after) -> [ns, count]
    during = defaultdict(lambda: defaultdict(int))
    busy_end, prev_kind = lo, None
    for s, e, k in ev:
        if s > busy_end and prev_kind is not None:
            a, b = max(busy_end, lo), min(s, hi)
            if b > a:
                key = (prev_kind, k)
                gaps[key][0] += b - a
                gaps[key][1] += 1
                for idx in (api_idx, st_idx):
                    for name, ns in overlap(idx, a, b).items():
                        during[key][name] += ns
        if e > busy_end:
            busy_end, prev_kind = e, k
        elif e == busy_end:
            prev_kind = k

    total = sum(v[0] for v in gaps.values())
    print(f"{path}\ntokens in the window: {n_tok}; token {(hi - lo) / n_tok / 1e6:.1f} ms; "
          f"GPU idle {total / n_tok / 1e6:.2f} ms per token")
    for key, (ns, cnt) in sorted(gaps.items(), key=lambda x: -x[1][0]):
        top = sorted(during[key].items(), key=lambda x: -x[1])[:6]
        print(f"  {key[0]} -> {key[1]}: {ns / n_tok / 1e6:6.2f} ms/tok  {cnt / n_tok:6.1f} gaps per token  "
              f"{ns / max(cnt, 1) / 1e3:7.1f} us on average")
        print("        graph thread during these gaps: " + ", ".join(f"{n} {v / n_tok / 1e6:.2f}" for n, v in top))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        main(p)
