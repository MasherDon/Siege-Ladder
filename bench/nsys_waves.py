"""Analysis of a prefill window from an Nsight report with the fork's wave ranges: what a prompt is busy with.

The window runs from the start of the first to the end of the last wave (`wave` on the graph thread). In the window:
  graph thread  time in waves and waiting for their loads (`wave.stall`), outside waves (launches, syncs, own compute);
  GPU           compute (kernels), H2D expert copies (1 MiB and up), compute and copies at once, idle;
  drive         the reads `read` (on demand) and `spec` (read-ahead): how much of the window has at least one in flight,
                how many are in flight at once on average, their mean duration; gaps - stretches without a single read;
  copies        `upload` (a copy with its sync under the shared lock) and `upload.wait` (the queue for the lock) by
                threads.
Plus a wave's split: its wait, the launch of the next one and the GPU compute between waves; and the copy start (as in
bench/nsys_split.py): from the end of a copy's call to its start on the GPU, and the queue behind the copies of its
CUDA stream.

    python bench/nsys_waves.py REPORT.sqlite

Prints the summary (bench/prefill_nsys.py and bench/turn_nsys.py keep it as waves.txt next to the report). Standard
library only.
"""
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_split import copy_calls, copy_start  # noqa: E402

BIG = 1 << 20


def union(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(u, a=None, b=None):
    t = 0
    for s, e in u:
        if a is not None:
            s, e = max(s, a), min(e, b)
        if e > s:
            t += e - s
    return t


def inter(u1, u2):
    i = j = t = 0
    while i < len(u1) and j < len(u2):
        lo, hi = max(u1[i][0], u2[j][0]), min(u1[i][1], u2[j][1])
        if hi > lo:
            t += hi - lo
        if u1[i][1] < u2[j][1]:
            i += 1
        else:
            j += 1
    return t


def gaps(u, a, b):
    g, cur = [], a
    for s, e in u:
        s, e = max(s, a), min(e, b)
        if e <= s:
            continue
        if s > cur:
            g.append(s - cur)
        cur = max(cur, e)
    if b > cur:
        g.append(b - cur)
    return g


def main():
    db = sqlite3.connect(sys.argv[1])
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute("select id, value from StringIds"))
    cols = [r[1] for r in db.execute("pragma table_info(NVTX_EVENTS)")]
    pay = [c for c in ("int32Value", "int64Value") if c in cols]
    nv = defaultdict(list)
    for row in db.execute(f"select start, end, globalTid, text, textId{''.join(', ' + c for c in pay)} "
                          "from NVTX_EVENTS where end is not null and end > start"):
        name = row[3] if row[3] is not None else strings.get(row[4])
        nv[name].append((row[0], row[1], row[2], next((v for v in row[5:] if v is not None), None)))
    if not nv.get("wave"):
        sys.exit("no waves in the report (NVTX `wave`): a fork build without wave ranges, or a window without prefill")
    a = min(s for s, _, _, _ in nv["wave"])
    b = max(e for _, e, _, _ in nv["wave"])
    W = b - a
    ms = lambda ns: ns / 1e6  # noqa: E731

    waves = sorted(nv["wave"])
    stalls = sorted(nv.get("wave.stall", []))
    t_wave = sum(e - s for s, e, _, _ in waves)
    t_stall = sum(e - s for s, e, _, _ in stalls)
    print(f"wave window: {ms(W):.0f} ms; waves {len(waves)}, {len(stalls)} of them with a wait; layers "
          f"{len({il for _, _, _, il in waves})}")
    print(f"graph thread: in waves {ms(t_wave):.0f} ms ({100 * t_wave / W:.0f}%), of which waiting for loads "
          f"{ms(t_stall):.0f} ms ({100 * t_stall / W:.0f}%); outside waves {ms(W - t_wave):.0f} ms")

    comp = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        if t in tables:
            comp += [(s, e) for s, e in db.execute(f"select start, end from {t}") if e > a and s < b]
    h2d, nbytes = [], 0
    for s, e, nb, kind in db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        if kind == 1 and nb >= BIG and e > a and s < b:
            h2d.append((s, e))
            nbytes += nb
    uc, uh = union(comp), union(h2d)
    tc, th = length(uc, a, b), length(uh, a, b)
    both = inter(uc, uh)
    idle = W - tc - th + both
    print(f"GPU: compute {ms(tc):.0f} ms ({100 * tc / W:.0f}%), expert copies {ms(th):.0f} ms ({100 * th / W:.0f}%; "
          f"{nbytes / 1e9:.1f} GB, {nbytes / max(th, 1):.1f} GB/s while copying, {len(h2d)} copies), at once "
          f"{ms(both):.0f} ms, idle {ms(idle):.0f} ms ({100 * idle / W:.0f}%)")

    for names, label in ((("read", "spec"), "drive (read + spec)"), (("read",), "  of which read"),
                         (("spec",), "  of which spec")):
        iv = [(s, e) for n in names for s, e, _, _ in nv.get(n, []) if e > a and s < b]
        if not iv:
            print(f"{label}: none")
            continue
        u = union(iv)
        busy = length(u, a, b)
        tot = sum(min(e, b) - max(s, a) for s, e in iv)
        durs = [e - s for s, e in iv]
        line = (f"{label}: {len(iv)} reads, at least one in flight {100 * busy / W:.0f}% of the window, "
                f"{tot / W:.2f} at once on average, a read {ms(statistics.mean(durs)):.2f} ms "
                f"(median {ms(statistics.median(durs)):.2f})")
        if names == ("read", "spec"):
            g = gaps(u, a, b)
            g_big = [x for x in g if x >= 200_000]
            line += (f"; gaps without reads {ms(sum(g)):.0f} ms in {len(g)} stretches, of which 0.2 ms and longer - "
                     f"{ms(sum(g_big)):.0f} ms in {len(g_big)} "
                     f"(median {ms(statistics.median(g_big)) if g_big else 0:.2f} ms)")
        print(line)

    for n in ("upload", "upload.wait", "l2"):
        iv = [(s, e) for s, e, _, _ in nv.get(n, []) if e > a and s < b]
        if iv:
            print(f"{n}: {len(iv)} times, thread time {ms(sum(e - s for s, e in iv)):.0f} ms, at least one running "
                  f"{100 * length(union(iv), a, b) / W:.0f}% of the window, "
                  f"{ms(statistics.mean([e - s for s, e in iv])):.3f} ms on average")

    # a wave: its wait; the gap to the next wave (the GPU computing its GEMMs, and the sync)
    st_by = {}
    for s, e, _, il in stalls:
        st_by[s] = e - s
    between = [waves[i + 1][0] - waves[i][1] for i in range(len(waves) - 1)]
    stall_per = [e - s for s, e, _, _ in stalls]
    print(f"per wave: in stage_wave {ms(t_wave / len(waves)):.2f} ms, wait (of the waves that waited) "
          f"{ms(statistics.mean(stall_per)) if stall_per else 0:.2f} ms, from the end of a wave to the start of the "
          f"next - median {ms(statistics.median(between)):.2f} ms, mean {ms(statistics.mean(between)):.2f}")
    big_between = sorted(between)[-5:]
    print(f"longest gaps between waves, ms: {', '.join(f'{ms(x):.1f}' for x in big_between)}")
    # does WDDM hold a copy back in a batch?
    copy_start(copy_calls(db.cursor()), a, b, len(waves), unit="wave")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
