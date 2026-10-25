"""A prompt layer by layer from an Nsight report with the fork's wave ranges: where the reading time goes in each
layer - before the layer's waves (attention, router, norms, the tail of the previous layer) and in the waves (experts,
their loads and the wait for them).

A pass of a batch is a run of waves with a non-decreasing layer number (the layer number of a `wave` is its NVTX
payload); layer L of a pass runs from the end of the last wave of layer L-1 to the end of its own last wave: "before
waves" - from the end of the previous layer to the layer's first wave, "waves" - from the first to the last wave of the
layer. For layer 0, "before waves" counts from the first kernel of the pass.
The summary splits the encoder (layers 0-19) from the decoder (20-39). The ceiling of C7c (a further step of C7, the
YoCo prefill: compacting the batch so that the decoder layers compute only the rows from the YoCo bound) is the "before
waves" time of the decoder layers - their attention, dense layers and norms run on every row of the batch, and C7c
leaves them only the rows from the bound.

    python bench/nsys_layers.py REPORT.sqlite

Prints a table per pass (bench/turn_nsys.py keeps it as layers.txt next to the report). Standard library only.
"""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_waves import BIG, length, union  # noqa: E402


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
        nv[name].append((row[0], row[1], next((v for v in row[5:] if v is not None), None)))
    waves = sorted(w for w in nv.get("wave", []) if w[2] is not None)
    if not waves:
        sys.exit("no waves with a layer number in the report (NVTX `wave`)")
    stalls = sorted(w for w in nv.get("wave.stall", []) if w[2] is not None)

    kern = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        if t in tables:
            kern += [(s, e) for s, e in db.execute(f"select start, end from {t}")]
    h2d = [(s, e) for s, e, nb, kind in db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")
           if kind == 1 and nb >= BIG]
    uk, uh = union(kern), union(h2d)
    ms = lambda ns: ns / 1e6  # noqa: E731

    # passes: when a wave's layer number drops, the next pass (the next batch) has begun
    passes, cur = [], []
    for w in waves:
        if cur and w[2] < cur[-1][2]:
            passes.append(cur)
            cur = []
        cur.append(w)
    passes.append(cur)

    prev_pass_end = 0
    for p_i, p in enumerate(passes):
        by = defaultdict(list)
        for s, e, il in p:
            by[il].append((s, e))
        first = min(s for s, _, _ in p)
        # a pass starts at the first kernel after the end of the previous pass (the first one - after recording began)
        start = min((s for s, e in kern if prev_pass_end <= s < first), default=first)
        prev_end = start
        rows = []
        for il in sorted(by):
            s = min(x for x, _ in by[il])
            e = max(y for _, y in by[il])
            st = sum(min(b, e) - max(a, s) for a, b, l in stalls if l == il and b > s and a < e)
            rows.append((il, s - prev_end, e - s, st, length(uk, prev_end, s), length(uk, s, e), length(uh, s, e),
                         len(by[il])))
            prev_end = e
        last = max(e for _, e, _ in p)
        nxt = passes[p_i + 1][0][0] if p_i + 1 < len(passes) else None
        tail_end = max((e for s, e in kern if last <= s and (nxt is None or s < nxt)), default=last)
        prev_pass_end = last
        print(f"== pass {p_i + 1}: layers {len(rows)}, waves {len(p)}; from the first kernel to the last wave "
              f"{ms(last - start):.0f} ms; tail after the last wave (head, output) {ms(tail_end - last):.0f} ms")
        print("layer | before waves, ms (GPU compute) | waves, ms (wait; compute; copies) | waves")
        for il, pre, wv, st, cpre, cw, hw, n in rows:
            print(f"{il:4d} | {ms(pre):8.1f} ({ms(cpre):7.1f}) | "
                  f"{ms(wv):8.1f} ({ms(st):7.1f}; {ms(cw):7.1f}; {ms(hw):7.1f}) | {n}")
        for lo, hi, label in ((0, 19, "encoder 0-19"), (20, 39, "decoder 20-39")):
            sel = [r for r in rows if lo <= r[0] <= hi]
            if not sel:
                continue
            pre = sum(r[1] for r in sel)
            wv = sum(r[2] for r in sel)
            print(f"{label}: before waves {ms(pre):.0f} ms (GPU compute {ms(sum(r[4] for r in sel)):.0f}), waves "
                  f"{ms(wv):.0f} ms (wait {ms(sum(r[3] for r in sel)):.0f}, compute {ms(sum(r[5] for r in sel)):.0f}, "
                  f"copies {ms(sum(r[6] for r in sel)):.0f}); total {ms(pre + wv):.0f} ms")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
