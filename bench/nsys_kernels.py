"""GPU compute by kernel: what the "compute (kernels, graphs)" line of nsys_split.py is made of.

Needs a report taken with `--cuda-graph-trace=node` (bench/oracle_nsys.py --nodes): otherwise the kernels inside CUDA
graphs are not recorded. Tokens are the same as nsys_split.py's: the starts of the remap ranges of the lowest layer on
the graph thread, whole tokens of the window only; kernels are taken between these bounds. Kernels are grouped into
kinds by name (mmvq, mmq, flash_attn, ...): per token - time on the GPU, launches, and the mean per launch.

    python bench/nsys_kernels.py REPORT.sqlite [--all]     # --all - every kernel name on its own
"""
import re
import sqlite3
import sys
from collections import defaultdict

KINDS = [   # the first substring match of the name wins
    ("mul_mat_vec_q", "mmvq: GEMV from quantized weights"),
    ("mul_mat_q", "mmq: GEMM from quantized weights"),
    ("mul_mat_vec_f", "mmvf: GEMV from float weights"),
    ("flash_attn", "flash_attn: attention"),
    ("quantize", "quantize: activations to q8_1"),
    ("argsort", "argsort / top-k"),
    ("top_k", "argsort / top-k"),
    ("k_get_rows", "get_rows"),
    ("get_rows", "get_rows"),
    ("set_rows", "set_rows: KV writes"),
    ("rms_norm", "rms_norm / norm"),
    ("norm_f32", "rms_norm / norm"),
    ("soft_max", "softmax"),
    ("rope", "rope"),
    ("bin_bcast", "add / mul / sub (bcast)"),
    ("unary", "unary: silu, sigmoid, ..."),
    ("glu", "glu: silu(gate) * up"),
    ("cpy", "cpy / convert"),
    ("convert", "cpy / convert"),
    ("concat", "concat"),
    ("sum_rows", "sum_rows / reduce"),
    ("scale", "scale"),
    ("gemm", "cuBLAS / CUTLASS gemm"),
    ("gemv", "cuBLAS gemv"),
    ("cutlass", "cuBLAS / CUTLASS gemm"),
]


def kind(name):
    for key, k in KINDS:
        if key in name:
            return k
    return "other: " + re.sub(r"<.*", "", name)[:70]


def token_bounds(db):
    """Starts of the lowest layer's remap ranges - as in nsys_split.py; the NVTX payload is the layer number."""
    cols = [r[1] for r in db.execute("pragma table_info(NVTX_EVENTS)")]
    pay = [c for c in ("int32Value", "int64Value", "uint32Value", "uint64Value") if c in cols]
    rows = db.execute(f"""SELECT n.start, n.globalTid, COALESCE(n.text, s.value){''.join(', n.' + c for c in pay)}
                          FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
                          WHERE n.end IS NOT NULL AND n.end > n.start""").fetchall()
    remaps = [(r[0], r[1], next((v for v in r[3:] if v is not None), None)) for r in rows if r[2] == "remap"]
    if not remaps:
        sys.exit("no remap ranges in the report (the fork's NVTX)")
    by_tid = defaultdict(list)
    for st, tid, lay in remaps:
        by_tid[tid].append((st, lay))
    tid, rs = max(by_tid.items(), key=lambda kv: len(kv[1]))   # the graph thread has the most remaps
    lmin = min(l for _, l in rs if l is not None)
    return sorted(st for st, l in rs if l == lmin)


def main():
    path = sys.argv[1]
    every = "--all" in sys.argv
    db = sqlite3.connect(path)
    try:
        ks = db.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                           JOIN StringIds s ON k.shortName = s.id""").fetchall()
    except sqlite3.OperationalError:
        sys.exit("no kernels in the report: take it with --cuda-graph-trace=node (oracle_nsys.py --nodes)")
    b = token_bounds(db)
    t0, t1, ntok = b[0], b[-1], len(b) - 1
    if ntok < 1:
        sys.exit("no whole token in the window")
    agg = defaultdict(lambda: [0, 0])
    busy = []
    for s, e, name in ks:
        if s < t0 or s >= t1:
            continue
        a = agg[name if every else kind(name)]
        a[0] += e - s
        a[1] += 1
        busy.append((s, e))
    # the GPU's time under kernels (union of intervals) - to check against "compute" of nsys_split.py
    busy.sort()
    union, cs, ce = 0, None, None
    for s, e in busy:
        if cs is None or s > ce:
            if cs is not None:
                union += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        union += ce - cs
    total = sum(v[0] for v in agg.values())
    print(f"tokens in the window: {ntok}; token {(t1 - t0) / ntok / 1e6:.1f} ms; kernels - sum {total / ntok / 1e6:.2f} "
          f"ms/tok, GPU under kernels {union / ntok / 1e6:.2f} ms/tok, {sum(v[1] for v in agg.values()) / ntok:.0f} "
          f"launches/tok")
    print(f"{'kind':52s} {'ms/tok':>8s} {'share':>6s} {'launches/tok':>13s} {'us/launch':>11s}")
    for k, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if not every and t / ntok < 1e3:   # under 1 us per token - not printed
            continue
        print(f"{k[:52]:52s} {t / ntok / 1e6:8.3f} {100 * t / total:5.1f}% {n / ntok:13.1f} {t / max(n, 1) / 1e3:11.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
