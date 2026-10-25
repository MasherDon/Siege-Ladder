"""Summary of a series of teacher-forced decode runs (the fork's llama-perplexity at -ub 1, one token per micro-batch),
as tf_run.py makes them.

A run is OUT/<arm>-<repeat>.log (base-a, base-b, idle-a ...; warm and other logs without a "-" appear only in the
table of runs). Time per token: from the line "calculating perplexity" to "Final estimate" by the log's timestamps,
divided by the number of tokens; the wait for expert loads: load stall; "outside": the difference. An arm is the mean
of its repeats; the difference is taken to the arm base (or --base NAME). The PPL of arms whose loss depends on
timing (dropped experts) is not compared - it only checks that the text is the same.

    python bench/tf_series.py OUT_DIR
    python bench/tf_series.py OUT_DIR --base base
    python bench/tf_series.py OUT_DIR --base def --only ns    # the block of *-ns.log alone (--without ns: the rest)

Prints the tables; writes nothing. The log lines it reads are the fork's: print_stats counters, drive-read latencies,
the D14 slow-read warnings (LLAMA_MOE_STREAM_SLOW_READ_MS: the drive's stalls), and the counters of the fork's other
techniques (G16, G31, G34, C14, G4, P16) when they ran.
"""
import re
import sys
from pathlib import Path

RE_TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")


def ts(line):
    m = RE_TS.match(line)
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


PATS = {
    "stall_ms": (r"load stall = ([\d.]+) ms total", float),
    "calls": (r"remap calls = (\d+), expert hits", int),
    "hits": (r"expert hits = (\d+), misses", int),
    "misses": (r"expert hits = \d+, misses = (\d+)", int),
    "reads": (r"drive reads = (\d+), [\d.]+ GB in", int),
    "gb": (r"drive reads = \d+, ([\d.]+) GB in", float),
    "dem_n": (r"drive reads \(demand\+stage\) = (\d+), avg", int),
    "dem_avg": (r"drive reads \(demand\+stage\) = \d+, avg ([\d.]+) ms", float),
    "dem_max": (r"drive reads \(demand\+stage\) = \d+, avg [\d.]+ ms, max ([\d.]+) ms", float),
    "spec_n": (r"drive reads \(speculative\) = (\d+), avg", int),
    "spec_avg": (r"drive reads \(speculative\) = \d+, avg ([\d.]+) ms", float),
    "dw_calls": (r"decode waits by source: (\d+) calls waited with a drive read", int),
    "dw_drive_s": (r"decode waits by source: \d+ calls waited with a drive read in flight, ([\d.]+) s", float),
    "dw_host_s": (r"; (\d+) on host-tier copies only, ([\d.]+) s", lambda m: float(m)),
    "left_drive": (r"time left from then: from the host tier \d+, mean [\d.]+ ms.*?; from the drive \d+, mean ([\d.]+) ms",
                   float),
    "drop_pairs": (r"drop rule .*?: (\d+) pairs of \d+ experts dropped", int),
    "mbox_ms": (r"G16 mailbox: \d+ posts read, the host waited ([\d.]+) ms", float),
    "mirror_gb": (r"P16 reads from the copies = \d+, ([\d.]+) GB", float),
    "g31_first_pct": (r"G31: \d+ calls split .*?; \d+ pairs \(([\d.]+)%\) computed before the wait", float),
    "g31_gap_ms": (r"first to second half ([\d.]+) ms per call", float),
    "g31_redo": (r"stream flushes; (\d+) experts loaded again", int),
    "g34_skip": (r"G34: (\d+) dropped experts not read ahead", int),
    "c14_pairs": (r"C14 probe E1 \(lossy\): (\d+) pairs summed apart", int),
    "g4_lane": (r"G4 probe \d \(lossy\): (\d+) experts to the stand-in lane", int),
    "ppl": (r"Final estimate: PPL = ([\d.]+)", float),
}


def one(path):
    d = {}
    last_slow = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if "calculating perplexity over" in line:
            d["t0"] = ts(line)
            m = re.search(r"over (\d+) chunks, n_ctx=(\d+)", line)
            d["ntok"] = int(m[1]) * int(m[2])
        elif "Final estimate" in line:
            d["t1"] = ts(line)
        elif "D14 slow read" in line:
            # D14 (LLAMA_MOE_STREAM_SLOW_READ_MS): reads ending within 5 ms of each other are one stall of the drive
            m = re.search(r"(\d+)\.(\d+)\.(\d+)\.(\d+) W pread_slab", line)
            if m:
                t = (int(m[1]) * 60 + int(m[2])) * 1000 + int(m[3]) + int(m[4]) / 1000.0
                if last_slow is None or t - last_slow > 5:
                    d["stalls"] = d.get("stalls", 0) + 1
                last_slow = t
            continue
        for k, (pat, cast) in PATS.items():
            m = re.search(pat, line)
            if m:
                d[k] = cast(m[m.lastindex] if k == "dw_host_s" else m[1])
    if "t0" not in d or "t1" not in d:
        return None
    n = d["ntok"]
    d["ms"] = (d["t1"] - d["t0"]) / n * 1000
    d["stall"] = d.get("stall_ms", 0.0) / n
    d["outside"] = d["ms"] - d["stall"]
    d["dw_drive"] = d.get("dw_drive_s", 0.0) * 1000 / n
    d["hit"] = 100 * d["hits"] / max(1, d["hits"] + d["misses"]) if "hits" in d else 0.0
    d["drop_tok"] = d.get("drop_pairs", 0) / n
    d["stalls_k"] = d.get("stalls", 0) * 1000.0 / n  # D14 stalls per 1000 tokens (0 without the slow-read log)
    return d


COLS = [("ms", "ms/tok", 7, 2), ("stall", "stall", 6, 2), ("outside", "outside", 7, 2), ("dw_drive", "drvwait", 7, 2),
        ("dem_avg", "dem_ms", 6, 2), ("dem_max", "dem_max", 7, 1), ("spec_n", "spec_n", 7, 0), ("left_drive", "left", 5, 2),
        ("hit", "hit%", 6, 2), ("drop_tok", "drop/t", 6, 2), ("mbox_ms", "mbox_ms", 8, 0), ("stalls_k", "D14/k", 6, 1),
        ("ppl", "PPL", 7, 4)]


def fmt(d, key, w, p):
    v = d.get(key)
    if v is None:
        return " " * (w - 1) + "-"
    return f"{v:{w}.{p}f}" if p else f"{int(v):{w}d}"


def main():
    args = sys.argv[1:]
    base = "base"
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1]
        del args[i:i + 2]
    # --only SUF: only the runs OUT/*-SUF.log, with the suffix cut off (a repeated block of the series, tf_run.py
    # --suffix SUF); --without SUF: all the others
    only = without = None
    for key in ("--only", "--without"):
        if key in args:
            i = args.index(key)
            if key == "--only":
                only = "-" + args[i + 1]
            else:
                without = "-" + args[i + 1]
            del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    folder = Path(args[0])
    rows = {}
    for p in sorted(folder.glob("*.log")):
        stem = p.stem
        if only:
            if not stem.endswith(only):
                continue
            stem = stem[:-len(only)]
        elif without and stem.endswith(without):
            continue
        d = one(p)
        if d is not None:
            rows[stem] = d
    print(f"{'run':12s} " + " ".join(f"{h:>{w}s}" for _, h, w, _ in COLS))
    for k, d in rows.items():
        print(f"{k:12s} " + " ".join(fmt(d, key, w, p) for key, _, w, p in COLS))
        extra = []
        for key in ("g31_first_pct", "g31_gap_ms", "g31_redo", "g34_skip", "c14_pairs", "g4_lane", "mirror_gb"):
            if key in d:
                extra.append(f"{key}={d[key]}")
        if extra:
            print(f"{'':12s} " + ", ".join(extra))
    arms = {}
    for k, d in rows.items():
        if "-" not in k:
            continue
        arms.setdefault(k.rsplit("-", 1)[0], []).append(d)
    if not arms:
        return
    print()
    mean = {}
    for arm, ds in arms.items():
        m = {key: sum(d.get(key, 0.0) for d in ds) / len(ds) for key in ("ms", "stall", "outside", "dw_drive", "dem_avg",
                                                                          "drop_tok", "hit")}
        mean[arm] = m
        reps = " / ".join(f"{d['ms']:.2f}" for d in ds)
        print(f"{arm:10s} n={len(ds)}: ms/tok {reps} -> {m['ms']:.3f} ({1000 / m['ms']:.2f} tok/s); stall {m['stall']:.2f}, "
              f"outside {m['outside']:.2f}, drive wait {m['dw_drive']:.2f}, demand read {m['dem_avg']:.2f} ms, "
              f"drops/tok {m['drop_tok']:.2f}, hit {m['hit']:.2f}%")
    if base in mean:
        print()
        b = mean[base]
        for arm, m in mean.items():
            if arm == base:
                continue
            print(f"{arm} - {base}: {m['ms'] - b['ms']:+.2f} ms/token ({100 * (b['ms'] / m['ms'] - 1):+.1f}% speed); "
                  f"stall {m['stall'] - b['stall']:+.2f}, outside {m['outside'] - b['outside']:+.2f}, "
                  f"drops/tok {m['drop_tok'] - b['drop_tok']:+.2f}")
    # D14: the drive stalls now and then under load (reads of 16 ms and more, LLAMA_MOE_STREAM_SLOW_READ_MS), and the
    # stalls hit arms unevenly; one price per stall for all arms (the regression inside each arm, the warm run counted
    # with the base), then each arm's mean at zero stalls
    groups = {arm: list(ds) for arm, ds in arms.items()}
    if "warm" in rows and base in groups:
        groups[base].append(rows["warm"])
    num = den = 0.0
    for ds in groups.values():
        mx = sum(d["stalls_k"] for d in ds) / len(ds)
        my = sum(d["ms"] for d in ds) / len(ds)
        for d in ds:
            num += (d["stalls_k"] - mx) * (d["ms"] - my)
            den += (d["stalls_k"] - mx) ** 2
    if den > 0:
        slope = num / den
        print(f"\nD14: {slope:.3f} ms/token per stall per 1000 tokens (one slope for all arms); at zero stalls:")
        free = {}
        for arm, ds in groups.items():
            mx = sum(d["stalls_k"] for d in ds) / len(ds)
            my = sum(d["ms"] for d in ds) / len(ds)
            free[arm] = my - slope * mx
            print(f"{arm:10s} n={len(ds)}: {my:.2f} at {mx:.1f} stalls/1000 -> {free[arm]:.2f} ms/token "
                  f"({1000 / free[arm]:.2f} tok/s)")
        if base in free:
            for arm, v in free.items():
                if arm != base:
                    print(f"{arm} - {base} at zero stalls: {v - free[base]:+.2f} ms/token "
                          f"({100 * (free[base] / v - 1):+.1f}% speed)")


if __name__ == "__main__":
    main()
