"""Paired comparison of ds4_score.py runs on one anchor: ΔNLL on the same API tokens against a base run, a 95%
bootstrap over the prompts, agreement with the API's top-1.

    python bench/ds4_compare.py BASE_DIR RUN_DIR [RUN_DIR ...]

BASE_DIR and RUN_DIR are OUT folders of ds4_score.py (positions.tsv or lp.tsv, and scores.tsv); the table goes to the
console. The yardstick: a mode counts as the original when it differs by no more than a change of engine or the
statistical error. A change of engine is the spread of ds4's own Q2 over four execution paths: 0.0042 nat of mean NLL
and 9 of 2994 top-1 matches. The error is the 95% interval of the paired difference. For scale: Q2 -> Q4 is 0.114 nat.
"""

import csv
import random
import sys
from pathlib import Path

ENGINE_NLL = 0.0042
ENGINE_TOP1 = 9
BOOT = 4000


def load_positions(d):
    """(case, pos) -> our log-prob of the API token; the repeat of case_000 at the end is checked, not used.

    positions.tsv (~2.7 MB) is read when present, and the columns needed here are written next to it as lp.tsv
    (~70 KB, small enough for git), so a comparison can be redone without positions.tsv."""
    src = Path(d) / "positions.tsv"
    if not src.exists():
        src = Path(d) / "lp.tsv"
    lp, again, rows = {}, {}, []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            rows.append("\t".join(p[:4]))  # case, pos, target, target_lp: the same columns in both files
            case, pos, v = p[0], int(p[1]), float(p[3])
            if "@" in case:  # the repeat of the first case at the end of the run (ds4_score.py), "case_000@again"
                again[(case.split("@")[0], pos)] = v
                continue
            lp[(case, pos)] = v
    if src.name == "positions.tsv" and not (Path(d) / "lp.tsv").exists():
        (Path(d) / "lp.tsv").write_text("# case\tpos\ttarget\ttarget_lp\n" + "\n".join(rows) + "\n", encoding="utf-8")
    repeat_diff = sum(1 for k, v in again.items() if lp.get(k) != v)
    return lp, repeat_diff


def load_top1(d):
    with open(Path(d) / "scores.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return sum(int(r["api_top1_match"]) for r in rows), sum(int(r["api_top1_count"]) for r in rows)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base_dir, runs = sys.argv[1], sys.argv[2:]
    base, base_rep = load_positions(base_dir)
    base_t1, n_t1 = load_top1(base_dir)
    base_nll = -sum(base.values()) / len(base)
    print(f"base {Path(base_dir).name}: NLL {base_nll:.4f}, top-1 {base_t1}/{n_t1}"
          + (f", the repeat of case_000 differs in {base_rep} tokens" if base_rep
             else ", the repeat of case_000 is bit for bit the same"))
    print(f"yardstick: a change of engine is {ENGINE_NLL} nat and {ENGINE_TOP1} top-1 matches; Q2 -> Q4 is 0.114 nat\n")
    print(f"{'mode':10s} {'NLL':>7s} {'ΔNLL':>8s} {'95% interval':>20s} {'top-1':>10s} {'Δ':>4s}  verdict")
    rng = random.Random(1)
    for d in runs:
        run, _ = load_positions(d)
        keys = [k for k in base if k in run]
        cases = {}
        for k in keys:
            s = cases.setdefault(k[0], [0.0, 0])
            s[0] += base[k] - run[k]  # = NLL(run) - NLL(base)
            s[1] += 1
        names = list(cases)
        dsum = sum(cases[c][0] for c in names)
        n = sum(cases[c][1] for c in names)
        dnll = dsum / n
        boot = []
        for _ in range(BOOT):
            pick = [names[rng.randrange(len(names))] for _ in names]
            boot.append(sum(cases[c][0] for c in pick) / sum(cases[c][1] for c in pick))
        boot.sort()
        lo, hi = boot[int(0.025 * BOOT)], boot[int(0.975 * BOOT) - 1]
        t1, _ = load_top1(d)
        nll = -sum(run[k] for k in keys) / n
        if lo <= 0.0 <= hi:
            verdict = "within error"
        elif dnll <= ENGINE_NLL and base_t1 - t1 <= ENGINE_TOP1:
            verdict = "at the level of a change of engine"
        else:
            verdict = "noticeable" + (" (better than the base)" if dnll < 0 else "")
        miss = len(base) - len(keys)
        print(f"{Path(d).name:10s} {nll:7.4f} {dnll:+8.4f} [{lo:+8.4f}, {hi:+8.4f}] {t1:5d}/{n_t1} {t1 - base_t1:+4d}  "
              f"{verdict}" + (f" ({miss} tokens of the base are not in the run)" if miss else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
