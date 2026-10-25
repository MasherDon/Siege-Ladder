"""C7 check: the decoder truncation in prefill (C7 - DeepInfer's YoCo prefill) is bitwise exact at -ub 1, as DeepInfer's
examples/yoco/yoco_equiv.py checks it (mask against compact) and as bench/p4_check.py checks P4 (the exact encoder-only
prefill).

At -ub 1 every ubatch is one token, so rows do not affect each other. C7 (LLAMA_DSV4_YOCO=K): the decoder (layers 20-39)
of a call keeps its last K positions - the rows from the bound keep_lo = (first output + 1) - K see the raw window of
these layers only from the bound on; whole ubatches below the bound run P4's encoder graph. Arms (the consultant's C
text from consult.context, a prompt of PROMPT tokens, the target - the next TARGET tokens, llama-logits --score-file,
one call per prompt):
  A   no C7 (P4 as by default)              - the exact path
  M   YOCO=140, MODE=mask                   - the bound only, no skips (neither P4 nor C7 skipping): C7's reference
  C   YOCO=140                              - the bound and the skips: must equal M byte for byte
  N   YOCO=140, NOBOUND=1                   - the skips without the bound (not C7): must differ from M
  X   YOCO=2541                             - the bound at the edge of the decoder's reach: e_39 = keep_lo + 2540 is
                                              below the first output - must equal A
  Y   YOCO=2540                             - one position further: it touches the first output - must differ from A
  m   as M, but the window cache is a ring of its window (--swa-window, as in the server and the quality anchors)
  c   as C, with --swa-window               - must equal m byte for byte
  D   the fork's default (C7 with K 140 when no key is set) - no preset, the exact path by LLAMA_MOE_STREAM_DROP_W=0,
      LLAMA_DSV4_YOCO unset: must equal C byte for byte

    python bench/c7_check.py OUT [PROMPT [TARGET [ARMS]]]     # defaults: 3000 16 AMCNXY

Results in OUT: positions-<arm>.tsv (llama-logits' top 64 per target position; the ring arms m / c are named
<arm>-ring), run-<arm>.log, score_jobs.tsv and the tokenizer's files; the verdict is printed. Needs SL_MODEL,
SL_LLAMA_BIN (llama-logits) and SL_LLAMA_SRC (the C text). Windows only (the fork's Windows build,
SetThreadExecutionState).
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from consult import context  # noqa: E402
from p4_check import tokenize  # noqa: E402
from harness import MODEL, fork_env, keep_awake, slots_for  # noqa: E402

EXE = config.exe("llama-logits")   # the patched build's scoring tool, in SL_LLAMA_BIN (None until configured)
ARMS = {
    "A": {},
    "M": {"LLAMA_DSV4_YOCO": "140", "LLAMA_DSV4_YOCO_MODE": "mask"},
    "C": {"LLAMA_DSV4_YOCO": "140"},
    "N": {"LLAMA_DSV4_YOCO": "140", "LLAMA_DSV4_YOCO_NOBOUND": "1"},
    "X": {"LLAMA_DSV4_YOCO": "2541"},
    "Y": {"LLAMA_DSV4_YOCO": "2540"},
    "m": {"LLAMA_DSV4_YOCO": "140", "LLAMA_DSV4_YOCO_MODE": "mask"},
    "c": {"LLAMA_DSV4_YOCO": "140"},
    "D": {"LLAMA_MOE_STREAM_DROP_W": "0"},
}
RING = {"m", "c"}
# arms without a preset and without LLAMA_DSV4_YOCO - the fork's default (fork_env's exact path sets YOCO=0; it is
# removed here)
DEFAULT_ARMS = {"D"}
EXPECT = {"C": ("M", True), "N": ("M", False), "X": ("A", True), "Y": ("A", False), "M": ("A", False),
          "c": ("m", True), "D": ("C", True)}


def name_of(arm):
    """The arm's name in file names: Windows file names ignore case, so the ring arms get the suffix -ring."""
    return arm + ("-ring" if arm in RING else "")


def lines_of(p):
    return [ln for ln in p.read_bytes().decode("ascii").split("\n") if ln and not ln.startswith("#")]


def main():
    if MODEL is None or EXE is None:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE})")
    out = Path(sys.argv[1])
    n_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    arms = sys.argv[4] if len(sys.argv) > 4 else "AMCNXY"
    out.mkdir(parents=True, exist_ok=True)
    ids = tokenize(out, context(n_prompt + n_target + 400))
    if len(ids) < n_prompt + n_target:
        sys.exit(f"the text is shorter than needed: {len(ids)} tokens")
    jobs = out / "score_jobs.tsv"
    with open(jobs, "w", encoding="ascii", newline="\n") as f:
        f.write("# case\tprompt ids\ttarget ids\n")
        f.write(f"c{n_prompt}\t{','.join(map(str, ids[:n_prompt]))}\t"
                f"{','.join(map(str, ids[n_prompt:n_prompt + n_target]))}\n")
    keep_awake(True)
    try:
        for arm in arms:
            pos = out / f"positions-{name_of(arm)}.tsv"
            log = out / f"run-{name_of(arm)}.log"
            # the exact preset (no drops): C7 is the only difference between the arms
            if arm in DEFAULT_ARMS:
                env = fork_env(ARMS[arm])
                env.pop("LLAMA_DSV4_YOCO", None)
                env.pop("LLAMA_MOE_STREAM_PRESET", None)
            else:
                env = fork_env({"LLAMA_MOE_STREAM_PRESET": "exact", **ARMS[arm]})
            cmd = [str(EXE), "-m", str(MODEL), "-ngl", "99", "-t", "6", "-c", "8192", "-ub", "1",
                   "-b", str(n_prompt + n_target),
                   "--moe-stream-direct", "--moe-stream-cache", f"{slots_for(17)}s", "--moe-stream-l2", "46",
                   "--moe-stream-io-threads", "17"] + (["--swa-window"] if arm in RING else []) + \
                  ["--top", "64", "--score-file", str(jobs), "-o", str(pos)]
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            print(f"{arm} rc={rc} {time.time() - t0:.0f} s", flush=True)
    finally:
        keep_awake(False)
    got = {a: (out / f"positions-{name_of(a)}.tsv").read_bytes() for a in arms
           if (out / f"positions-{name_of(a)}.tsv").exists()}
    ok_all = True
    for arm in arms:
        if arm not in got:
            print(f"{arm}: no output")
            ok_all = False
            continue
        c7 = [ln.strip() for ln in (out / f"run-{name_of(arm)}.log").read_text(encoding="utf-8", errors="replace")
              .splitlines() if "SiegeLadder C7" in ln]
        line = (f"{arm}: {len(lines_of(out / f'positions-{name_of(arm)}.tsv'))} positions; "
                f"{c7[-1][:170] if c7 else 'no C7 lines'}")
        if arm in EXPECT and EXPECT[arm][0] in got:
            ref, want_same = EXPECT[arm]
            same = got[arm] == got[ref]
            ok = same == want_same
            ok_all &= ok
            line += f"\n   {'=' if same else '!='} {ref} byte for byte - {'as expected' if ok else 'NOT AS EXPECTED'}"
            if not same:
                ra = lines_of(out / f"positions-{name_of(ref)}.tsv")
                la = lines_of(out / f"positions-{name_of(arm)}.tsv")
                diff = [i for i, (x, y) in enumerate(zip(ra, la)) if x != y]
                line += f" (first differing position: {diff[0] if diff else '-'} of {min(len(ra), len(la))})"
        print(line)
    print("RESULT:", "all as expected" if ok_all else "some arms differ from the expectation")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
