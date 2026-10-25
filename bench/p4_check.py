"""P4 check: the encoder-only prefill (P4) is bitwise exact at -ub 1.

V4.1 is a causal encoder-decoder: layers 20-39 leave only their window rings to later positions, so the fork with
LLAMA_DSV4_ENC_ONLY=1 computes a ubatch whose positions all lie further than the decoder's reach (2,413) before the
first position the call hands back with the encoder only (layers 0-19 and layer 20's cache writes). At -ub 1 every
ubatch is one token and rows do not affect each other, so the logits must equal the full path byte for byte.

The case: the consultant's C text (consult.context), a prompt of PROMPT tokens, the target - the next TARGET tokens
(teacher forcing, llama-logits --score-file, one call per prompt: -b is the case's length). Arms:
  A  no LLAMA_DSV4_ENC_ONLY key                - the reference
  B  LLAMA_DSV4_ENC_ONLY=1                     - must equal A byte for byte
  C  B + LLAMA_DSV4_ENC_ONLY_REACH=2412        - one position short of the reach: must differ
  D  B + LLAMA_DSV4_ENC_ONLY_REACH=2600        - with a margin: must equal A
The fork now runs P4 by default, so arm A takes it too unless the environment sets LLAMA_DSV4_ENC_ONLY=0 (the arms B-D
set the key themselves).

    python bench/p4_check.py OUT [PROMPT [TARGET [ARMS]]]     # defaults: 3000 16 ABCD

Results in OUT: positions-<arm>.tsv (llama-logits' top 64 per target position), run-<arm>.log, score_jobs.tsv and the
tokenizer's tok_jobs.tsv / tok_out.tsv; the verdict is printed. Needs SL_MODEL, SL_LLAMA_BIN (llama-logits) and
SL_LLAMA_SRC (the C text). Windows only (the fork's Windows build, SetThreadExecutionState).
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from consult import context  # noqa: E402
from harness import MODEL, fork_env, keep_awake, slots_for  # noqa: E402

EXE = config.exe("llama-logits")   # the patched build's scoring tool, in SL_LLAMA_BIN (None until configured)
ARMS = {
    "A": {},
    "B": {"LLAMA_DSV4_ENC_ONLY": "1"},
    "C": {"LLAMA_DSV4_ENC_ONLY": "1", "LLAMA_DSV4_ENC_ONLY_REACH": "2412"},
    "D": {"LLAMA_DSV4_ENC_ONLY": "1", "LLAMA_DSV4_ENC_ONLY_REACH": "2600"},
}


def tokenize(out, text):
    """The token ids of `text` by the model's tokenizer (llama-logits --tokenize-file)."""
    jobs = out / "tok_jobs.tsv"
    with open(jobs, "w", encoding="ascii", newline="\n") as f:
        f.write("# tag\tmode\thex\n")
        f.write(f"DOC\tt\t{text.encode('utf-8').hex()}\n")
    res = out / "tok_out.tsv"
    subprocess.run([str(EXE), "-m", str(MODEL), "--tokenize-file", str(jobs), "-o", str(res)], check=True,
                   capture_output=True)
    for line in res.read_bytes().decode("ascii").split("\n"):
        if line.startswith("DOC\t"):
            return [int(x) for x in line.split("\t")[2].split(",")]
    sys.exit("tok_out: no DOC line")


def main():
    if MODEL is None or EXE is None:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE})")
    out = Path(sys.argv[1])
    n_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    arms = sys.argv[4] if len(sys.argv) > 4 else "ABCD"
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
            pos = out / f"positions-{arm}.tsv"
            log = out / f"run-{arm}.log"
            # -b: the whole case in one call (P4 needs it); the window rings are sized by it (8448 cells x 40 layers -
            # 320 MiB at -b 8192 did not fit next to 18 slots), hence -b = the case's length and 17 slots
            cmd = [str(EXE), "-m", str(MODEL), "-ngl", "99", "-t", "6", "-c", "8192", "-ub", "1",
                   "-b", str(n_prompt + n_target),
                   "--moe-stream-direct", "--moe-stream-cache", f"{slots_for(17)}s", "--moe-stream-l2", "46",
                   "--moe-stream-io-threads", "17", "--top", "64", "--score-file", str(jobs), "-o", str(pos)]
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, env=fork_env(ARMS[arm]), stdout=lf, stderr=subprocess.STDOUT).returncode
            print(f"{arm} rc={rc} {time.time() - t0:.0f} s", flush=True)
    finally:
        keep_awake(False)
    ref = (out / "positions-A.tsv").read_bytes() if (out / "positions-A.tsv").exists() else None
    for arm in arms:
        p = out / f"positions-{arm}.tsv"
        if not p.exists():
            print(f"{arm}: no output")
            continue
        b = p.read_bytes()
        same = ref is not None and b == ref
        lines = [ln for ln in b.decode("ascii").split("\n") if ln and not ln.startswith("#")]
        p4 = [ln.strip() for ln in (out / f"run-{arm}.log").read_text(encoding="utf-8", errors="replace").splitlines()
              if "SiegeLadder P4" in ln]
        print(f"{arm}: {'= A byte for byte' if same else '!= A'} ({len(lines)} positions); "
              f"{p4[-1][:160] if p4 else 'no P4 lines'}")
        if not same and ref is not None:
            ra = [ln for ln in ref.decode("ascii").split("\n") if ln and not ln.startswith("#")]
            diff = [i for i, (x, y) in enumerate(zip(ra, lines)) if x != y]
            print(f"   first differing position: {diff[0] if diff else '-'} of {min(len(ra), len(lines))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()
