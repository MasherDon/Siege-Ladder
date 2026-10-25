"""Teacher-forced decode series: the fork's llama-perplexity at -ub 1 over a fixed text, the arms in A B B A order.

Every token goes through the model as a micro-batch of its own (-ub 1), so expert streaming does the work of decode
(VRAM expert cache, RAM tier, drive reads, read-ahead) on a fixed token sequence: every arm sees the same text, and
the differences between arms come from their switches, not from what they generated. -c 512 -b 512 cut the text into
chunks of 512 tokens (llama-perplexity needs at least 1,024 tokens); tf_series.py takes the time per token over all
of them. Arms differ by the fork's environment switches.

    python bench/tf_run.py OUT_DIR TEXT_FILE [VAR=VALUE ...] --arm NAME [VAR=VALUE ...] [--arm NAME [VAR=VALUE ...]]
                           [--reps N] [--suffix SUF] [-- llama-perplexity flags]
    python bench/tf_run.py bench/results/tf/presets text.txt --arm base --arm exact PRESET=exact --arm fast PRESET=fast
    python bench/tf_run.py bench/results/tf/g40 text.txt --arm base --arm g40 LLAMA_MOE_STREAM_G40=1 --reps 4
    python bench/tf_series.py bench/results/tf/presets       # the summary

VAR=VALUE pairs are environment variables: after an --arm they belong to that arm, before the first --arm to every run
(warm too). PRESET=NAME is short for LLAMA_MOE_STREAM_PRESET=NAME; VAR= passes an empty value. Unless a pair says
otherwise, every run gets
  LLAMA_MOE_STREAM_PRESET=default    the fork's preset (exact, default or fast)
  OMP_WAIT_POLICY=PASSIVE            OpenMP threads sleep instead of spinning while they wait
  LLAMA_MOE_STREAM_SLOW_READ_MS=16   D14: every drive read of 16 ms or more goes to the log; tf_series.py counts them
                                     as stalls of the drive and fits the arms' speeds at zero stalls
  LLAMA_MOE_STREAM_L2_SCAN=0         P9 off - the RAM tier's prompt mode (slabs a prompt touches are held, the rest
                                     stream through a ring): llama-perplexity hands each 512-token chunk over as one
                                     batch, which P9 would take for a prompt; in real decode the tier evicts by CLOCK
  LLAMA_MOE_STREAM_MIRROR            the expert copies of SL_MIRRORS, if any (harness.fork_env)

First a run named warm with the shared settings only (no arm's pairs); then N runs of each arm (--reps, default 2) in
sweeps that alternate direction: A B B A for two arms, A B C C B A for three. The logs - one per run, overwritten if
present - are OUT_DIR/warm.log and OUT_DIR/<arm>-a.log, <arm>-b.log, ...; the script prints a line per run (exit code,
wall time). tf_series.py counts the warm run with the base arm (by default the arm named base) in its D14 stall fit,
so give the base arm no pairs of its own. --suffix SUF names the logs <arm>-a-SUF.log and warm-SUF.log, for a second
block in the same folder (tf_series.py --only SUF / --without SUF). llama-perplexity flags after -- go at the end of
every run's command line (a later flag wins, e.g. -- --moe-stream-cache 25s).

Each run: llama-perplexity of SL_LLAMA_BIN on SL_MODEL with -ngl 99 -t 1 -c 512 -b 512 -ub 1 --moe-stream-direct,
expert slots harness.slots_for(17) (23 for cd, 25 for qd), RAM tier 46 GiB, 17 I/O threads. The text is yours: any
text of ~3K tokens works, e.g. the model's own answers to public prompts; absolute speeds depend on the text (the
experts it routes to), so compare arms within one series only. Before the first run the script waits until no llama-*
process runs (another GPU job would skew the times) and keeps the machine awake while it runs. Windows only (tasklist,
harness.keep_awake).
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import fork_env, keep_awake, slots_for  # noqa: E402

PERPLEXITY = config.exe("llama-perplexity")
# the series' environment; a pair on the command line wins
SERIES_ENV = {"OMP_WAIT_POLICY": "PASSIVE", "LLAMA_MOE_STREAM_SLOW_READ_MS": "16", "LLAMA_MOE_STREAM_L2_SCAN": "0",
              "LLAMA_MOE_STREAM_PRESET": "default"}
SHORT = {"PRESET": "LLAMA_MOE_STREAM_PRESET"}
RE_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.S)
RE_NAME = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")


def parse(argv):
    """-> (out_dir, text, shared pairs, [(arm, pairs)], reps, suffix, extra flags)."""
    flags = []
    if "--" in argv:
        i = argv.index("--")
        argv, flags = argv[:i], argv[i + 1:]
    if len(argv) < 2 or argv[0].startswith("-") or argv[1].startswith("-"):
        sys.exit(__doc__)
    out, text = Path(argv[0]), Path(argv[1])
    shared, arms, reps, suffix = {}, [], 2, ""
    cur = shared
    it = iter(argv[2:])
    for a in it:
        if a in ("--arm", "--reps", "--suffix"):
            v = next(it, None)
            if v is None:
                sys.exit(f"{a} needs a value")
            if a == "--arm":
                if not RE_NAME.fullmatch(v) or v == "warm" or v in (n for n, _ in arms):
                    sys.exit(f"bad or repeated arm name: {v} (letters, digits, _ . -; not warm)")
                cur = {}
                arms.append((v, cur))
            elif a == "--reps":
                reps = int(v) if v.isdigit() else 0
                if not 1 <= reps <= 26:
                    sys.exit("--reps: 1 to 26")
            else:
                if not RE_NAME.fullmatch(v):
                    sys.exit(f"bad suffix: {v}")
                suffix = v
        elif m := RE_PAIR.fullmatch(a):
            cur[SHORT.get(m[1], m[1])] = m[2]
        else:
            sys.exit(f"neither a VAR=VALUE pair nor an option: {a}")
    if not arms:
        sys.exit("give at least one --arm NAME")
    return out, text, shared, arms, reps, suffix, flags


def sequence(arms, reps):
    """[(arm, pairs, repeat letter)] in sweeps that alternate direction: A B, B A, A B, ..."""
    seq = []
    for r in range(reps):
        sweep = arms if r % 2 == 0 else arms[::-1]
        seq += [(name, pairs, chr(ord("a") + r)) for name, pairs in sweep]
    return seq


def llama_running():
    out = subprocess.run(["tasklist"], capture_output=True, text=True, errors="replace").stdout
    return "llama-" in out.lower()


def run_one(tag, pairs, out, cmd):
    env = fork_env({**SERIES_ENV, **pairs})
    t0 = time.time()
    with open(out / f"{tag}.log", "wb") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"{tag} rc={rc} wall={round(time.time() - t0)} s {time.strftime('%H:%M:%S')}", flush=True)
    return rc


def main():
    out, text, shared, arms, reps, suffix, flags = parse(sys.argv[1:])
    if config.MODEL is None or PERPLEXITY is None:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE})")
    if not PERPLEXITY.is_file():
        sys.exit(f"no {PERPLEXITY}")
    if not text.is_file():
        sys.exit(f"no text file {text}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(PERPLEXITY), "-m", str(config.MODEL), "-ngl", "99", "-t", "1", "-c", "512", "-b", "512", "-ub", "1",
           "--moe-stream-direct", "--moe-stream-cache", f"{slots_for(17)}s", "--moe-stream-l2", "46",
           "--moe-stream-io-threads", "17", "-f", str(text.resolve()), *flags]
    end = f"-{suffix}" if suffix else ""
    seq = sequence(arms, reps)
    show = lambda p: " ".join(f"{k}={v}" for k, v in p.items()) or "(series settings only)"   # noqa: E731
    print(f"tf_run: warm + {len(seq)} runs into {out}, start {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"  every run: {show(SERIES_ENV)}", flush=True)
    print(f"  warm: {show(shared)}", flush=True)
    for name, pairs in arms:
        print(f"  {name}: {show({**shared, **pairs})}", flush=True)
    print("  order: " + " ".join(f"{name}-{rep}" for name, _, rep in seq), flush=True)
    keep_awake(True)
    try:
        if llama_running():
            print("waiting for the running llama-* processes to exit", flush=True)
            while llama_running():
                time.sleep(5)
        run_one("warm" + end, shared, out, cmd)
        for name, pairs, rep in seq:
            run_one(f"{name}-{rep}{end}", {**shared, **pairs}, out, cmd)
    finally:
        keep_awake(False)
    names = [name for name, _ in arms]
    base = "" if "base" in names else f" --base {names[0]}"
    print(f"summary: python bench/tf_series.py {out}{base}" + (f" --only {suffix}" if suffix else ""), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
