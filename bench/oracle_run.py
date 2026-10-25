"""The fork's oracle: what perfect knowledge of future routes gives when experts are read ahead N calls.

At decode one call is one layer of one token, so AHEAD = 40 is one token ahead. LLAMA_MOE_STREAM_TRACE writes the
sequence of calls (layer -> experts) on a normal exit; LLAMA_MOE_STREAM_ORACLE replays it as read-ahead, and
LLAMA_MOE_STREAM_AHEAD sets the depth. The fork's read-ahead puts experts into the RAM tier only (drive -> RAM), not
into VRAM; LLAMA_MOE_STREAM_VRAM_AHEAD stages them into VRAM slots as well.

A replay matches the trace only if generation is deterministic. The fork's nondeterminism comes from the prefill waves
(their order depends on when the reads arrive; see the fork's README), so the prompt is read one token at a time
(-ub 1): no waves, as at decode. The request is a chat turn without thinking, greedy; the match is checked on the
answer's text. A raw continuation of a short prompt does not work: greedy generation with --ignore-eos falls into a
loop.

    python bench/oracle_run.py OUT trace             # record the trace (also a measurement without the oracle)
    python bench/oracle_run.py OUT base [M [FIRST]]  # M repeats without the oracle (base<FIRST>...): noise, determinism
    python bench/oracle_run.py OUT ahead 1,2,4,40    # the oracle N calls ahead
    python bench/oracle_run.py OUT stage 200 1 [sv]  # oracle 200 calls ahead into RAM, VRAM staging 1 call ahead
    python bench/oracle_run.py OUT stage 0 1         # the ceiling of a one-layer predictor: staging, no RAM read-ahead
    python bench/oracle_run.py OUT pred 6 0          # staging from the model's own prediction (A'): the best 6 into
                                                     # VRAM, the best 0 from the drive into RAM

Environment: ORACLE_IO=N - N I/O threads; ORACLE_ARGS="..." - more llama-completion flags for a series (they come
last, so a repeated flag overrides); ORACLE_REP=r - a repeat of the same configuration gets files of its own.
Results in OUT: <tag>.txt (the answer), <tag>.log (the fork's log with its counters), <tag>.gpu.csv (GPU samples) and a
row per run in runs.jsonl (speed, counters, drive temperatures and throttling, memory, GPU state, and same_text - the
answer compared with trace.txt). trace-calls.txt in OUT is the recorded trace. Needs SL_MODEL and SL_LLAMA_BIN
(llama-completion). Windows only (NVMe health through DeviceIoControl, PDH counters, process times).
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import nvme_health  # noqa: E402
from gpu_watch import GpuWatch, gen_window, wait_shared_free  # noqa: E402
from mem_watch import MemWatch, proc_times  # noqa: E402
from harness import MODEL, fork_env, keep_awake, slots_for  # noqa: E402

EXE = config.exe("llama-completion")   # in SL_LLAMA_BIN, next to llama-server (None until configured)
# the request of every oracle run: a long code answer (any such request works; ours was another code task)
PROMPT = ("Write a complete C module that parses a WAV audio file: the RIFF header, the fmt and data chunks, "
          "other chunks skipped. Define error codes, check every offset against the buffer length, and add "
          "unit tests.")
N_GEN = 512
ARGS = ["-m", str(MODEL), "-ngl", "99", "-t", "6", "--temp", "0", "-cnv", "-st", "--reasoning", "off", "--no-warmup",
        "-ub", "1", "-n", str(N_GEN), "-c", "4096", "--moe-stream-direct", "--moe-stream-cache", f"{slots_for(18)}s",
        "--moe-stream-l2", "24", "-p", PROMPT]
# the I/O thread count through the environment (the fork's default is 17; read-ahead takes half the pool: with 9
# threads that was a queue depth of ~5 and ~5.2 GB/s from the drive, while a drive test gave 10-13 GB/s at 8-9)
IO = os.environ.get("ORACLE_IO")
if IO:
    ARGS = ARGS[:-2] + ["--moe-stream-io-threads", IO] + ARGS[-2:]
# more llama-completion flags for a series (K - the experts per token, the RAM tier's size ...), e.g.
# ORACLE_ARGS="--moe-stream-l2 32"; they come after the flags above, so a repeated flag overrides them
EXTRA = shlex.split(os.environ.get("ORACLE_ARGS", ""))
if EXTRA:
    ARGS = ARGS[:-2] + EXTRA + ARGS[-2:]

PATS = {
    "pp_ms": r"prompt eval time =\s+([\d.]+) ms /\s+\d+ tokens",
    "pp_n": r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens",
    "tg_ms": r"\beval time =\s+([\d.]+) ms /\s+\d+ runs",
    "tg_n": r"\beval time =\s+[\d.]+ ms /\s+(\d+) runs",
    "stall_ms": r"load stall = ([\d.]+) ms total",
    "vram_hit": r"misses = \d+ \(\d+ cold\), hit rate = ([\d.]+)%",
    "misses": r"expert hits = \d+, misses = (\d+)",
    "l2_hit": r"L2 slab hits = \d+, fills = \d+, evictions = \d+, hit rate = ([\d.]+)%",
    "l2_fills": r"L2 slab hits = \d+, fills = (\d+)",
    "spec_queued": r"anticipated (\d+) slabs",
    "spec_filled": r"anticipated \d+ slabs, (\d+) read into the host tier",
    # a demand load that found its slab still LOADING in L2 waits for it instead of reading it again
    "l2_wait": r"L2 demand waits on a slab in flight = (\d+)",
    "l2_wait_hit": r"L2 demand waits on a slab in flight = \d+ \((\d+) served",
    "l2_wait_ms": r"served from it, ([\d.]+) ms waited",
    # experts staged into VRAM slots ahead of their call, with the oracle (LLAMA_MOE_STREAM_VRAM_AHEAD)
    "staged": r"staged (\d+) experts into VRAM",
    "stage_ready": r"call\(s\) ahead: (\d+) ready on arrival",
    "stage_late": r"ready on arrival, (\d+) still loading",
    "stage_skip": r"still loading, (\d+) left to demand",
    # staging from the model's own prediction (A', LLAMA_MOE_STREAM_PRED_VRAM / _RAM)
    "pred_staged": r"predicted the next layer on \d+ calls .*?: staged (\d+) experts",
    "pred_ready": r"staged \d+ experts - (\d+) ready on arrival",
    "pred_waste": r"still loading, (\d+) not asked for",
    "pred_offered": r"(\d+) offered to read ahead",
    # raw counts, so the books can be balanced (L2 lookups = 3 slabs × (misses + wasted stages))
    "hits": r"expert hits = (\d+)",
    "l2_hits": r"L2 slab hits = (\d+)",
    "io_reads": r"drive reads = (\d+), [\d.]+ GB",
    "io_gb": r"drive reads = \d+, ([\d.]+) GB",
    # drive reads as the workers see them (time inside ReadFile, no profiler)
    "io_gbps": r"drive reads = \d+, [\d.]+ GB in [\d.]+ s \(([\d.]+) GB/s\)",
    "io_inflight": r"GB/s\), ([\d.]+) in flight on average",
    "io_demand_ms": r"drive reads \(demand\+stage\) = \d+, avg ([\d.]+) ms",
    "io_spec_ms": r"drive reads \(speculative\) = \d+, avg ([\d.]+) ms",
}

# M12 (the fork's counters of where decode loads come from and how long they take): their lines repeat "ready on
# arrival / still loading / not asked for" and come after the staging line, so the patterns above would take their
# numbers (parse() keeps the last match); they are taken out before PATS and read by these instead
M12_LINE = re.compile(r"staged from the drive \(|decode waits by source|decode loads, reservation|time left from then")
M12_PATS = {
    "st_drive": r"staged from the drive \(some slab not in the host tier yet\): (\d+) of",
    "st_drive_ready": r"staged from the drive \(.*?\): \d+ of \d+ - (\d+) ready on arrival",
    "st_drive_late": r"staged from the drive \(.*?\): \d+ of \d+ - \d+ ready on arrival, (\d+) still loading",
    "st_tier_late": r"from the host tier: \d+ ready, (\d+) still loading",
    "wait_drive_calls": r"decode waits by source: (\d+) calls waited with a drive read in flight",
    "wait_drive_s": r"with a drive read in flight, ([\d.]+) s",
    "wait_tier_calls": r"; (\d+) on host-tier copies only",
    "wait_tier_s": r"on host-tier copies only, ([\d.]+) s",
    "dt_demand_drive_ms": r"reservation to the last copy enqueued: .*?demand from the drive \d+, mean ([\d.]+) ms",
    "dt_staged_drive_ms": r"reservation to the last copy enqueued: .*?staged from the drive \d+, mean ([\d.]+) ms",
    "dt_staged_tier_ms": r"reservation to the last copy enqueued: .*?staged from the host tier \d+, mean ([\d.]+) ms",
    "left_drive_n": r"time left from then: .*?from the drive (\d+), mean",
    "left_drive_ms": r"time left from then: .*?from the drive \d+, mean ([\d.]+) ms",
    # the drive waits split by whose read it is - the call's own miss or a stage ahead
    "wait_own_calls": r"\((\d+) of them on a drive read of their own",
    "wait_own_s": r"of their own, a miss not staged ahead, ([\d.]+) s\)",
    "left_staged_drive_ms": r"time left from then: .*?staged ahead \d+, mean ([\d.]+) ms",
    "left_own_drive_ms": r"time left from then: .*?misses of the call \d+, mean ([\d.]+) ms",
    "drop_late_exp": r"late drop \(lossy.*?\): \d+ pairs of (\d+) experts not waited for",
    # G21 (making runahead's pass cheaper): the share of the pass's pairs found on the card, and the time between the
    # remap's two halves (runahead, G2: between the halves the next layer's experts are predicted)
    "ra_kept_pct": r"the approximate pass kept ([\d.]+)% of their pairs",
    "ra_gap_ms": r"([\d.]+) ms between the halves on average",
}


_DISK = []


def need_bins():
    """llama-completion and the model must be configured (SL_LLAMA_BIN, SL_MODEL)."""
    if MODEL is None or EXE is None:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE})")


def disk_state():
    """Temperature sensors and HCTM counters of the NVMe drive that holds the model (SL_MODEL), or None: throttling
    during a run is judged by the TMT1 counter (time at thermal management threshold 1), not guessed from the speed."""
    try:
        if not _DISK:
            _DISK.append(nvme_health.disk_of(MODEL.resolve().drive.rstrip(":") if MODEL else ""))
        return nvme_health.health(_DISK[0])
    except (OSError, ValueError):
        return None


def disk_delta(h0, h1):
    if not (h0 and h1):
        return {}
    ctrl = lambda h: h["t_sens"][0] if h["t_sens"] else h["t_comp"]   # noqa: E731
    return {"ctrl0": ctrl(h0), "ctrl1": ctrl(h1), "tmt1_n": h1["tmt1_n"] - h0["tmt1_n"],
            "tmt1_s": h1["tmt1_s"] - h0["tmt1_s"], "tmt2_s": h1["tmt2_s"] - h0["tmt2_s"]}


def parse(log):
    out = {}
    base = "\n".join(ln for ln in log.splitlines() if not M12_LINE.search(ln))
    for pats, text in ((PATS, base), (M12_PATS, log)):
        for k, p in pats.items():
            m = [x for x in re.findall(p, text)]
            if m:
                out[k] = float(m[-1])
    if out.get("tg_ms") and out.get("tg_n"):   # a run that stopped at the first token prints "eval time = 0.00 ms"
        out["tg_tps"] = round(1000.0 * out["tg_n"] / out["tg_ms"], 3)
        out["stall_per_tok_ms"] = round(out.get("stall_ms", 0) / max(out["tg_n"] + 1, 1), 2)
    return out


def run(out, tag, env_extra):
    if IO:
        tag = f"{tag}-io{IO}"
    if os.environ.get("ORACLE_REP"):   # a repeat of the same configuration: files of its own, not over the earlier ones
        tag = f"{tag}-r{os.environ['ORACLE_REP']}"
    env = fork_env(env_extra)
    # the stream's variables set outside (in the caller's environment) are recorded too
    env_rec = {k: v for k, v in env.items() if k.startswith("LLAMA_MOE_STREAM_")}
    shared_wait = wait_shared_free()   # the previous run's pinned tier must be handed back first
    t0, h0 = time.time(), disk_state()
    with open(out / f"{tag}.txt", "w", encoding="utf-8") as fo, open(out / f"{tag}.log", "w", encoding="utf-8") as fe, \
            MemWatch() as mw, GpuWatch(out / f"{tag}.gpu.csv") as gw:
        t_start = time.time()   # the GPU's state is summarized over the generation window
        p = subprocess.Popen([str(EXE), *ARGS], cwd=str(EXE.parent), env=env, stdout=fo, stderr=fe,
                             stdin=subprocess.DEVNULL)
        try:
            rc = p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        cpu = proc_times(p._handle)   # CPU time of the whole run (load + prompt + generation)
    log = (out / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
    row = {"tag": tag, "rc": rc, "wall_s": round(time.time() - t0, 1), "env": env_rec, **parse(log),
           **disk_delta(h0, disk_state()), **mw.result()}
    w = gen_window(log, row.get("pp_ms"))
    row.update(gw.result(t_start + w[0], t_start + w[1]) if w else gw.result())
    if shared_wait[0] > 0:
        row["shared_wait_s"], row["shared_before_gib"] = shared_wait
    if cpu:
        row["cpu_user_s"], row["cpu_kernel_s"] = round(cpu[0], 1), round(cpu[1], 1)
    if EXTRA:
        row["args"] = EXTRA
    # with ORACLE_IO the trace run's own text is trace-io<N>.txt - the reference then, if there is no trace.txt.
    # No reference at all prints "no trace", not "= trace"
    ref = out / "trace.txt"
    if not ref.exists() and IO:
        ref = out / f"trace-io{IO}.txt"
    if ref.exists() and not tag.startswith("trace"):
        row["same_text"] = (out / f"{tag}.txt").read_bytes() == ref.read_bytes()
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    same = row.get("same_text")
    print(f"   {tag:10s} exit {rc} | generation {row.get('tg_tps', float('nan')):6.2f} tok/s | wait "
          f"{row.get('stall_per_tok_ms', float('nan')):6.1f} ms/tok | VRAM {row.get('vram_hit', 0):.1f}% | "
          f"L2 {row.get('l2_hit', 0):.1f}% | read ahead {row.get('spec_queued', 0):.0f}/"
          f"{row.get('spec_filled', 0):.0f} | "
          f"text {'no trace' if same is None else ('= trace' if same else '!= trace')} | {row['wall_s']:.0f} s | "
          f"controller {row.get('ctrl0', '?')}->{row.get('ctrl1', '?')} °C, TMT1 +{row.get('tmt1_s', '?')} s",
          flush=True)
    return row


def main():
    need_bins()
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[2]
    keep_awake(True)
    try:
        if mode == "trace":
            run(out, "trace", {"LLAMA_MOE_STREAM_TRACE": str(out / "trace-calls.txt")})
        elif mode == "base":
            first = int(sys.argv[4]) if len(sys.argv) > 4 else 1   # number of the first repeat: keep the earlier ones
            for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
                run(out, f"base{first + i}", {})
        elif mode == "ahead":
            for n in sys.argv[3].split(","):
                run(out, f"ahead{n}", {"LLAMA_MOE_STREAM_ORACLE": str(out / "trace-calls.txt"),
                                       "LLAMA_MOE_STREAM_AHEAD": n})
        elif mode == "stage":   # the oracle's read-ahead into RAM N calls and its staging into VRAM V calls ahead
            n, v = sys.argv[3], sys.argv[4]
            sv = len(sys.argv) > 5 and sys.argv[5] == "sv"   # and read ahead into RAM the experts now in VRAM too
            env = {"LLAMA_MOE_STREAM_ORACLE": str(out / "trace-calls.txt"), "LLAMA_MOE_STREAM_AHEAD": n,
                   "LLAMA_MOE_STREAM_VRAM_AHEAD": v}
            if sv:
                env["LLAMA_MOE_STREAM_SPEC_VRAM"] = "1"
            run(out, f"stage{n}v{v}{'sv' if sv else ''}", env)
        elif mode == "pred":    # the model's own prediction (A'): the best V into VRAM, the best R drive -> RAM
            v, r = sys.argv[3], sys.argv[4]
            run(out, f"pred{v}r{r}", {"LLAMA_MOE_STREAM_PRED_VRAM": v, "LLAMA_MOE_STREAM_PRED_RAM": r})
        else:
            sys.exit(__doc__)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()
