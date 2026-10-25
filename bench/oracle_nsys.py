"""The fork's oracle under Nsight Systems: where a token's time goes with perfect drive -> RAM read-ahead.

The same run as bench/oracle_run.py (a chat turn, 512 tokens, -ub 1, K = 6 experts per token, RAM tier 24 GiB), with
the oracle of ORACLE_DIR (trace-calls.txt) AHEAD calls ahead (0 - no oracle), but llama-completion runs under
`nsys launch`. Collection covers a window of the generation: from WAIT_S s after the line "generate:" in the log (the
prompt is 49 tokens read one at a time, ~6-10 s), for DUR_S s. The fork marks NVTX ranges: remap / speculate / stall /
victim on the graph thread, l2 / read / upload.wait / upload / spec on the I/O threads; the payload is the layer.
bench/nsys_split.py analyses the window; its output goes to split.txt and split.json next to the report.

    python bench/oracle_nsys.py ORACLE_DIR AHEAD [WAIT_S [DUR_S]] [--light] [--vram V] [--sv] [--nodes]
    python bench/oracle_nsys.py bench/results/oracle/run1 200

ORACLE_DIR is a folder where `oracle_run.py ORACLE_DIR trace` recorded trace-calls.txt (and trace.txt, the answer the
run's text is compared with).
--light - no CPU stack sampling and no context switches: less profiler overhead on the CPU side (in one run 9.08 ->
10.30 tok/s under the profiler, against 11.18 without it), but it does not show where the threads wait.
--vram V - also stage into VRAM V calls ahead (LLAMA_MOE_STREAM_VRAM_AHEAD); --sv - read ahead into RAM the experts
that are in VRAM now, too (LLAMA_MOE_STREAM_SPEC_VRAM=1).
--nodes - trace every kernel inside CUDA graphs (`--cuda-graph-trace=node`): without it compute shows only as graph
launches and `cuda_gpu_kern_sum` is empty; the kernels go to kernels.txt (bench/nsys_kernels.py). The overhead is
higher - take the token split (split.txt) from a run without this flag.

Results: SL_RESULTS/nsys/<time>-oracle-<ORACLE_DIR name>-<tag>/ (run.txt - the answer, run.log, run.json - the parsed
counters, rep.nsys-rep, rep.sqlite, the `nsys stats` CSVs, split.txt, split.json, kernels.txt). Needs SL_NSYS,
SL_MODEL and SL_LLAMA_BIN. Windows only (a .bat file under `nsys launch`).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_watch import wait_shared_free  # noqa: E402
from oracle_run import ARGS, EXE, disk_delta, disk_state, need_bins, parse  # noqa: E402
from harness import RESULTS, fork_env, keep_awake  # noqa: E402
from nsys_server import NSYS, nsys  # noqa: E402

SESSION = "orclprof"


def main():
    need_bins()
    light, sv, nodes = "--light" in sys.argv, "--sv" in sys.argv, "--nodes" in sys.argv
    vram = sys.argv[sys.argv.index("--vram") + 1] if "--vram" in sys.argv else None
    argv = [a for i, a in enumerate(sys.argv) if a not in ("--light", "--sv", "--nodes", "--vram")
            and not (i > 0 and sys.argv[i - 1] == "--vram")]
    src = Path(argv[1]).resolve()
    ahead = argv[2]
    wait_s = float(argv[3]) if len(argv) > 3 else 15
    dur_s = float(argv[4]) if len(argv) > 4 else 12
    tag = (f"ahead{ahead}" + (f"-v{vram}" if vram else "") + ("-sv" if sv else "") + ("-light" if light else "")
           + ("-nodes" if nodes else ""))
    out = RESULTS / "nsys" / f"{time.strftime('%Y%m%d-%H%M%S')}-oracle-{src.name}-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    env = fork_env()
    if ahead != "0":
        env.update(LLAMA_MOE_STREAM_ORACLE=str(src / "trace-calls.txt"), LLAMA_MOE_STREAM_AHEAD=ahead)
    if vram:
        env["LLAMA_MOE_STREAM_VRAM_AHEAD"] = vram
    if sv:
        env["LLAMA_MOE_STREAM_SPEC_VRAM"] = "1"
    txt, log = out / "run.txt", out / "run.log"
    # under Windows `nsys launch` does not pass the program's output on (the logs got only "Collecting data..."),
    # so llama-completion runs through a .bat file with redirection: nsys traces the process tree
    bat = out / "run.bat"
    bat.write_text(f'@{subprocess.list2cmdline([str(EXE), *ARGS])} > "{txt}" 2> "{log}"\r\n', encoding="utf-8")
    keep_awake(True)
    wait_shared_free()   # the previous run's pinned RAM tier must be handed back first (else CUDA OOM at load)
    h0 = disk_state()
    try:
        p = subprocess.Popen([str(NSYS), "launch", f"--session-new={SESSION}", "--trace=cuda,nvtx",
                              f"--cuda-graph-trace={'node' if nodes else 'graph'}", "--cuda-memory-usage=false",
                              "cmd.exe", "/c", str(bat)],
                             cwd=str(EXE.parent), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL)
        t0 = time.time()
        while not log.exists() or "generate:" not in log.read_text(encoding="utf-8", errors="replace"):
            if p.poll() is not None or time.time() - t0 > 600:
                raise RuntimeError("llama-completion under nsys did not reach generation")
            time.sleep(0.5)
        print(f"== oracle {ahead} under nsys: loaded in {time.time() - t0:.0f} s; collecting after {wait_s:.0f} s "
              f"for {dur_s:.0f} s", flush=True)
        time.sleep(wait_s)
        # CPU sampling and context switches are set at start, not at launch (nsys 2026.5)
        nsys("start", f"--session={SESSION}", f"--output={out / 'rep'}", "--force-overwrite=true",
             *(["--sample=none", "--cpuctxsw=none"] if light else []))
        time.sleep(dur_s)
        nsys("stop", f"--session={SESSION}")
        rc = p.wait(timeout=1800)
        h1 = disk_state()
    finally:
        nsys("shutdown", f"--session={SESSION}", "--kill=true", check=False)
        keep_awake(False)
    row = {"tag": f"nsys{ahead}", "rc": rc, "env": {k: env[k] for k in env if k.startswith("LLAMA_MOE")},
           **parse(log.read_text(encoding="utf-8", errors="replace")), **disk_delta(h0, h1)}
    ref = src / "trace.txt"
    if ref.exists():
        # through `cmd >` the output has \n line ends, oracle_run.py's has \r\n: compare the text, not the line ends
        row["same_text"] = txt.read_bytes().replace(b"\r\n", b"\n") == ref.read_bytes().replace(b"\r\n", b"\n")
    (out / "run.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   exit {rc} | generation {row.get('tg_tps', float('nan')):.2f} tok/s under the profiler | wait "
          f"{row.get('stall_per_tok_ms', float('nan')):.1f} ms/tok | text "
          f"{'=' if row.get('same_text') else '!='} trace | controller {row.get('ctrl0', '?')}->"
          f"{row.get('ctrl1', '?')} °C, TMT1 +{row.get('tmt1_s', '?')} s", flush=True)
    rep = out / "rep.nsys-rep"
    if rep.exists():
        nsys("stats", "--report", "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum",
             "--format", "csv", "--output", str(out / "rep"), "--force-export=true", str(rep), check=False)
        with open(out / "split.txt", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "-I", str(HERE / "nsys_split.py"), str(out / "rep.sqlite"),
                            "--json", str(out / "split.json")], stdout=f, stderr=subprocess.STDOUT)
        print((out / "split.txt").read_text(encoding="utf-8"), flush=True)
        if nodes:
            with open(out / "kernels.txt", "w", encoding="utf-8") as f:
                subprocess.run([sys.executable, "-I", str(HERE / "nsys_kernels.py"), str(out / "rep.sqlite")],
                               stdout=f, stderr=subprocess.STDOUT)
            print((out / "kernels.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len([a for a in sys.argv if not a.startswith("--")]) < 3:
        sys.exit(__doc__)
    main()
