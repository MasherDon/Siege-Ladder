"""Prefill under Nsight Systems: what a prompt's time goes to, wave by wave.

The server in the configuration of `harness.py prefill` (`-c 12288`, RAM tier 24 GiB, -b / -ub 2048, the output head
on the CPU) starts under `nsys launch` without collecting; once it has loaded - `nsys start`, one prompt of
harness.long_prompts() without the prompt cache and with a one-token answer, `nsys stop`. The fork marks NVTX ranges:
`wave` and `wave.stall` on the graph thread (a prefill wave and its wait for loads), `read` (a demand read), `spec`
(read-ahead), `l2`, `upload.wait` and `upload` on the I/O threads. bench/nsys_waves.py analyses the window into
waves.txt next to the report.

    python bench/prefill_nsys.py [2k|8k] [UB|auto] [--nodes] [--head-gpu] [-- server flags]
    python bench/prefill_nsys.py 2k 2048

--nodes - trace every kernel inside CUDA graphs; the kernels outside graphs show without it anyway.
--head-gpu - today's configuration of `harness.py prefill auto --head-gpu`: the output head on the GPU (P1 - the head
yields its VRAM to a prompt; P11 - the expert cache does too, and the prompt's waves go through a shared slot pool),
the micro-batch by length (`auto` - harness.ub_for), -b, the RAM tier and the cache slots as in harness (batch_for,
l2_for, slots_for with a margin of one), KV q8_0.

The server runs through a .bat file with redirection: under Windows `nsys launch` does not pass the program's output on
(a server that crashed under the profiler left only "Collecting data..." in the log). Windows only.

Results: SL_RESULTS/nsys/<time>-prefill-<2k|8k>-ub<UB>/ (rep.nsys-rep, rep.sqlite, the `nsys stats` CSVs, waves.txt);
the server log goes to SL_RESULTS/logs. Needs SL_NSYS (nsys.exe) and SL_LLAMA_SRC (the prompts are cut from its ggml.c).
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import (LLAMA, LOGS, PORT, RESULTS, ask, batch_for, common, fork_env, get_json, keep_awake,  # noqa: E402
                     kv_args, l2_for, long_prompts, port_busy, slots_for, ub_for)

NSYS = config.need("SL_NSYS")   # nsys.exe of Nsight Systems
SESSION = "ppprof"


def nsys(*args, check=True):
    r = subprocess.run([str(NSYS), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"   nsys {args[0]}: {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"nsys {args[0]} -> {r.returncode}")
    return r


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    nodes, head_gpu = "--nodes" in argv, "--head-gpu" in argv
    argv = [x for x in argv if x not in ("--nodes", "--head-gpu")]
    which = argv[0] if argv else "2k"
    ub = argv[1] if len(argv) > 1 else "2048"
    if ub == "auto":
        ub = str(ub_for(12288))
    label, text, _ = {"2k": long_prompts()[0], "8k": long_prompts()[1]}[which]
    base = common()
    if port_busy():
        sys.exit(f"something already answers on port {PORT} - profile not started")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / "nsys" / f"{stamp}-prefill-{which}-ub{ub}"
    out.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{stamp}-t4-nsys-prefill-{which}-ub{ub}.log"
    if head_gpu:
        args = [*base, "-c", "12288", "--moe-stream-l2", str(l2_for(int(ub), 12288)), "--moe-stream-cache",
                f"{slots_for(17, margin=1)}s", "-b", str(batch_for(int(ub), 12288)), "-ub", ub, *kv_args(extra),
                "-lv", "4", *extra]
    else:
        args = [*base, "-c", "12288", "--moe-stream-l2", "24", "-b", str(max(int(ub), 2048)), "-ub", ub,
                "-ot", r"output\.weight=CPU", "-lv", "4", *extra]
    keep_awake(True)
    bat = out / "run.bat"
    bat.write_text(f'@{subprocess.list2cmdline([str(LLAMA), *args])} > "{log}" 2>&1\r\n', encoding="utf-8")
    launcher = subprocess.Popen([str(NSYS), "launch", f"--session-new={SESSION}", "--trace=cuda,nvtx",
                                 f"--cuda-graph-trace={'node' if nodes else 'graph'}", "--cuda-memory-usage=false",
                                 "cmd.exe", "/c", str(bat)], cwd=str(LLAMA.parent), env=fork_env(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while True:
            try:
                if get_json("/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            if time.time() - t0 > 600:
                raise TimeoutError("the server under nsys did not come up")
            time.sleep(1)
        print(f"== prefill {which}, -ub {ub} under nsys: loaded in {time.time() - t0:.0f} s", flush=True)
        nsys("start", f"--session={SESSION}", f"--output={out / 'rep'}", "--force-overwrite=true",
             "--sample=none", "--cpuctxsw=none")
        r = ask(text, 1, cache_prompt=False)
        nsys("stop", f"--session={SESSION}")
        print(f"   under the profiler: prompt {r['prompt_n']} tok, {r['pp_tps'] or 0:.2f} tok/s", flush=True)
    finally:
        nsys("shutdown", f"--session={SESSION}", "--kill=true", check=False)
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/T", "/F"], capture_output=True)
        try:
            launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            launcher.kill()
        keep_awake(False)
    rep = out / "rep.nsys-rep"
    if rep.exists():
        nsys("stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum",
             "--format", "csv", "--output", str(out / "rep"), "--force-export=true", str(rep), check=False)
        with open(out / "waves.txt", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "-I", str(HERE / "nsys_waves.py"), str(out / "rep.sqlite")],
                           stdout=f, stderr=subprocess.STDOUT)
        print((out / "waves.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
