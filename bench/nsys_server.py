"""Decode under Nsight Systems (the profiled counterpart of harness.py): where the time of a generated token goes.

The server starts under `nsys launch` (not collecting yet) and is warmed up with a 16-token request of the code prompt
(prompts.PROMPTS["en-code"]: the prompt goes into the prompt cache, the experts into the VRAM and RAM caches); then
`nsys start` -> the same request for TOKENS tokens (pure generation, the prompt comes from the cache) -> `nsys stop`.
The summaries of `nsys stats` are written as CSV next to the report.

    python bench/nsys_server.py [L2_GiB [TOKENS [graph|node]]]   # defaults: RAM tier 24 GiB, 48 tokens, graph
                                                                 # node - trace every kernel inside the CUDA graphs

Report: SL_RESULTS/nsys/<time>-decode-l2-<L2>-<graph|node>.nsys-rep and its CSV summaries; the server log goes to
SL_RESULTS/logs. Needs SL_NSYS (nsys.exe of Nsight Systems). Windows only (taskkill, the server's Windows build).
Other profiling scripts import NSYS and nsys() from here.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import LLAMA, LOGS, PORT, RESULTS, ask, common, fork_env, get_json, keep_awake, port_busy  # noqa: E402
from prompts import PROMPTS  # noqa: E402

NSYS = config.need("SL_NSYS")   # nsys.exe of Nsight Systems
SESSION = "t4prof"


def nsys(*args, check=True):
    r = subprocess.run([str(NSYS), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"   nsys {args[0]}: {(r.stdout + r.stderr).strip()[:400]}", flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"nsys {args[0]} -> {r.returncode}")
    return r


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    l2 = sys.argv[1] if len(sys.argv) > 1 else "24"
    n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    graph_trace = sys.argv[3] if len(sys.argv) > 3 else "graph"   # node - the kernels inside CUDA graphs too
    base = common()
    if port_busy():
        sys.exit(f"something already answers on port {PORT} - profile not started")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / "nsys" / f"{stamp}-decode-l2-{l2}-{graph_trace}"
    out.parent.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{stamp}-t4-nsys-l2-{l2}.log"
    keep_awake(True)
    lf = open(log, "w", encoding="utf-8")
    launcher = subprocess.Popen([str(NSYS), "launch", f"--session-new={SESSION}", "--trace=cuda,nvtx",
                                 f"--cuda-graph-trace={graph_trace}", "--cuda-memory-usage=false", str(LLAMA), *base,
                                 "-c", "4096", "--moe-stream-l2", l2], cwd=str(LLAMA.parent), env=fork_env(),
                                stdout=lf, stderr=subprocess.STDOUT)
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
        print(f"== nsys, L2 {l2} GiB: loaded in {time.time() - t0:.0f} s", flush=True)
        r = ask(PROMPTS["en-code"], 16)
        print(f"   warm-up: prompt {r['prompt_n']} tok, generated {r['n']} tok at {r['tg_tps'] or 0:.2f} tok/s",
              flush=True)
        nsys("start", f"--session={SESSION}", f"--output={out}", "--force-overwrite=true")
        r = ask(PROMPTS["en-code"], n_tok)
        nsys("stop", f"--session={SESSION}")
        print(f"   under the profiler: prompt {r['prompt_n']} tok, generated {r['n']} tok at "
              f"{r['tg_tps'] or 0:.2f} tok/s", flush=True)
    finally:
        nsys("shutdown", f"--session={SESSION}", "--kill=true", check=False)
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/T", "/F"], capture_output=True)
        try:
            launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            launcher.kill()
        lf.close()
        keep_awake(False)
    rep = out.with_suffix(".nsys-rep")
    if rep.exists():
        nsys("stats", "--report", "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
             "--format", "csv", "--output", str(out), "--force-export=true", str(rep), check=False)
        print("   report:", rep, flush=True)


if __name__ == "__main__":
    main()
