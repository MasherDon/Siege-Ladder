"""One agent turn under Nsight Systems: where the time of reading a warm turn goes, layer by layer.

The server is set up as in bench/agent_turns.py (`-c 32768`, the micro-batch of harness.ub_for, the RAM tier of l2_for,
the slots of slots_for with a margin of one, KV q8_0, -b by context; the `default` preset, which includes C7 - the YoCo
prefill, where the decoder half of the model computes only the last 140 prompt positions) and starts under
`nsys launch` without collecting. Turns 1..K-1 run unrecorded (the RAM tier warms up, the prompt cache keeps the
prefix, answers of ANSWER tokens as in agent_turns); turn K is read under `nsys start` with a one-token answer.
Analysis: bench/nsys_waves.py (window, drive, copies) -> waves.txt and bench/nsys_layers.py (per layer: before the
waves - attention, router, norms; the waves - experts and the wait for them) -> layers.txt.

    python bench/turn_nsys.py [--turn 7] [--turns 2000,5000,1000,6000,3000,500,8000,2000] [--answer 128] [--nodes]
                              [-- server flags]

--nodes - trace every kernel inside CUDA graphs. Another preset: LLAMA_MOE_STREAM_PRESET in the environment.
Results: SL_RESULTS/nsys/<time>-turn<K>-<size>/ (rep.nsys-rep, rep.sqlite, the `nsys stats` CSVs, waves.txt,
layers.txt); the server log goes to SL_RESULTS/logs. Needs SL_NSYS and SL_LLAMA_SRC (the turns are cut from its
ggml-cuda sources). Windows only (a .bat file under `nsys launch`, taskkill).
"""
import os
import subprocess
import sys
import time

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_turns import CTX, QUESTIONS, chunks  # noqa: E402
from prefill_nsys import NSYS, nsys  # noqa: E402
from harness import (LLAMA, LOGS, PORT, RESULTS, ask, batch_for, common, fork_env, get_json, keep_awake,  # noqa: E402
                     kv_args, l2_for, port_busy, require_gpu, slots_for, ub_for)

SESSION = "turnprof"


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    nodes = "--nodes" in argv
    argv = [x for x in argv if x != "--nodes"]
    opt = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
    sizes = [int(x) for x in opt.get("--turns", "2000,5000,1000,6000,3000,500,8000,2000").split(",")]
    turn = int(opt.get("--turn", 7))
    answer = int(opt.get("--answer", 128))
    if not 1 <= turn <= len(sizes):
        sys.exit(f"--turn {turn}: there are {len(sizes)} turns")
    base = common()
    require_gpu()
    if port_busy():
        sys.exit(f"something already answers on port {PORT} - profile not started")
    # the conversation of agent_turns.py (Russian): "You help make sense of code. Below are the tool outputs so far;
    # answer the last question." Each turn appends "### Tool output k", a chunk of C and "Question: ..."
    convo = "Ты помогаешь разбираться в коде. Ниже — выводы инструментов по ходу работы; отвечай на последний вопрос.\n"
    turns = []
    for k, (n, body) in enumerate(zip(sizes[:turn], chunks(sizes[:turn]))):
        convo += f"\n### Вывод инструмента {k + 1}\n```c\n{body}\n```\nВопрос: {QUESTIONS[k % len(QUESTIONS)]}\n"
        turns.append((f"turn{k + 1}-{n}", convo))
    ub = ub_for(CTX)
    args = [*base, "-c", str(CTX), "--moe-stream-l2", str(l2_for(ub, CTX)), "--moe-stream-cache",
            f"{slots_for(17, margin=1)}s", "-b", str(batch_for(ub, CTX)), "-ub", str(ub), *kv_args(extra), "-lv", "4",
            *extra]
    env = fork_env({} if "LLAMA_MOE_STREAM_PRESET" in os.environ else {"LLAMA_MOE_STREAM_PRESET": "default"})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / "nsys" / f"{stamp}-turn{turn}-{sizes[turn - 1]}"
    out.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{stamp}-t4-nsys-turn{turn}.log"
    keep_awake(True)
    bat = out / "run.bat"
    bat.write_text(f'@{subprocess.list2cmdline([str(LLAMA), *args])} > "{log}" 2>&1\r\n', encoding="utf-8")
    launcher = subprocess.Popen([str(NSYS), "launch", f"--session-new={SESSION}", "--trace=cuda,nvtx",
                                 f"--cuda-graph-trace={'node' if nodes else 'graph'}", "--cuda-memory-usage=false",
                                 "cmd.exe", "/c", str(bat)], cwd=str(LLAMA.parent), env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
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
        print(f"== turn {turn} under nsys: loaded in {time.time() - t0:.0f} s", flush=True)
        for label, text in turns[:-1]:
            r = ask(text, answer, cache_prompt=True)
            print(f"   {label} (not recorded): read {r['prompt_n']} tok, {r['pp_tps'] or 0:.1f} tok/s; answer "
                  f"{r['n']} tok, {r['tg_tps'] or 0:.2f} tok/s", flush=True)
        nsys("start", f"--session={SESSION}", f"--output={out / 'rep'}", "--force-overwrite=true",
             "--sample=none", "--cpuctxsw=none")
        r = ask(turns[-1][1], 1, cache_prompt=True)
        nsys("stop", f"--session={SESSION}")
        print(f"   {turns[-1][0]} under the profiler: read {r['prompt_n']} tok, {r['pp_tps'] or 0:.1f} tok/s",
              flush=True)
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
        for script, name in (("nsys_waves.py", "waves.txt"), ("nsys_layers.py", "layers.txt")):
            with open(out / name, "w", encoding="utf-8") as f:
                subprocess.run([sys.executable, "-I", str(HERE / script), str(out / "rep.sqlite")],
                               stdout=f, stderr=subprocess.STDOUT)
            print((out / name).read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
