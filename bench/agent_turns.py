"""Agent turns: an agent session replayed against one server - reading and answering on a warm RAM tier, the way an
agent works (reading is most of an agent's time). One server for the whole series; the context grows turn by turn:
each turn reads a new chunk (a tool output: a source of ggml's CUDA backend from SL_LLAMA_SRC, 0.5-8K tokens) and a
short question, and answers with ANSWER tokens. The prompt cache is on: the prefix of the past turns is not
recomputed, only the new chunk is read (the model's answers do not enter the context - reading does not depend on
them). consult.py runs on a cold tier - a new server each run; here the first turn is cold and the next ones run on
the tier warmed by the past turns and answers.

    python bench/agent_turns.py TAG [--turns 2000,5000,1000,6000,3000,500,8000,2000] [--answer 128] [-- server flags]

The server is the consultant's up to 32K (-c 32768, batch harness.ub_for, RAM tier l2_for, expert slots slots_for one
below, KV q8_0, -b by the context, so that P4 - the fork's encoder-only prefill - gets each turn in one call), preset
default (another one: LLAMA_MOE_STREAM_PRESET in the environment). The row goes to SL_RESULTS/test4.jsonl (name TAG);
per turn: new prompt tokens, reading tok/s, the answer, GB read from the drives during the turn. The summary of the
warm turns (2 and on) is printed. Windows only, like harness.py.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import batch_for, keep_awake, kv_args, l2_for, require_gpu, run, slots_for, ub_for  # noqa: E402

LLAMA_SRC = config.path("SL_LLAMA_SRC")
SRC = LLAMA_SRC / "ggml" / "src" / "ggml-cuda" if LLAMA_SRC else None
CHARS_PER_TOKEN = 3.3
# the questions of our runs, the fourth reworded for publishing (a few tokens' difference in turns 4 and 8):
# "What does this code do? Briefly, in 2-3 sentences." / "Find possible bugs or bottlenecks in this fragment -
# briefly, as a list." / "Which data structures and kernels are the main ones here? Briefly." / "Which function
# of this fragment is the most complex, and why? Briefly."
QUESTIONS = ["Что делает этот код? Кратко, в 2–3 фразах.",
             "Найди в этом фрагменте возможные ошибки или узкие места — коротко, списком.",
             "Какие структуры данных и ядра здесь главные? Коротко.",
             "Какая функция этого фрагмента самая сложная и почему? Коротко."]
CTX = 32768


def chunks(sizes):
    """Consecutive pieces of the sources, no overlap: turn k gets the next sizes[k] tokens (by CHARS_PER_TOKEN)."""
    if SRC is None:
        sys.exit(f"set SL_LLAMA_SRC (environment or {config.FILE})")
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(SRC.glob("*.cu")))
    pos, out = 0, []
    for n in sizes:
        take = int(n * CHARS_PER_TOKEN)
        out.append(text[pos:pos + take])
        pos += take
    return out


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    if not argv:
        sys.exit(__doc__)
    tag = argv[0]
    opt = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1, 2)}
    sizes = [int(x) for x in opt.get("--turns", "2000,5000,1000,6000,3000,500,8000,2000").split(",")]
    answer = int(opt.get("--answer", 128))
    require_gpu()
    # "You help make sense of code. Below are the tool outputs so far; answer the last question." Each turn adds
    # "### Tool output k", the chunk and "Question: ..." (our runs opened with a slightly different first sentence)
    convo = "Ты помогаешь разбираться в коде. Ниже — выводы инструментов по ходу работы; отвечай на последний вопрос.\n"
    order = []
    for k, (n, body) in enumerate(zip(sizes, chunks(sizes))):
        convo += f"\n### Вывод инструмента {k + 1}\n```c\n{body}\n```\nВопрос: {QUESTIONS[k % len(QUESTIONS)]}\n"
        order.append((f"turn{k + 1}-{n}", convo, answer))
    ub = ub_for(CTX)
    args = ["-c", str(CTX), "--moe-stream-l2", str(l2_for(ub, CTX)), "--moe-stream-cache", f"{slots_for(17, margin=1)}s",
            "-b", str(batch_for(ub, CTX)), "-ub", str(ub), *kv_args(extra), "-lv", "4", *extra]
    env = {} if "LLAMA_MOE_STREAM_PRESET" in os.environ else {"LLAMA_MOE_STREAM_PRESET": "default"}
    keep_awake(True)
    try:
        row = run(tag, args, order, env_extra=env, cache_prompt=True)
    finally:
        keep_awake(False)
    reqs = row.get("requests", [])
    warm = reqs[1:]
    if warm:
        pp_n = sum(r["prompt_n"] or 0 for r in warm)
        pp_s = sum((r["prompt_n"] or 0) / (r["pp_tps"] or 1) for r in warm)
        tg_n = sum(r["n"] or 0 for r in warm)
        tg_s = sum((r["n"] or 0) / (r["tg_tps"] or 1) for r in warm)
        gb = sum(r.get("read_gb", 0) for r in warm)
        print(f"== {tag}, turns 2-{len(reqs)} (warm): read {pp_n} tok in {pp_s:.1f} s ({pp_n / max(pp_s, 1e-9):.0f} tok/s), "
              f"answers {tg_n} tok in {tg_s:.1f} s ({tg_n / max(tg_s, 1e-9):.2f} tok/s), {gb:.1f} GB from the drives; "
              f"reading is {100 * pp_s / max(pp_s + tg_s, 1e-9):.0f}% of the turns' time", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
