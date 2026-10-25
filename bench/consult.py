"""Consultant scenario: one long document and several questions about it - the main use of the model here.

Measured with the fork's server in the harness configuration:
  1. whether a context of CTX tokens fits the 16 GB card with the expert slots of cache_args and the micro-batch UB
     (VRAM after load and at the end); the document goes first as a request of its own (D9, "document first"),
     --no-doc-first sends the questions with the document right away;
  2. the prefill of the long context and the generation right after it - decode speed at a long context;
  3. the prompt cache: the same context with a second question and then again with the first - how many tokens the
     server recomputed (prompt_n) and what it logged about the cache (slot choice by common prefix, checkpoints, full
     re-processing).

The document is real C / C++ from the patched tree's sources (SL_LLAMA_SRC: ggml first; the longest documents go on
into ggml's CUDA backend and llama.cpp's own sources), ~3.1 characters a token. The run's row goes to
SL_RESULTS/test4.jsonl (as harness.py writes it); the log's prompt-cache lines, the memory figures and the request
timings to SL_RESULTS/consult.jsonl.

    python bench/consult.py TAG CTX UB TOKENS [server flags...]
    python bench/consult.py c32 32768 auto 30000        # UB auto: by context length (harness.ub_for), tier l2_for
    python bench/consult.py c32-nop11 32768 1536 30000  # with LLAMA_MOE_STREAM_P11=0 in the environment: no P11

Back to a context without a prefill: the context with no question -> the slot to disk (--slot-save-path, SL_SLOTS_DIR)
-> a new server -> the slot from disk -> a question. The DSV4 cache does not trim its tail (seq_rm works only inside
the rollback window) and the server's checkpoints do not go into the slot file, so what was saved must be exactly a
prefix of the next prompt: the prompts are raw, through /completion, with the chat template from /apply-template.
The row goes to SL_RESULTS/consult.jsonl (mode "slot" or "restore").

    python bench/consult.py slot TAG CTX UB TOKENS [server flags...]
    python bench/consult.py slot s2k 32768 512 2000
    python bench/consult.py restore TAG CTX UB TOKENS [server flags...]   # the second server only: slot TAG from
                                                                          # disk and both questions (A/B of decode)

Labels of the fork's techniques: P11 - the expert cache yields VRAM to a long prompt, which runs waves of all layers
through a shared pool of 64 experts; D9 - the document as a request of its own; D10 - sparse attention in prefill.
Windows only, like harness.py.
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import (LLAMA, LOGS, PORT, RESULTS, batch_for, common, fork_env, keep_awake, kv_args, l2_for,  # noqa: E402
                     p11_on, port_busy, require_gpu, run, slots_for, stop, ub_for, wait_ready)
from gpu_watch import wait_shared_free  # noqa: E402
from mem_watch import MemWatch  # noqa: E402

SLOT_DIR = config.SLOTS_DIR

LLAMA_SRC = config.path("SL_LLAMA_SRC")
SRC = LLAMA_SRC / "ggml" / "src" if LLAMA_SRC else None
FILES = ["ggml.c", "ggml-quants.c", "ggml-cpu/ggml-cpu.c", "ggml-cpu/ops.cpp"]
# The four files hold ~300K tokens (the 256K document is ~234K); for 512K / 1M the document goes on into the sources of
# ggml's CUDA backend and of llama.cpp itself, by name. Documents up to ~300K come from the four files alone
if SRC is not None:
    FILES += sorted(str(p.relative_to(SRC)).replace("\\", "/") for p in (SRC / "ggml-cuda").glob("*.cu"))
    FILES += sorted("../../src/" + p.name for p in (LLAMA_SRC / "src").glob("*.cpp"))
CHARS_PER_TOKEN = 3.1
# kept as measured: "Find the function in this code that quantizes a block of floats to an 8-bit format in blocks of
# 32, name it and explain in 3-4 sentences how it works."
Q1 = ("Найди в этом коде функцию, которая квантует блок чисел float в 8-битный формат блоками по 32, назови её и в "
      "3–4 фразах объясни, как она работает.")
# "Which structure describes a tensor in this code? List its main fields and what they are for, in 4-5 sentences."
Q2 = "Какая структура описывает тензор в этом коде? Перечисли её главные поля и их назначение, в 4–5 фразах."
CACHE_LINES = re.compile(r"LCP similarity|checkpoint|re-processing|cache_reuse|n_past|prompt cache|memory_seq_rm|"
                         r"cached n_tokens|erased invalidated", re.I)


def context(n_tokens):
    if SRC is None:
        sys.exit(f"set SL_LLAMA_SRC (environment or {config.FILE})")
    budget = int(n_tokens * CHARS_PER_TOKEN)
    parts, size = [], 0
    for name in FILES:
        text = (SRC / name).read_text(encoding="utf-8", errors="replace")
        take = text[:budget - size]
        parts.append(f"// ===== {name} =====\n{take}")
        size += len(take)
        if size >= budget:
            break
    # kept as measured: "Below are the sources of a large C program."
    return "Ниже — исходники большой программы на C.\n```c\n" + "\n".join(parts) + "\n```\n\n"


def env_rec():
    """The fork's switches in the environment: the server inherits them, so the result row records them."""
    return {k: v for k, v in fork_env().items()
            if k.startswith(("LLAMA_MOE_STREAM_", "LLAMA_DSV4", "LLAMA_HEAD_YIELD", "OMP_"))}


def l2_args(extra, ctx, ub):
    """RAM tier, unless the server flags carry their own --moe-stream-l2: harness.l2_for - 46 GiB up to a batch of
    2048; above it (P11) 44 up to 64K and 43 beyond and at a batch of 8192 (the batch's inputs share the WDDM budget
    of shared memory with the tier). 46 rather than 24 made decode at 30K 12-15% faster for a 1.7% slower prefill. At
    64K / 128K the pinned compute buffer of P1 (the output head yields its VRAM to a long prompt) on top of a 46 GiB
    tier ran past the WDDM budget; the fork now trims the tier to the budget minus 640 MiB by itself."""
    return [] if "--moe-stream-l2" in extra else ["--moe-stream-l2", str(l2_for(ub, ctx))]


def cache_args(extra, ctx):
    """Expert slots on the card, unless the server flags carry their own --moe-stream-cache (COMMON has the decode
    value for a short context; the server takes the last flag): the VRAM budget of 17 slots of the first build up to
    32K, 16 up to 64K and 15 beyond (harness.slots_for), one slot less (margin=1): cd 22 / 21 / 20, qd 24 / 23 / 22.
    Each step is what still loaded on the first build: at 64K it did not load with 18 slots, and with 17 at a batch of
    1536 the CUDA pool was ~60 MiB short; at 128K it held 16 with KV f16 or q4 and 15 with q8 (15 with P11 too)."""
    if "--moe-stream-cache" in extra:
        return []
    return ["--moe-stream-cache", f"{slots_for(17 if ctx <= 32768 else 16 if ctx <= 65536 else 15, margin=1)}s"]


def ub_arg(s, ctx):
    """The micro-batch from the command line: a number, or auto - by context length (harness.ub_for: with P11 8192 up
    to 32K with D10, otherwise 6144, then 4608 up to 64K and 3072 beyond)."""
    return ub_for(ctx) if s == "auto" else int(s)


def main():
    tag, ctx, n_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[4])
    ub = ub_arg(sys.argv[3], ctx)
    extra = sys.argv[5:]
    require_gpu()
    body = context(n_tokens)
    order = [("q1", body + Q1, 128), ("q2", body + Q2, 128), ("q1-again", body + Q1, 32)]
    # The document as a request of its own (D9, the default): the server puts a checkpoint at its end, so each question
    # recomputes only its own tail, not the batch up to the end of the previous prompt. --no-doc-first sends the
    # questions with the document right away; --doc-first is still accepted
    doc_first = "--no-doc-first" not in extra
    extra = [a for a in extra if a not in ("--doc-first", "--no-doc-first")]
    if doc_first:
        order = [("doc", body, 1), *order]
    # KV q8_0 by default (harness.kv_args); f16 with an explicit -ctk f16 -ctv f16
    args = ["-c", str(ctx), *l2_args(extra, ctx, ub), *cache_args(extra, ctx), *kv_args(extra), "-b", str(batch_for(ub, ctx)),
            "-ub", str(ub), "-lv", "4", *extra]
    print(f"== consult {tag}: -c {ctx}, -ub {ub}, P11 {'on' if p11_on(fork_env()) else 'off'}", flush=True)
    keep_awake(True)
    try:
        # the driver hands back the pinned tier of the previous server only after that server has exited
        shared_wait = wait_shared_free()
        with MemWatch() as mw:
            row = run(f"consult-{tag}", args, order)
    finally:
        keep_awake(False)
    lines = [ln.strip()[:300] for ln in (LOGS / row["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
             if CACHE_LINES.search(ln)]
    with open(RESULTS / "consult.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": row["time"], "tag": tag, "ctx": ctx, "ub": ub, "tokens": n_tokens, "args": args,
                            "env": env_rec(),
                            "vram_mib": row.get("vram_mib"), "vram_mib_end": row.get("vram_mib_end"),
                            "shared_wait": shared_wait, **mw.result(),
                            "error": row.get("error"),
                            "requests": [{k: r.get(k) for k in ("label", "prompt_n", "pp_tps", "n", "tg_tps", "wall_s")}
                                         for r in row["requests"]],
                            "cache_log": lines[-40:]}, ensure_ascii=False) + "\n")
    print("   prompt-cache lines of the log:")
    for ln in lines[-25:]:
        print("     " + ln)


def post(path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, round(time.time() - t0, 2)


def complete(label, prompt, n_predict):
    """A raw prompt through /completion; n_predict 0 only puts the prompt into the cache."""
    resp, wall = post("/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 1,
                                       "cache_prompt": True})
    t = resp.get("timings") or {}
    r = {"label": label, "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "n": t.get("predicted_n"),
         "tg_tps": t.get("predicted_per_second"), "wall_s": wall, "tokens_cached": resp.get("tokens_cached"),
         "text": (resp.get("content") or "").strip()}
    print(f"   {label:12s} recomputed {r['prompt_n']} tok in {(r['prompt_ms'] or 0) / 1000:.1f} s | generated "
          f"{r['n']} tok, {r['tg_tps'] or 0:.2f} tok/s | total {wall:.1f} s", flush=True)
    return r


def slot_action(action, filename):
    resp, wall = post(f"/slots/0?action={action}", {"filename": filename})
    print(f"   slot {action}: {wall:.1f} s | {json.dumps(resp, ensure_ascii=False)[:240]}", flush=True)
    return {"action": action, "wall_s": wall, **resp}


def server(name, args):
    """The fork's server with the harness flags and args; returns (process, log)."""
    base = common()
    LOGS.mkdir(parents=True, exist_ok=True)
    wait_shared_free()
    log = LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}-t4-{name}.log"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([str(LLAMA), *base, *args], cwd=str(LLAMA.parent), env=fork_env(), stdout=lf,
                                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    print(f"== {name}: loaded in {wait_ready(proc, log):.0f} s", flush=True)
    return proc, log


def split_template(body):
    """The chat prompt with the first question -> (the part before the question, the template's tail after it)."""
    fmt = post("/apply-template", {"messages": [{"role": "user", "content": body + Q1}]})[0]["prompt"]
    i = fmt.index(Q1)
    return fmt[:i], fmt[i + len(Q1):]


def slot_main(restore_only=False):
    tag, ctx, n_tokens = sys.argv[2], int(sys.argv[3]), int(sys.argv[5])
    ub = ub_arg(sys.argv[4], ctx)
    require_gpu()
    if port_busy():
        sys.exit(f"something already answers on port {PORT} - nothing measured")
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"consult-{tag}.bin"
    if restore_only and not (SLOT_DIR / fname).exists():
        sys.exit(f"no slot file {SLOT_DIR / fname}: run consult.py slot {tag} ... first")
    body = context(n_tokens)
    args = ["-c", str(ctx), *l2_args(sys.argv[6:], ctx, ub), *kv_args(sys.argv[6:]), "-b", str(batch_for(ub, ctx)), "-ub", str(ub),
            "-lv", "4", "--slot-save-path", str(SLOT_DIR), *sys.argv[6:]]
    row = {"time": time.strftime("%Y%m%d-%H%M%S"), "tag": tag, "mode": "restore" if restore_only else "slot",
           "ctx": ctx, "ub": ub, "tokens": n_tokens, "args": args,
           "env": env_rec(),
           "steps": [], "logs": []}
    keep_awake(True)
    try:
        if not restore_only:
            proc, log = server(f"consult-{tag}-save", args)
            row["logs"].append(log.name)
            try:
                prefix, suffix = split_template(body)
                row["template_suffix"] = suffix
                row["steps"].append(complete("context", prefix, 0))
                row["steps"].append(slot_action("save", fname))
                row["steps"].append(complete("q1", prefix + Q1 + suffix, 128))
                # the same slot back in the same process: is the restore itself exact, apart from another process
                row["steps"].append(slot_action("restore", fname))
                row["steps"].append(complete("q1-same", prefix + Q1 + suffix, 128))
            finally:
                stop(proc)
        row["slot_file_mb"] = round((SLOT_DIR / fname).stat().st_size / 2**20, 1)
        print(f"   slot file {row['slot_file_mb']} MiB", flush=True)
        proc, log = server(f"consult-{tag}-restore", args)
        row["logs"].append(log.name)
        try:
            if restore_only:
                prefix, suffix = split_template(body)
            row["steps"].append(slot_action("restore", fname))
            row["steps"].append(complete("q2", prefix + Q2 + suffix, 128))
            row["steps"].append(slot_action("restore", fname))
            row["steps"].append(complete("q1-restored", prefix + Q1 + suffix, 128))
        finally:
            stop(proc)
    except Exception as e:   # noqa: BLE001 - a failed run still gets its row
        row["error"] = str(e)
        print("   ERROR:", e, flush=True)
    finally:
        keep_awake(False)
    q1 = {s["label"]: s["text"] for s in row["steps"] if s.get("label") in ("q1", "q1-same", "q1-restored")}
    for other, key in (("q1-same", "same_q1_in_process"), ("q1-restored", "same_q1")):
        if "q1" in q1 and other in q1:
            row[key] = q1["q1"] == q1[other]
            print(f"   first question, {other}: {'the same' if row[key] else 'a different'} answer", flush=True)
    for s in row["steps"]:
        if "text" in s:
            s["text"] = s["text"][:800]
    with open(RESULTS / "consult.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1:2] in (["slot"], ["restore"]) and len(sys.argv) >= 6:
        slot_main(restore_only=sys.argv[1] == "restore")
    elif len(sys.argv) >= 5 and sys.argv[1] not in ("slot", "restore"):
        main()
    else:
        sys.exit(__doc__)
