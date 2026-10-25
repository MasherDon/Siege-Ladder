"""HumanEval pass@1 of the local build against the DeepSeek API: quality measured by tasks, not by token agreement.

164 Python tasks with tests (openai/human-eval, data/HumanEval.jsonl.gz - not included: download it and set
SL_HUMANEVAL to the file, or pass --data PATH), thinking off, greedy (temperature 0), one user message with no system
prompt - a case the GGUF's chat template renders exactly as DeepSeek's own encoder does. The answer is the whole
function in a ```python block; the prompt's imports, the task's test and check(entry_point) are added to it, and each
program runs in its own `python -I` in a temporary folder with a timeout. Scoring runs model-written code on this
machine (idle priority, 15 s timeout, no other isolation).

    python bench/humaneval.py api OUT                              # deepseek-flash, thinking off (key: api_collect.py)
    python bench/humaneval.py local OUT --preset default [--k 5]   # the patched llama-server, RAM tier 46 GiB, SL_MODEL
                                                                   # (cd: 23 VRAM slots by harness.slots_for)
    SL_MODEL=<qd part 1> SL_BUILD=qd python bench/humaneval.py local OUT --preset fast   # fast: the qd build, 25 slots
    python bench/humaneval.py local OUT --preset default --env LLAMA_MOE_STREAM_G35=0  # one fork switch over the preset
    python bench/humaneval.py score OUT                            # run the tests, pass@1 -> OUT/score.json

--k N routes each token to N experts instead of the model's 6 (lossy); --slots N overrides the VRAM expert slots;
--env KEY=VALUE sets one fork switch on top of the preset (e.g. LLAMA_MOE_STREAM_G35=0 turns off G35, which reads the
heavy experts a cheap router lens predicts into RAM two layers early). --data PATH works in every mode.
OUT - the run's folder: answers.jsonl (answers and timings; a run resumes past the tasks already there), score.json
(the result), fails.txt (what failed), server.log (local runs). Windows only as written (the server's process group).
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import harness  # noqa: E402
import api_collect  # noqa: E402

# HumanEval.jsonl.gz of openai/human-eval: SL_HUMANEVAL or --data PATH
DATA = config.path("SL_HUMANEVAL") or config.RESULTS / "humaneval" / "data" / "HumanEval.jsonl.gz"
MAX_TOKENS = 1024
PY = sys.executable


def tasks():
    if not DATA.is_file():
        sys.exit(f"no HumanEval data at {DATA}: get data/HumanEval.jsonl.gz of openai/human-eval and set SL_HUMANEVAL "
                 f"(environment or {config.FILE}) or pass --data PATH")
    return [json.loads(line) for line in gzip.open(DATA, "rt", encoding="utf-8")]


def prompt_of(t):
    return ("Complete the following Python function. Reply with the complete function (signature and body, plus any "
            "helper code it needs) in a single ```python code block, without explanations.\n\n```python\n"
            + t["prompt"] + "```")


def done(out):
    f = out / "answers.jsonl"
    return {json.loads(line)["task_id"] for line in f.open(encoding="utf-8")} if f.exists() else set()


def record(out, row):
    with open(out / "answers.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def api(out):
    out.mkdir(parents=True, exist_ok=True)
    key, have = api_collect.api_key(), done(out)
    for t in tasks():
        if t["task_id"] in have:
            continue
        body = {"model": api_collect.MODEL, "messages": [{"role": "user", "content": prompt_of(t)}],
                "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False, "thinking": {"type": "disabled"}}
        req = urllib.request.Request(api_collect.ENDPOINT, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.load(r)
                break
            except OSError as e:   # network / 429 / 5xx - retry
                if attempt == 3:
                    raise
                print(t["task_id"], "retry:", e, flush=True)
                time.sleep(5 * (attempt + 1))
        record(out, {"task_id": t["task_id"], "text": resp["choices"][0]["message"].get("content") or "",
                     "usage": resp.get("usage"), "model": resp.get("model"),
                     "fingerprint": resp.get("system_fingerprint"), "wall_s": round(time.time() - t0, 1)})
    print("api: done", len(done(out)), flush=True)


def local(out, preset, k, slots=17, env_extra=None):
    base = harness.common()   # -m SL_MODEL and the common server flags; exits when SL_MODEL / SL_LLAMA_BIN are unset
    out.mkdir(parents=True, exist_ok=True)
    if harness.port_busy():
        sys.exit(f"something already answers on port {harness.PORT}")
    harness.require_gpu()
    extra = ["-c", "8192", "--moe-stream-cache", f"{slots}s", "--moe-stream-l2", "46", "--moe-stream-io-threads", "17"]
    if k:
        extra += ["--override-kv", f"deepseek41.expert_used_count=int:{k}"]
    env = harness.fork_env({"LLAMA_MOE_STREAM_PRESET": preset, "OMP_WAIT_POLICY": "PASSIVE", **(env_extra or {})})
    log = out / "server.log"
    with open(log, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen([str(harness.LLAMA), *base, *extra], cwd=str(harness.LLAMA.parent), env=env,
                                stdout=lf, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        print(f"loaded in {harness.wait_ready(proc, log):.0f} s", flush=True)
        have = done(out)
        for t in tasks():
            if t["task_id"] in have:
                continue
            r = harness.ask(prompt_of(t), MAX_TOKENS, cache_prompt=False)
            record(out, {"task_id": t["task_id"], "preset": preset, "k": k, **r})
            print(f"{t['task_id']}: {r['n']} tok, {r['tg_tps'] or 0:.2f} tok/s", flush=True)
    finally:
        harness.stop(proc)


def code_of(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def score(out):
    by = {t["task_id"]: t for t in tasks()}
    answers = {}
    for line in (out / "answers.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        answers[row["task_id"]] = row
    passed, fails = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for tid in sorted(answers, key=lambda s: int(s.split("/")[1])):
            t = by[tid]
            imports = "\n".join(ln for ln in t["prompt"].splitlines() if ln.startswith(("import ", "from ")))
            prog = imports + "\n\n" + code_of(answers[tid]["text"]) + "\n\n" + t["test"] + f"\n\ncheck({t['entry_point']})\n"
            f = Path(tmp) / "prog.py"
            f.write_text(prog, encoding="utf-8")
            try:
                # idle priority: stay out of the way of measurements running alongside
                r = subprocess.run([PY, "-I", str(f)], cwd=tmp, capture_output=True, timeout=15,
                                   creationflags=getattr(subprocess, "IDLE_PRIORITY_CLASS", 0))
                ok = r.returncode == 0
                why = (r.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1][:200]
            except subprocess.TimeoutExpired:
                ok, why = False, "timeout"
            passed += ok
            if not ok:
                fails.append(f"{tid}: {why}")
    res = {"n": len(answers), "passed": passed, "pass_at_1": round(passed / max(len(answers), 1), 4)}
    (out / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "fails.txt").write_text("\n".join(fails) + "\n", encoding="utf-8")
    print(f"{out.name}: {passed} of {len(answers)} ({100 * res['pass_at_1']:.1f}%)", flush=True)


def main():
    global DATA
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, out = sys.argv[1], Path(sys.argv[2])
    args = sys.argv[3:]
    opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
    if "--data" in opt:
        DATA = Path(opt["--data"])
    if mode == "api":
        api(out)
    elif mode == "local":
        # --slots N (the builds with ISTA-DASLab's lighter dense part hold more slots: cd 23, qd 25), --env KEY=VALUE
        # (one fork switch on top of the preset, e.g. LLAMA_MOE_STREAM_G35=2); the model is SL_MODEL, as in harness.py
        env_extra = dict([opt["--env"].split("=", 1)]) if "--env" in opt else {}
        local(out, opt.get("--preset", "default"), int(opt["--k"]) if "--k" in opt else 0,
              int(opt.get("--slots", harness.slots_for(17))), env_extra)
    elif mode == "score":
        score(out)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
