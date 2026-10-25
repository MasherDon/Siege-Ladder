"""Server benchmark harness: start the patched llama-server, send requests, record speeds and the fork's counters.

    python bench/harness.py run NAME -- <server flags>    # one configuration with the DECODE prompt set
    python bench/harness.py ladder [TAG]                  # RAM tier of 24 / 0 / 16 / 32 / 40 GiB, the DECODE set
    python bench/harness.py prefill [TAG [UB,...|auto [flags...]]]   # long prompts (~2K and ~8K tokens)
    python bench/harness.py prefill p11 auto --head-gpu   # output head on the GPU: the batch and RAM tier by length
    python bench/harness.py report                        # a table of SL_RESULTS/test4.jsonl

Paths come from bench/config.py. Requests are greedy (temperature 0, seed 1), thinking off. Every run appends a row to
SL_RESULTS/test4.jsonl: the server flags, the fork's environment switches, load time, VRAM, and per request the prompt
and answer token counts and speeds (the server's own timings), bytes the process read from disk, pages faulted in,
and the delta of the fork's cumulative counters (print_stats: VRAM cache hits and misses, the wait for expert loads,
RAM-tier hits, prefill waves). Windows only (CTRL_BREAK to stop the server, PowerShell for the I/O counters).
"""

import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from prompts import CEIL, PROMPTS  # noqa: E402

LLAMA = config.exe("llama-server")
MODEL = config.MODEL
MIRRORS = config.MIRRORS
RESULTS = config.RESULTS
LOGS = config.LOGS
PORT = config.PORT
LLAMA_SRC = config.path("SL_LLAMA_SRC")
LONG_SRC = LLAMA_SRC / "ggml" / "src" / "ggml.c" if LLAMA_SRC else None

# VRAM expert slots per layer. A slot is one expert in all 40 layers: 408.9 MiB in cd (IQ2_XS gate/up), 380.8 MiB in
# qd. Our first build held 18 slots (decode) or 17 (long prompts) of 380.8 MiB with a dense part 3,092.5 MiB heavier
# than ISTA-DASLab's Q5_K one that cd and qd use; slots_for() spends the same VRAM budget on the build's own slots:
# cd 24 / 23 (22 with margin=1 - long prompts and the agent turns), qd 26 / 25 (24).
SLOT_MIB = {"cd": (408.9, 3092.5), "qd": (380.8, 3092.5)}


def slots_for(base, margin=0):
    """VRAM slots of this build for the budget of `base` slots of the first build, minus `margin`."""
    slot, saved = SLOT_MIB.get(config.BUILD, SLOT_MIB["cd"])
    return int((base * 380.8 + saved) // slot) - margin


COMMON = ["-ngl", "99", "--parallel", "1", "-t", "6", "--seed", "1", "--reasoning", "off",
          "--moe-stream-direct", "--moe-stream-cache", f"{slots_for(18)}s", "--host", "127.0.0.1", "--port", str(PORT)]
DECODE = [("en-code", PROMPTS["en-code"], 256), ("ru-code", PROMPTS["ru-code"], 256),
          ("en-tech", PROMPTS["en-tech"], 256), ("ru-prose", PROMPTS["ru-prose"], 256),
          ("en-code-again", PROMPTS["en-code"], 256), ("ceil-1", CEIL, 8), ("ceil-2", CEIL, 8)]

RE_CALLS = re.compile(r"moe stream: remap calls = (\d+), expert hits = (\d+), misses = (\d+) \((\d+) cold\)")
RE_STALL = re.compile(r"moe stream: load stall = ([\d.]+) ms total")
RE_L2 = re.compile(r"moe stream: L2 slab hits = (\d+), fills = (\d+), evictions = (\d+)")
RE_WAVES = re.compile(r"moe stream: waves = (\d+) \((\d+) non-empty\), preloads issued = (\d+) "
                      r"\(ready on arrival = (\d+)\), wave stall = ([\d.]+) ms")


def common():
    if MODEL is None or LLAMA is None:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE})")
    return ["-m", str(MODEL), *COMMON]


def fork_env(extra=None, exact=True):
    """The environment of the fork's binaries. Without a preset (LLAMA_MOE_STREAM_PRESET) the measurement takes the
    exact path: no dropped experts (LLAMA_MOE_STREAM_DROP_W=0) and no YoCo truncation (LLAMA_DSV4_YOCO=0), unless the
    caller set them; exact=False keeps the fork's defaults. A preset sets its own switches (explicit ones win).
    LLAMA_MOE_STREAM_DROP_W=default removes the key. The expert copies in SL_MIRRORS are passed on when present."""
    env = dict(os.environ)
    if exact and "LLAMA_MOE_STREAM_PRESET" not in env and "LLAMA_MOE_STREAM_PRESET" not in (extra or {}):
        env.setdefault("LLAMA_MOE_STREAM_DROP_W", "0")
        env.setdefault("LLAMA_DSV4_YOCO", "0")
    # a copy of the expert parts on other drives: the fork checks each part's size and 17 sampled blocks and ignores a
    # copy of another build; LLAMA_MOE_STREAM_MIRROR= (empty) turns it off
    have = [str(d).replace("\\", "/") for d in MIRRORS if d.is_dir()]
    if have:
        env.setdefault("LLAMA_MOE_STREAM_MIRROR", ";".join(have))
    env.update(extra or {})
    if env.get("LLAMA_MOE_STREAM_DROP_W") == "default":
        del env["LLAMA_MOE_STREAM_DROP_W"]
    return env


def kv_args(extra):
    """KV cache of the long-context servers: q8_0 unless the flags set their own -ctk / --cache-type-k (on a 64K
    prefill as fast as f16 with 80 MiB less VRAM; quality within error on both ds4 prompt sets)."""
    return [] if ("-ctk" in extra or "--cache-type-k" in extra) else ["-ctk", "q8_0", "-ctv", "q8_0"]


def _env_on(key, env, positive=False):
    v = (os.environ if env is None else env).get(key)
    if v is None:
        return True
    m = re.match(r"\s*[+-]?\d+", v)
    return m is not None and (int(m.group()) > 0 if positive else int(m.group()) != 0)


def p4_on(env=None):
    """Encoder-only prefill (LLAMA_DSV4_ENC_ONLY): on unless set to 0."""
    return _env_on("LLAMA_DSV4_ENC_ONLY", env)


def p11_on(env=None):
    """The expert cache yields VRAM to a long prompt, waves of all layers through a pool of 64 (LLAMA_MOE_STREAM_P11):
    on unless set to 0 or empty (the fork reads it with atoi)."""
    return _env_on("LLAMA_MOE_STREAM_P11", env, positive=True)


def d10_on(env=None):
    """Sparse attention in prefill (LLAMA_DSV4_SPARSE_PREFILL): on unless set to 0 (atoi: not a number is off)."""
    return _env_on("LLAMA_DSV4_SPARSE_PREFILL", env)


def batch_for(ub, ctx=None):
    """-b for the micro-batch ub: the smallest multiple of ub that is at least 2048 (with -b 2048 and -ub 1536 the
    server cut each chunk into 1536 + 512: -b 3072 was 18.5% faster on a 64K prefill). With encoder-only prefill and a
    known ctx, a multiple of ub of at least ctx: the decoder half runs on the last positions of each call, so the
    prompt must go in one call."""
    if ctx is not None and p4_on():
        return ub * -(-max(ctx, 2048) // ub)
    return ub * -(-2048 // ub)


def ub_for(ctx, env=None):
    """Micro-batch of a long prompt by context length: with the shared expert pool (P11) 8192 up to 32K (6144 without
    sparse prefill attention), 4608 up to 64K, 3072 beyond; without P11 1536 up to 64K, 1024 beyond."""
    if p11_on(env):
        if ctx <= 32768:
            return 8192 if d10_on(env) else 6144
        return 4608 if ctx <= 65536 else 3072
    return 1536 if ctx <= 65536 else 1024


def l2_for(ub, ctx):
    """RAM tier (GiB) for the micro-batch when --moe-stream-l2 is not given. A batch over 2048 keeps its inputs in
    pinned CUDA_Host memory (attention masks, Engram rows, embeddings - ~0.2 MiB a token at 32K), which shares the
    WDDM budget of shared memory with the tier: 44 up to 64K, 43 beyond and at a batch of 8192. Up to 2048 - 46 (the
    fork itself trims the tier to the budget minus 640 MiB)."""
    if ub <= 2048:
        return 46
    if ub >= 8192:
        return 43
    return 44 if ctx <= 65536 else 43


def require_gpu():
    """No GPU, no measurement: llama.cpp without CUDA silently computes everything on the CPU."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        sys.exit(f"no GPU: nvidia-smi does not start ({e}) - nothing measured")
    if r.returncode != 0 or "GPU 0" not in r.stdout:
        sys.exit("no GPU: " + (r.stdout + r.stderr).strip()[:300] + " - nothing measured")


def keep_awake(on):
    """Keep the system from sleeping while a measurement runs; switching on first checks the GPU (require_gpu)."""
    if on:
        require_gpu()
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def ps(command):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                          capture_output=True, text=True, timeout=60).stdout.strip()


def io_raw(pid):
    """(pages faulted in from disk - system-wide, cumulative; free MB; bytes the process has read)."""
    out = ps("$m = Get-CimInstance Win32_PerfRawData_PerfOS_Memory; "
             f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
             "@{pages=[int64]$m.PagesInputPersec; avail=[int64]$m.AvailableMBytes; "
             "read=[int64]$p.ReadTransferCount} | ConvertTo-Json -Compress")
    d = json.loads(out)
    return int(d["pages"]), int(d["avail"]), int(d["read"] or 0)


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    return used, total


def get_json(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.load(r)


def port_busy():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def wait_ready(proc, log, limit=1800):
    start = time.time()
    while time.time() - start < limit:
        if proc.poll() is not None:
            tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            raise RuntimeError("the server exited while loading:\n" + "\n".join(tail))
        try:
            if get_json("/health").get("status") == "ok":
                return time.time() - start
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("the server did not come up")


def ask(text, max_tokens, cache_prompt=True):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "seed": 1, "stream": False, "cache_prompt": cache_prompt}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        resp = json.load(r)
    wall = time.time() - t0
    t = resp.get("timings") or {}
    content = (resp["choices"][0]["message"].get("content") or "").strip()
    return {"prompt_n": t.get("prompt_n"), "pp_tps": t.get("prompt_per_second"), "n": t.get("predicted_n"),
            "tg_tps": t.get("predicted_per_second"), "wall_s": round(wall, 1), "text": content}


def stats_blocks(log):
    """All print_stats blocks of a server log: a list of dicts of cumulative counters."""
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    blocks, cur = [], None
    for line in text.splitlines():
        if m := RE_CALLS.search(line):
            cur = {"calls": int(m[1]), "hit": int(m[2]), "miss": int(m[3]), "cold": int(m[4])}
            blocks.append(cur)
        elif cur is not None and (m := RE_STALL.search(line)):
            cur["stall_ms"] = float(m[1])
        elif cur is not None and (m := RE_L2.search(line)):
            cur.update(l2_hit=int(m[1]), l2_fill=int(m[2]), l2_evict=int(m[3]))
        elif cur is not None and (m := RE_WAVES.search(line)):
            cur.update(waves=int(m[1]), waves_run=int(m[2]), wave_stall_ms=float(m[5]))
    return blocks


def wait_block(log, n_before, limit=30):
    for _ in range(limit * 4):
        blocks = stats_blocks(log)
        if len(blocks) > n_before and "stall_ms" in blocks[-1]:
            time.sleep(0.3)   # let the RAM-tier and wave lines of the same block land
            return stats_blocks(log)
        time.sleep(0.25)
    return stats_blocks(log)


def delta(cur, prev):
    return {k: round(v - prev.get(k, 0), 2) if isinstance(v, float) else v - prev.get(k, 0)
            for k, v in cur.items()}


def stop(proc):
    if proc.poll() is None:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.wait(timeout=60)


def run(name, extra, order, env_extra=None, cache_prompt=True):
    """Start a server with `extra` flags, send the requests of `order` [(label, text, max_tokens)], append a row to
    SL_RESULTS/test4.jsonl. A text of None (or a function) is a pause of max_tokens seconds between requests."""
    base = common()
    if port_busy():
        sys.exit(f"something already answers on port {PORT} - nothing measured")
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = LOGS / f"{stamp}-t4-{name}.log"
    env = fork_env(env_extra)
    cmd = [str(LLAMA), *base, *extra]
    print(f"== {name}: {' '.join(extra)}", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=str(LLAMA.parent), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    row = {"time": stamp, "name": name, "args": extra, "log": log.name, "requests": [],
           "env": {k: v for k, v in env.items() if k.startswith(("LLAMA_", "GGML_", "OMP_"))}}
    try:
        row["load_s"] = round(wait_ready(proc, log), 1)
        row["vram_mib"], total = vram_mib()
        warn = "  WARNING: less than 150 MiB of VRAM left" if total - row["vram_mib"] < 150 else ""
        print(f"   loaded in {row['load_s']} s, VRAM {row['vram_mib']} of {total} MiB{warn}", flush=True)
        prev = {}
        for label, text, max_tokens in order:
            if text is None or callable(text):   # a pause between requests: max_tokens is seconds
                print(f"   {label:18s} pause {max_tokens} s", flush=True)
                (text or time.sleep)(max_tokens)
                continue
            n_blocks = len(stats_blocks(log))
            pages0, _, read0 = io_raw(proc.pid)
            t0 = time.time()
            r = ask(text, max_tokens, cache_prompt)
            r.update(t0=round(t0, 2), t1=round(time.time(), 2))
            pages1, avail, read1 = io_raw(proc.pid)
            blocks = wait_block(log, n_blocks)
            cur = blocks[-1] if len(blocks) > n_blocks else prev
            s = delta(cur, prev)
            prev = cur
            r.update(label=label, page_mb=round((pages1 - pages0) * 4096 / 2**20),
                     read_gb=round((read1 - read0) / 1e9, 2), avail_mb=avail, stats=s)
            row["requests"].append(r)
            touched = s.get("hit", 0) + s.get("miss", 0)
            l2 = s.get("l2_hit", 0) + s.get("l2_fill", 0)
            print(f"   {label:18s} prompt {r['prompt_n']:>5} tok {r['pp_tps'] or 0:7.2f} tok/s | answer "
                  f"{r['n']:>4} tok {r['tg_tps'] or 0:5.2f} tok/s | VRAM cache {100 * s.get('hit', 0) / max(touched, 1):5.1f}%"
                  f" | L2 {100 * s.get('l2_hit', 0) / max(l2, 1):5.1f}% | wait {s.get('stall_ms', 0) / 1000:6.1f} s"
                  f" | disk {r['read_gb']:6.2f} GB, paging {r['page_mb']} MB, free {avail} MB", flush=True)
        row["vram_mib_end"], _ = vram_mib()
    except Exception as e:   # noqa: BLE001 - a failed run still gets its row
        row["error"] = str(e)
        print("   ERROR:", e, flush=True)
    finally:
        stop(proc)
    for r in row["requests"]:
        r["text"] = r["text"][:800]
    with open(RESULTS / "test4.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def long_prompts():
    if LONG_SRC is None:
        sys.exit(f"set SL_LLAMA_SRC (environment or {config.FILE})")
    src = LONG_SRC.read_text(encoding="utf-8", errors="replace")
    # kept as measured: "here is a fragment of C source ... in 3-4 sentences say what this file does and its main parts"
    ask_ = "\n```\nКратко (3–4 фразы) скажи, что делает этот файл и какие в нём главные подсистемы."
    # different parts of the file: no common prefix, so the prompt cache cannot shorten the measurement
    return [("prefill-2k", "Вот фрагмент исходника на C:\n```c\n" + src[:7000] + ask_, 32),
            ("prefill-8k", "Вот фрагмент исходника на C:\n```c\n" + src[40000:68000] + ask_, 32)]


def report():
    rows = [json.loads(line) for line in open(RESULTS / "test4.jsonl", encoding="utf-8")]
    for row in rows:
        print(f"{row['name']:14s} {' '.join(row['args']):45s} VRAM {row.get('vram_mib', 0):6d}"
              + (f"  ERROR: {row['error'][:100]}" if row.get("error") else ""))
        for r in row["requests"]:
            s = r.get("stats", {})
            touched = s.get("hit", 0) + s.get("miss", 0)
            l2 = s.get("l2_hit", 0) + s.get("l2_fill", 0)
            print(f"    {r['label']:18s} pp {r['prompt_n']:>5} @ {r['pp_tps'] or 0:7.2f} | tg {r['n']:>4} @ "
                  f"{r['tg_tps'] or 0:5.2f} | VRAM {100 * s.get('hit', 0) / max(touched, 1):5.1f}% | "
                  f"L2 {100 * s.get('l2_hit', 0) / max(l2, 1):5.1f}% | stall {s.get('stall_ms', 0) / 1000:6.1f} s | "
                  f"disk {r.get('read_gb', 0):6.2f} GB | pages {r.get('page_mb', 0)} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1:2]
    if cmd == ["report"]:
        report()
        sys.exit()
    if cmd not in (["ladder"], ["run"], ["prefill"]):
        sys.exit(__doc__)
    keep_awake(True)
    try:
        if cmd == ["ladder"]:
            tag = sys.argv[2] + "-" if len(sys.argv) > 2 else ""
            for gib in (24, 0, 16, 32, 40):
                run(f"{tag}l2-{gib}", ["-c", "4096", "--moe-stream-l2", str(gib)], DECODE)
        elif cmd == ["run"]:
            a = sys.argv[3:]
            run(sys.argv[2], a[1:] if a[:1] == ["--"] else a, DECODE)
        else:
            tag = sys.argv[2] + "-" if len(sys.argv) > 2 else ""
            extra = sys.argv[4:]   # more server flags, e.g. --moe-stream-io-threads 17
            # --head-gpu (a flag of this script): the output head stays on the GPU (it yields VRAM to the prompt);
            # otherwise it goes to the CPU
            head_gpu = "--head-gpu" in extra
            head = [] if head_gpu else ["-ot", r"output\.weight=CPU"]
            extra = [a for a in extra if a != "--head-gpu"]
            # micro-batches: a comma-separated list; "auto" - by context length with the head on the GPU, the ladder
            # 512 / 1024 / 2048 with the head on the CPU
            ub_arg = sys.argv[3] if len(sys.argv) > 3 else "auto"
            if ub_arg == "auto":
                ubs = (ub_for(12288),) if head_gpu else (512, 1024, 2048)
            else:
                ubs = [int(u) for u in ub_arg.split(",")]
            for ub in ubs:
                # RAM tier: 24 GiB with the head on the CPU; by batch with the head on the GPU (l2_for). Expert slots
                # with the head on the GPU: the long-prompt budget (slots_for(17, margin=1)). Flags given win
                tier = [] if "--moe-stream-l2" in extra else ["--moe-stream-l2", str(l2_for(ub, 12288) if head_gpu else 24)]
                cache = (["--moe-stream-cache", f"{slots_for(17, margin=1)}s"]
                         if head_gpu and "--moe-stream-cache" not in extra else [])
                run(f"{tag}pp-ub{ub}", ["-c", "12288", *tier, *cache, "-b", str(batch_for(ub, 12288)), "-ub", str(ub),
                                   *head, *kv_args(extra), "-lv", "4", *extra], long_prompts(), cache_prompt=False)
    finally:
        keep_awake(False)
