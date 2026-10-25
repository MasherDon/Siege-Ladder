"""Quality anchors: fresh continuations of the official DeepSeek API on the prompts of ds4's quality sets.

ds4's sets (antirez/ds4, gguf-tools/quality-testing: general-100, router-112) were collected from the API before the
1017 checkpoint; `deepseek-flash` now serves DeepSeek-V4.1-Flash-1017, so a build of 1017 cannot be compared with
them. This collects the same prompts byte for byte with the source's own parameters (its collection.json): no system
prompt, thinking off, temperature 1, the source's max_tokens, top-20 logprobs - the request of ds4's
collect_official.py. The output is a folder in ds4's layout (manifest.tsv, prompts/, continuations/, responses/,
collection.json) that ds4_score.py reads as FIXTURE.

The key: the DEEPSEEK_API_KEY environment variable, or a line DEEPSEEK_API_KEY=... in bench/.env next to this script
(keep that file out of git). The key is never printed or written to the output.

    python bench/api_collect.py estimate SRC     # tokens and price by the source's collection.json, offline
    python bench/api_collect.py run SRC OUT [--limit N] [--budget 0.5] [--resume]

SRC - a ds4 set (e.g. gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-general in a clone of antirez/ds4; the
sets are not included here). OUT - a new folder; a relative OUT that does not exist from the current folder goes under
SL_REF, the root of the anchors, when that is set. Progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
# what deepseek-flash serves by DeepSeek's pricing page; written into collection.json - change it when the API moves on
CHECKPOINT = "DeepSeek-V4.1-Flash-1017 (DeepSeek's pricing page)"
ENV_FILE = HERE / ".env"
REF = config.path("SL_REF")
# DeepSeek's pricing page for deepseek-flash, $ per 1M tokens - off-peak / peak (peak 01-04 and 06-10 UTC, Mon-Fri)
PRICE_IN_MISS = (0.15, 0.30)
PRICE_IN_HIT = (0.003, 0.006)
PRICE_OUT = (0.60, 1.20)


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        sys.exit(f"no key: neither DEEPSEEK_API_KEY in the environment nor a line of it in {ENV_FILE}")
    return key


def peak_now() -> bool:
    t = time.gmtime()
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


def cost(usage: dict, peak: bool) -> float:
    i = 1 if peak else 0
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = max(0, (usage.get("prompt_tokens", 0) or 0) - hit)
    out = usage.get("completion_tokens", 0) or 0
    return (miss * PRICE_IN_MISS[i] + hit * PRICE_IN_HIT[i] + out * PRICE_OUT[i]) / 1e6


def anchor_dir(arg) -> Path:
    """OUT as given when it is absolute or exists; otherwise under SL_REF when that is set."""
    p = Path(arg)
    if p.is_absolute() or p.exists() or REF is None:
        return p
    return REF / p


def load_source(src: Path):
    """The source's cases in the order of its manifest.tsv: (id, the prompt's text byte for byte)."""
    cases = []
    for line in (src / "manifest.tsv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = [c for c in line.split("\t") if c]
        prompt = src.joinpath(*cols[1].replace("\\", "/").split("/")[-2:])
        cases.append((cols[0], prompt.read_bytes()))
    coll = json.loads((src / "collection.json").read_text(encoding="utf-8"))
    return cases, coll


def request(key: str, prompt: str, coll: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": coll.get("temperature", 1),
        "stream": False,
        "logprobs": True,
        "top_logprobs": coll.get("top_logprobs", 20),
        "max_tokens": coll["max_tokens"],
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    delay = 1.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as fp:
                return json.loads(fp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code < 500 and e.code != 429:
                sys.exit(f"HTTP {e.code}: {body}")
            last = f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001 - retry on network failures
            last = repr(e)
        print(f"  retry {attempt + 1}: {last}", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay *= 1.7
    sys.exit(f"failed after 6 attempts: {last}")


def estimate(args):
    cases, coll = load_source(Path(args.src))
    pin = sum(c.get("prompt_tokens", 0) for c in coll.get("cases", [])) or None
    pout = sum(c.get("completion_tokens", 0) for c in coll.get("cases", [])) or len(cases) * coll["max_tokens"]
    if pin is None:
        pin = sum(len(p) for _, p in cases) // 3   # ~3 bytes a token - rough, for a source that did not record tokens
    for peak in (False, True):
        u = {"prompt_tokens": pin, "completion_tokens": pout}
        print(f"{'peak' if peak else 'off-peak'}: {len(cases)} cases, ~{pin} prompt tokens, ~{pout} answer tokens - "
              f"~${cost(u, peak):.3f}")


def run(args):
    src, out = Path(args.src), anchor_dir(args.out)
    cases, coll = load_source(src)
    if args.limit:
        cases = cases[: args.limit]
    key = api_key()
    print(f"writing to {out}", file=sys.stderr, flush=True)
    for d in ("prompts", "continuations", "responses"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, fps, models, spent, usage_sum = [], set(), set(), 0.0, {}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for n, (cid, prompt) in enumerate(cases, 1):
        pp, cp, rp = out / "prompts" / f"{cid}.txt", out / "continuations" / f"{cid}.txt", out / "responses" / f"{cid}.json"
        if args.resume and pp.exists() and cp.exists() and rp.exists() and pp.read_bytes() == prompt:
            resp = json.loads(rp.read_text(encoding="utf-8"))
        else:
            for empty in range(3):
                resp = request(key, prompt.decode("utf-8"), coll)
                content = resp["choices"][0].get("message", {}).get("content")
                if isinstance(content, str) and content:
                    break
                print(f"  {cid}: empty continuation, retrying", file=sys.stderr, flush=True)
                time.sleep(1.0 + empty)
            else:
                sys.exit(f"{cid}: three empty continuations in a row")
            if not (resp["choices"][0].get("logprobs") or {}).get("content"):
                sys.exit(f"{cid}: no logprobs in the response")
            pp.write_bytes(prompt)
            cp.write_bytes(resp["choices"][0]["message"]["content"].encode("utf-8"))
            rp.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            u = resp.get("usage", {})
            spent += cost(u, peak_now())
            for k, v in u.items():
                if isinstance(v, int):
                    usage_sum[k] = usage_sum.get(k, 0) + v
        fps.add(resp.get("system_fingerprint"))
        models.add(resp.get("model"))
        rows.append(cid)
        print(f"{n}/{len(cases)} {cid}: tokens {resp.get('usage', {}).get('prompt_tokens')} + "
              f"{resp.get('usage', {}).get('completion_tokens')}, fp {resp.get('system_fingerprint')}, "
              f"spent ~${spent:.4f}", file=sys.stderr, flush=True)
        if args.budget and spent > args.budget:
            print(f"the budget of ${args.budget} is spent - stopping after {cid}", file=sys.stderr)
            break
        time.sleep(0.05)
    with (out / "manifest.tsv").open("w", encoding="utf-8", newline="\n") as fp:
        fp.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n")
        for cid in rows:
            fp.write(f"{cid}\tprompts/{cid}.txt\tcontinuations/{cid}.txt\tresponses/{cid}.json\n")
    meta = {
        "schema": "ds4-official-continuations-v1",
        "created_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "checkpoint": CHECKPOINT,
        "endpoint": ENDPOINT,
        "source_fixture": str(src),
        "source_created_at": coll.get("created_at"),
        "observed_models": sorted(m for m in models if m),
        "fingerprints": sorted(f for f in fps if f),
        "temperature": coll.get("temperature", 1),
        "top_p": "provider default",
        "max_tokens": coll["max_tokens"],
        "top_logprobs": coll.get("top_logprobs", 20),
        "thinking": "disabled",
        "system": None,
        "case_count": len(rows),
        "usage": usage_sum,
        "cost_usd_estimate": round(spent, 4),
        "tool": "Siege-Ladder bench/api_collect.py",
    }
    (out / "collection.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: {len(rows)} cases, fingerprints {meta['fingerprints']}, ~${spent:.4f}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("src")
    r = sub.add_parser("run")
    r.add_argument("src")
    r.add_argument("out")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--budget", type=float, default=0.5)
    r.add_argument("--resume", action="store_true")
    args = p.parse_args()
    {"estimate": estimate, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
