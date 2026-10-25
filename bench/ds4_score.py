"""Quality anchor: the NLL of the DeepSeek API's own continuations under our build (teacher forcing).

An anchor is a folder in the layout of antirez/ds4's quality tests: short prompts and the official API's continuations
with top-20 logprobs. ds4's general-100 (gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-general: 100 prompts,
no thinking, temperature 1, up to 32 tokens, 2994 answer tokens) was collected before the 1017 checkpoint;
api_collect.py collects the same prompts again from the current API into SL_REF. ds4's references for general-100
(QA_BEFORE_RELEASES.md, section 17): Q2 (IQ2_XXS gate/up + Q2_K down - the class of our builds) mean NLL 0.3615-0.3657
and 2696-2705 of 2994 matches with the API's top-1; Q4 0.2465-0.2524 and 2896. The full model itself (the API's
logprobs of its own tokens) is at 0.2282. We report quality as the distance to the full model: smaller is better.

Everything as in ds4's score_official.c: the prompt is BOS <｜User｜> text <｜Assistant｜></think>, with no system
prompt and no reasoning-effort line; the continuation is tokenized separately and fed one token at a time (teacher
forcing); NLL is the mean over all continuation tokens of all cases; a match is the model's argmax against the API's
most likely alternative, counted only in cases whose API token bytes line up with our token boundaries. Rendering, the
byte alignment and all metrics are done here; llama-logits with our patch (patches/0023-tools-llama-logits-...) only
tokenizes (the vocabulary, no weights) and runs the passes - one model load for all cases. The TSV is written in
score_official's format (26 columns), so ds4's validate_scores.py and compare_scores.py read it.

    python bench/ds4_score.py check    FIXTURE               # the set only, no binaries
    python bench/ds4_score.py run      FIXTURE OUT [options] # tokenize + score + report
    python bench/ds4_score.py report   FIXTURE OUT           # the report again from OUT's tok_out / positions.tsv
    python bench/ds4_score.py selftest FIXTURE OUT           # the metrics on "model = API", no binaries

    # our anchors collected into SL_REF/general and SL_REF/router, run on the cd build (--slots: the VRAM expert
    # slots that fit beside each context):
    python bench/ds4_score.py run general OUT/g --slots 23 --env LLAMA_MOE_STREAM_PRESET=exact
    python bench/ds4_score.py run router OUT/r --ctx 32768 --ub 6144 --slots 22 --l2 44 --extra=--swa-window

FIXTURE - the set's folder (manifest.tsv, prompts/, continuations/, responses/); a relative path that does not exist
from the current folder is taken under SL_REF. OUT - the run's folder: the job files, tok_out.tsv, positions.tsv,
scores.tsv (ds4's format), summary.txt, report.json, meta.json and the binaries' logs.
Options of run: --model GGUF (default SL_MODEL), --bin DIR (where llama-logits.exe is; default SL_LLAMA_BIN), --ctx
4096, --ub 1, --batch (prompt and answer in one call), --slots 23 (cd) / 25 (qd), --l2 46, --io N, --threads 6, --top 64,
--max-cases N (a probe; the result is not comparable), --env NAME=VALUE (repeatable; no other LLAMA_* or GGML_* of the
environment is passed on), --extra "FLAGS" (more llama-logits flags), --no-repeat (no repeat of the first case at the
end), --dry-run (only the job files), --force (do not check that the GPU is there and no other llama-* runs).

Windows only as written (llama-logits.exe, tasklist, SetThreadExecutionState).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from harness import slots_for  # noqa: E402

REPO = config.path("SL_LLAMA_SRC")        # the patched llama.cpp tree: its commit goes into meta.json
LLAMA_BIN = config.path("SL_LLAMA_BIN")   # where llama-logits.exe is
MODEL = config.MODEL                      # the first GGUF part of the build
REF = config.path("SL_REF")               # root of the quality anchors

# ds4.c, encode_chat_prompt() for DeepSeek V4.1, DS4_THINK_NONE, system = NULL: BOS, <｜User｜>, the text,
# <｜Assistant｜>, </think> (the same strings as render_deepseek_ds4_prompt() in score_llama.cpp)
BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
USER = "<\uff5cUser\uff5c>"
ASSISTANT = "<\uff5cAssistant\uff5c>"
THINK_END = "</think>"
SPECIALS = (BOS, USER, ASSISTANT, THINK_END)
REPLACEMENT = b"\xef\xbf\xbd"
TOP_CAP = 64            # ds4: local_top_ids() and the list of pairs hold at most 64
SEC_PER_TOKEN = 0.088   # the main mode (46 GiB RAM tier): 88 ms per generated token - for the time estimate only
REPEAT = "@again"       # the first case again at the end of a run: the same arithmetic in another expert-cache state

COLUMNS = (
    "id prompt_tokens target_tokens nll avg_nll first_match greedy_lcp "
    "api_ref_tokens api_target_tokens api_target_mae api_target_mean_delta "
    "api_top_items api_top_mapped api_top_coverage api_top1_count api_top1_match "
    "api_top1_rate api_topn_ref api_topn_hit api_topn_recall api_top_logprob_count "
    "api_top_mae api_top_mean_delta api_pair_total api_pair_agree api_pair_rate"
).split()

# ds4's QA_BEFORE_RELEASES.md, section 17 (General 100): label, mean NLL, top-1 matches, out of
REFERENCE = (
    ("Q2, M5 SSD", 0.364576009, 2697, 2994),
    ("Q2, M3 Ultra, batched prefill", 0.365680596, 2705, 2994),
    ("Q2, Metal TP", 0.361471738, 2699, 2994),
    ("Q2, CUDA SSD (DGX Spark)", 0.363135483, 2696, 2994),
    ("Q4, M5 SSD", 0.246523256, 2896, 2994),
    ("Q4, M3 Ultra", 0.252409161, 2896, 2994),
)
# Proposed thresholds: the share of the Q2 -> Q4 gap (by NLL and by top-1) on the worse side of Q2. Thresholds are
# fixed before a run: change them here, before measuring, and record the change with the results.
WARN_GAP, FAIL_GAP = 0.25, 0.5


# ---------------------------------------------------------------- the set

def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _bytes(d, key):
    """The token's bytes; None - null or no field (ds4: bytes = NULL); False - ds4 would not parse the response."""
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 255
                                          for x in v):
        return False
    return bytes(v)


def parse_api(raw: bytes):
    """ds4's api_ref_parse(): [(logprob, bytes, [(bytes, logprob)])] or None (ds4: "no API logprobs parsed")."""
    try:
        content = json.loads(raw.decode("utf-8"))["choices"][0]["logprobs"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, list) or not content:
        return None
    out = []
    for item in content:
        if not isinstance(item, dict):
            return None
        lp, b = _num(item.get("logprob")), _bytes(item, "bytes")
        if lp is None or b is False:
            return None
        alts = []
        if "top_logprobs" in item:
            if not isinstance(item["top_logprobs"], list):
                return None
            for a in item["top_logprobs"]:
                if not isinstance(a, dict):
                    return None
                alp, ab = _num(a.get("logprob")), _bytes(a, "bytes")
                if alp is None or ab is False:
                    return None
                alts.append((ab, alp))
        out.append((lp, b, alts))
    return out


class Case:
    def __init__(self, cid, files, prompt, cont, api, resp):
        self.id, self.files, self.prompt, self.cont, self.api = cid, files, prompt, cont, api
        usage = resp.get("usage") if isinstance(resp, dict) else None
        self.usage_prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        self.model = resp.get("model") if isinstance(resp, dict) else None
        self.fingerprint = resp.get("system_fingerprint") if isinstance(resp, dict) else None


def fixture_dir(arg) -> Path:
    """FIXTURE as given when that path exists; otherwise the folder of that name under SL_REF (when it is there)."""
    p = Path(arg)
    if not p.exists() and not p.is_absolute() and REF is not None and (REF / p).exists():
        p = REF / p
    return p.resolve()


def load_fixture(fixture: Path, max_cases=0):
    """manifest.tsv read the way ds4 reads it (fgets + strtok: empty fields vanish); its paths are relative to the
    root of the ds4 repository, so only their last two parts (prompts/case_000.txt) are taken, relative to FIXTURE."""
    cases, seen = [], set()
    for line in (fixture / "manifest.tsv").read_bytes().decode("utf-8").split("\n"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        if max_cases and len(cases) >= max_cases:
            break
        cols = [c for c in line.split("\t") if c]
        if len(cols) < 3:
            sys.exit(f"manifest: bad line {line[:80]!r}")
        if cols[0] in seen:
            sys.exit(f"manifest: case {cols[0]} twice")
        seen.add(cols[0])
        files = [fixture.joinpath(*p.replace("\\", "/").split("/")[-2:]) for p in cols[1:4]]
        api, resp = None, None
        if len(files) > 2:
            raw = files[2].read_bytes()
            api = parse_api(raw)
            try:
                resp = json.loads(raw.decode("utf-8"))
            except ValueError:
                resp = None
        cases.append(Case(cols[0], files, files[0].read_bytes(), files[1].read_bytes(), api, resp))
    if not cases:
        sys.exit("no cases in manifest")
    return cases


def is_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def alt_candidate(b) -> bool:
    """ds4's api_alt_token_id() before tokenizing: empty pieces and pieces with NUL or U+FFFD (the provider lost bytes)
    are not mapped."""
    return bool(b) and b"\x00" not in b and REPLACEMENT not in b


def fixture_digest(cases, fixture: Path) -> str:
    h = hashlib.sha256((fixture / "manifest.tsv").read_bytes())
    for c in cases:
        for f in c.files:
            h.update(f.read_bytes())
    return h.hexdigest()


def check(cases, fixture: Path, quiet=False) -> dict:
    """The set's integrity, without the model. Returns a summary; problems go into its "problems" field."""
    problems = []
    api_tokens = self_nll = sampled_top = 0
    alts, bad_utf8, fffd = set(), set(), 0
    for c in cases:
        for name, data in (("prompt", c.prompt), ("continuation", c.cont)):
            if not is_utf8(data):
                problems.append(f"{c.id}: {name} is not UTF-8")
            if b"\r" in data:
                problems.append(f"{c.id}: {name} contains CR - Windows line endings (git autocrlf?); the API bytes "
                                f"will not match")
            if b"\x00" in data:
                problems.append(f"{c.id}: {name} contains NUL - ds4 would cut the string there")
            for s in SPECIALS:
                if s.encode() in data:
                    problems.append(f"{c.id}: {name} contains {s!r} - ds4 tokenizes it as text")
        if c.api is None:
            problems.append(f"{c.id}: API logprobs not parsed - ds4 would skip the comparison with the API")
            continue
        api_tokens += len(c.api)
        if all(b for _, b, _ in c.api) and b"".join(b for _, b, _ in c.api) != c.cont:
            problems.append(f"{c.id}: the bytes of the API tokens do not add up to the continuation file")
        for lp, b, al in c.api:
            self_nll -= lp
            if al and b == max(al, key=lambda a: a[1])[0]:
                sampled_top += 1
            for ab, _ in al:
                if ab is not None:
                    alts.add(ab)
                    if not is_utf8(ab):
                        bad_utf8.add(ab)
                    if REPLACEMENT in ab:
                        fffd += 1
    usage = [c.usage_prompt for c in cases if isinstance(c.usage_prompt, int)]
    out = {
        "cases": len(cases), "api_tokens": api_tokens,
        "api_prompt_tokens": sum(usage) if len(usage) == len(cases) else None,
        "api_self_nll": self_nll / api_tokens if api_tokens else None,
        "sampled_is_api_top1": sampled_top,
        "distinct_alternatives": len(alts), "alternatives_not_utf8": len(bad_utf8), "alternatives_with_fffd": fffd,
        "models": sorted({str(c.model) for c in cases}), "fingerprints": sorted({str(c.fingerprint) for c in cases}),
        "fixture_sha256": fixture_digest(cases, fixture), "problems": problems,
    }
    note = None
    prompts_jsonl = fixture.parent / "prompts.jsonl"
    if prompts_jsonl.exists():
        rows = [json.loads(x)["prompt"] for x in prompts_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
        # in a clone of ds4 every set sits next to prompts.jsonl, the prompt list of general-100: it is compared only
        # with a set that has as many cases
        n_manifest = sum(1 for x in (fixture / "manifest.tsv").read_bytes().decode("utf-8").split("\n")
                         if x.rstrip("\r\n") and not x.startswith("#"))
        if len(rows) == n_manifest:
            diff = [c.id for i, c in enumerate(cases)
                    if i >= len(rows) or c.prompt.decode("utf-8", "replace") != rows[i]]
            out["prompts_jsonl_mismatch"] = diff
            if diff:
                problems.append(f"prompts differ from prompts.jsonl: {', '.join(diff[:5])}")
        else:
            note = (f"  prompts.jsonl next to the set lists {len(rows)} prompts, the set has {n_manifest} cases: "
                    f"not compared")
    if not quiet:
        print(f"Set: {out['cases']} cases, {api_tokens} API answer tokens, {out['api_prompt_tokens']} prompt tokens "
              f"by the API's usage; models {out['models']}, fingerprints {out['fingerprints']}")
        if out["api_self_nll"] is not None:
            print(f"  NLL of the full model itself (the API's logprobs of its own tokens): {out['api_self_nll']:.6f}; "
                  f"the sampled token is the API's top-1 in {sampled_top} of {api_tokens}")
        print(f"  alternatives: {len(alts)} distinct, {len(bad_utf8)} not UTF-8 (mapped through the vocabulary), "
              f"{fffd} with U+FFFD (ds4 excludes them)")
        print(f"  sha256 of the set: {out['fixture_sha256']}")
        if note:
            print(note)
        print("  no problems" if not problems else "  PROBLEMS:\n    " + "\n    ".join(problems))
    return out


# ---------------------------------------------------------------- tokenizing (llama-logits --tokenize-file)

def render(prompt: bytes) -> bytes:
    return (BOS + USER).encode() + prompt + (ASSISTANT + THINK_END).encode()


def tokenize_jobs(cases):
    """Lines <tag>\\t<t|p|v>\\t<hex>. The whole prompt with special tokens (R:) only cross-checks the piecewise
    rendering (P:), which repeats ds4: special tokens as ids, the text without parsing special tokens."""
    jobs = [("SPEC", "p", "".join(SPECIALS).encode()), ("VOCAB", "v", b"")]
    alt_tags = {}
    for c in cases:
        jobs += [(f"P:{c.id}", "t", c.prompt), (f"R:{c.id}", "p", render(c.prompt)), (f"C:{c.id}", "t", c.cont)]
        for _, _, al in c.api or ():
            for ab, _ in al:
                # llama.cpp's tokenizer would turn non-UTF-8 bytes into U+FFFD: such pieces are looked up in the
                # vocabulary
                if alt_candidate(ab) and is_utf8(ab) and ab not in alt_tags:
                    alt_tags[ab] = f"A:{len(alt_tags)}"
    jobs += [(tag, "t", ab) for ab, tag in alt_tags.items()]
    return jobs, alt_tags


def write_tokenize_jobs(path: Path, jobs):
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# tag\tmode\thex\n")
        for tag, mode, data in jobs:
            f.write(f"{tag}\t{mode}\t{data.hex()}\n")


def read_tokenize_out(path: Path):
    toks, vocab = {}, {}
    for line in path.read_bytes().decode("ascii").split("\n"):
        if not line or line.startswith("#"):
            continue
        tag, n, ids, pieces = line.split("\t")
        n = int(n)
        if tag == "VOCAB":
            vocab[int(ids)] = bytes.fromhex(pieces)
            continue
        idl = [int(x) for x in ids.split(",")] if n else []
        pcs = [bytes.fromhex(x) for x in pieces.split(",")] if n else []
        if len(idl) != n or len(pcs) != n:
            sys.exit(f"tok_out: line {tag} parsed wrongly")
        toks[tag] = (idl, pcs)
    return toks, vocab


class Prepared:
    """A case after tokenizing: the prompt and answer ids, the alignment with the API, the API's alternatives as our
    ids."""

    def __init__(self, case, prompt_ids, target_ids, target_pieces, render_same):
        self.case, self.id = case, case.id
        self.prompt_ids, self.target_ids, self.target_pieces = prompt_ids, target_ids, target_pieces
        self.render_same = render_same
        self.aligned, self.why, self.alt_ids = False, "", None


def prepare(cases, toks, vocab):
    by_piece = {}
    for i, b in vocab.items():
        by_piece.setdefault(b, []).append(i)
    # ds4 takes the special tokens' ids by looking their text up in the vocabulary (vocab_lookup), and so does this;
    # llama.cpp's parse of the special-token string is only a cross-check
    spec_ids = []
    for s in SPECIALS:
        cand = by_piece.get(s.encode(), [])
        if len(cand) != 1:
            sys.exit(f"special token {s!r} found {len(cand)} times in the vocabulary")
        spec_ids.append(cand[0])
    if toks["SPEC"][0] != spec_ids:
        print(f"   WARNING: llama.cpp parses the special-token string as {toks['SPEC'][0]}, "
              f"the vocabulary gives {spec_ids}")
    bos, user, assistant, think_end = spec_ids
    alt_tags = {}
    for tag, mode, data in tokenize_jobs(cases)[0]:
        if tag.startswith("A:"):
            alt_tags[data] = tag
    stats = {"tok": 0, "vocab": 0, "unmapped": 0}
    cache = {}

    def map_alt(ab):
        """ds4's api_alt_token_id(): exactly one token, and its bytes equal the alternative's bytes."""
        if ab in cache:
            return cache[ab]
        tok, how = -1, "unmapped"
        if alt_candidate(ab):
            if is_utf8(ab):
                ids, pcs = toks[alt_tags[ab]]
                if len(ids) == 1 and pcs[0] == ab:
                    tok, how = ids[0], "tok"
            else:
                cand = by_piece.get(ab, [])
                if len(cand) == 1:
                    tok, how = cand[0], "vocab"
        cache[ab] = (tok, how)
        return cache[ab]

    out = []
    for c in cases:
        p_ids = toks[f"P:{c.id}"][0]
        prompt_ids = [bos, user, *p_ids, assistant, think_end]
        t_ids, t_pcs = toks[f"C:{c.id}"]
        pc = Prepared(c, prompt_ids, t_ids, t_pcs, toks[f"R:{c.id}"][0] == prompt_ids)
        if not t_ids:
            sys.exit(f"{c.id}: empty continuation")
        if c.api is None:
            pc.why = "no API logprobs"
        elif len(c.api) != len(t_ids):
            pc.why = f"API tokens {len(c.api)}, ours {len(t_ids)}"
        elif any(not b or b != p for (_, b, _), p in zip(c.api, t_pcs)):
            pc.why = "the API tokens' bytes do not match our token boundaries"
        else:
            pc.aligned = True
            pc.alt_ids = []
            for _, _, al in c.api:
                row = []
                for ab, _ in al:
                    tok, how = map_alt(ab)
                    stats[how] += 1
                    row.append(tok)
                pc.alt_ids.append(row)
        out.append(pc)
    return out, stats


# ---------------------------------------------------------------- scoring (llama-logits --score-file)

def write_score_jobs(path: Path, prep, repeat=True):
    def line(pc, cid):
        s = f"{cid}\t{','.join(map(str, pc.prompt_ids))}\t{','.join(map(str, pc.target_ids))}"
        if pc.aligned:
            s += "\t" + ";".join(",".join(map(str, sorted({t for t in ids if t >= 0}))) for ids in pc.alt_ids)
        return s + "\n"

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# case\tprompt ids\ttarget ids\tids per target position\n")
        for pc in prep:
            f.write(line(pc, pc.id))
        if repeat:
            f.write(line(prep[0], prep[0].id + REPEAT))


def _float(s):
    try:
        return float(s)
    except ValueError:      # MSVC prints NaN as "-nan(ind)"
        return math.nan


def read_positions(path: Path):
    rows = {}
    for line in path.read_bytes().decode("ascii").split("\n"):
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 9:
            sys.exit(f"positions: {len(f)} fields instead of 9: {line[:80]}")
        cid, i, target, tlp, am, amlp, lse, top, extra = f
        ex = {}
        for e in filter(None, extra.split(",")):
            k, v = e.split(":")
            ex[int(k)] = _float(v)
        rows.setdefault(cid, []).append((int(i), (int(target), _float(tlp), int(am), _float(amlp), _float(lse),
                                                  [int(x) for x in top.split(",") if x], ex)))
    out = {}
    for cid, v in rows.items():
        v.sort(key=lambda r: r[0])
        if [i for i, _ in v] != list(range(len(v))):
            sys.exit(f"positions: the positions of {cid} are not consecutive")
        out[cid] = [r for _, r in v]
    return out


# ---------------------------------------------------------------- metrics (ds4's score_official.c, lines 990-1135)

def score_case(pc, rows):
    m = dict.fromkeys(("target_count", "top_items", "top_mapped", "top_logprob_count", "top1_count", "top1_match",
                       "topn_ref", "topn_hit", "pair_total", "pair_agree"), 0)
    m.update(target_abs=0.0, target_signed=0.0, top_abs=0.0, top_signed=0.0)
    if len(rows) != len(pc.target_ids):
        sys.exit(f"{pc.id}: {len(rows)} positions, {len(pc.target_ids)} answer tokens")
    nll, lcp, still, first = 0.0, 0, True, False
    for i, t in enumerate(pc.target_ids):
        target, tlp, greedy, _, _, top, extra = rows[i]
        if target != t or not math.isfinite(tlp):
            sys.exit(f"{pc.id}: position {i}: token {target} against {t}, logprob {tlp}")
        if i == 0:
            first = greedy == t
        if still and greedy == t:
            lcp += 1
        else:
            still = False
        nll += -tlp
        if not pc.aligned:
            continue
        api_lp, _, alts = pc.case.api[i]
        ids = pc.alt_ids[i]
        # api_top1_update(): exact maxima only; among equal ones, the one that matches greedy
        best = max((alp for _, alp in alts), default=-math.inf)
        ref = -1
        for (_, alp), tok in zip(alts, ids):
            if alp == best and tok >= 0 and (ref < 0 or tok == greedy):
                ref = tok
        if ref >= 0:
            m["top1_count"] += 1
            m["top1_match"] += ref == greedy
        d = tlp - api_lp
        m["target_count"] += 1
        m["target_abs"] += abs(d)
        m["target_signed"] += d
        if not alts:
            continue
        k = min(len(alts), TOP_CAP)
        if len(top) < k:
            sys.exit(f"{pc.id}: positions has top-{len(top)}, {k} needed (--top)")
        top_k = set(top[:k])
        m["top_items"] += len(alts)
        mapped = []
        for (_, alp), tok in zip(alts, ids):
            if tok < 0:
                continue
            lp = extra.get(tok)
            if lp is None:
                sys.exit(f"{pc.id}: position {i}: no logprob for id {tok}")
            if not math.isfinite(lp):
                continue
            d = lp - alp
            m["top_mapped"] += 1
            m["top_logprob_count"] += 1
            m["top_abs"] += abs(d)
            m["top_signed"] += d
            m["topn_ref"] += 1
            m["topn_hit"] += tok in top_k
            if len(mapped) < TOP_CAP:
                mapped.append((tok, alp, lp))
        for a in range(len(mapped)):
            for b in range(a + 1, len(mapped)):
                if mapped[a][0] == mapped[b][0]:
                    continue
                ad, ld = mapped[a][1] - mapped[b][1], mapped[a][2] - mapped[b][2]
                if abs(ad) < 1e-12 or abs(ld) < 1e-12:
                    continue
                m["pair_total"] += 1
                m["pair_agree"] += (ad > 0) == (ld > 0)
    return nll, first, lcp, m


def _avg(s, n):
    return s / n if n else 0.0


def tsv_row(pc, nll, first, lcp, m):
    n = len(pc.target_ids)
    v = [pc.id, len(pc.prompt_ids), n, f"{nll:.9f}", f"{_avg(nll, n):.9f}", int(first), lcp,
         len(pc.case.api) if pc.case.api is not None else 0, m["target_count"],
         f"{_avg(m['target_abs'], m['target_count']):.9f}", f"{_avg(m['target_signed'], m['target_count']):.9f}",
         m["top_items"], m["top_mapped"], f"{_avg(m['top_mapped'], m['top_items']):.9f}",
         m["top1_count"], m["top1_match"], f"{_avg(m['top1_match'], m['top1_count']):.9f}",
         m["topn_ref"], m["topn_hit"], f"{_avg(m['topn_hit'], m['topn_ref']):.9f}",
         m["top_logprob_count"], f"{_avg(m['top_abs'], m['top_logprob_count']):.9f}",
         f"{_avg(m['top_signed'], m['top_logprob_count']):.9f}",
         m["pair_total"], m["pair_agree"], f"{_avg(m['pair_agree'], m['pair_total']):.9f}"]
    return "\t".join(map(str, v))


def bootstrap(per_case, n=2000, seed=1):
    """The 95% interval of the mean NLL over prompts (resampling the cases; not a paired comparison with ds4)."""
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [per_case[rnd.randrange(len(per_case))] for _ in per_case]
        vals.append(sum(a for a, _ in s) / sum(b for _, b in s))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def report(fixture: Path, out: Path, prep, stats, positions, meta=None):
    tot = dict.fromkeys(("target_count", "top_items", "top_mapped", "top_logprob_count", "top1_count", "top1_match",
                         "topn_ref", "topn_hit", "pair_total", "pair_agree"), 0)
    tot.update(target_abs=0.0, target_signed=0.0, top_abs=0.0, top_signed=0.0)
    total_nll, total_tokens, total_lcp, firsts, ref_tokens = 0.0, 0, 0, 0, 0
    lines, per_case, table = ["\t".join(COLUMNS)], [], []
    for pc in prep:
        if pc.id not in positions:
            sys.exit(f"{pc.id} is not in positions - did the run break off?")
        nll, first, lcp, m = score_case(pc, positions[pc.id])
        lines.append(tsv_row(pc, nll, first, lcp, m))
        for k in tot:
            tot[k] += m[k]
        total_nll += nll
        total_tokens += len(pc.target_ids)
        total_lcp += lcp
        firsts += first
        ref_tokens += len(pc.case.api) if pc.case.api is not None else 0
        per_case.append((nll, len(pc.target_ids)))
        flags = ("" if pc.render_same else " render≠") + ("" if pc.aligned else f" [{pc.why}]")
        if isinstance(pc.case.usage_prompt, int) and pc.case.usage_prompt != len(pc.prompt_ids):
            flags += f" API prompt {pc.case.usage_prompt}"
        table.append(f"  {pc.id}  {len(pc.prompt_ids):3d} {len(pc.target_ids):3d}  {_avg(nll, len(pc.target_ids)):8.4f}"
                     f"  {int(first)} {lcp:3d}  {m['top1_match']:3d}/{m['top1_count']:<3d}{flags}")
    (out / "scores.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    n = len(prep)
    avg_nll = _avg(total_nll, total_tokens)
    summary = (f"summary cases={n} tokens={total_tokens} avg_nll={avg_nll:.9f} first_match={firsts} "
               f"avg_lcp={_avg(total_lcp, n):.3f}")
    api = (f"api_summary ref_tokens={ref_tokens} target_tokens={tot['target_count']} "
           f"target_mae={_avg(tot['target_abs'], tot['target_count']):.9f} "
           f"target_mean_delta={_avg(tot['target_signed'], tot['target_count']):.9f} "
           f"top_items={tot['top_items']} top_mapped={tot['top_mapped']} "
           f"top_coverage={_avg(tot['top_mapped'], tot['top_items']):.9f} "
           f"top1_match={tot['top1_match']}/{tot['top1_count']} top1_rate={_avg(tot['top1_match'], tot['top1_count']):.9f} "
           f"topn_hit={tot['topn_hit']}/{tot['topn_ref']} topn_recall={_avg(tot['topn_hit'], tot['topn_ref']):.9f} "
           f"top_logprob_count={tot['top_logprob_count']} top_mae={_avg(tot['top_abs'], tot['top_logprob_count']):.9f} "
           f"top_mean_delta={_avg(tot['top_signed'], tot['top_logprob_count']):.9f} "
           f"pair_agree={tot['pair_agree']}/{tot['pair_total']} pair_rate={_avg(tot['pair_agree'], tot['pair_total']):.9f}")
    lo, hi = bootstrap(per_case) if n > 1 else (avg_nll, avg_nll)

    q2 = [r for r in REFERENCE if r[0].startswith("Q2")]
    q4 = [r for r in REFERENCE if r[0].startswith("Q4")]
    q2_nll, q4_nll = sum(r[1] for r in q2) / len(q2), sum(r[1] for r in q4) / len(q4)
    q2_top, q4_top = sum(r[2] for r in q2) / len(q2), sum(r[2] for r in q4) / len(q4)
    gap_nll = (avg_nll - q2_nll) / (q2_nll - q4_nll)
    top_rate = _avg(tot["top1_match"], tot["top1_count"])
    gap_top = (q2_top / 2994 - top_rate) / ((q4_top - q2_top) / 2994)
    worst = max(gap_nll, gap_top)
    full = n == 100 and total_tokens == 2994 and tot["top1_count"] == 2994
    verdict = ("FAIL" if worst > FAIL_GAP else "SUSPICIOUS" if worst > WARN_GAP else
               "within ds4's Q2 band" if min(gap_nll, gap_top) >= -WARN_GAP else
               "better than ds4's Q2 - check that it is the right build (a BF16 head, another imatrix)")
    text = [
        "Cases (prompt, answer, NLL per token, first, LCP, top-1):", *table, "", summary, api, "",
        f"Our mean NLL {avg_nll:.6f} (95% over prompts {lo:.4f}-{hi:.4f}); agreement with the API's top-1 "
        f"{tot['top1_match']}/{tot['top1_count']} = {100 * top_rate:.2f}%",
        "ds4's references (General 100):",
        *[f"  {name:30s} NLL {v:.6f}  top-1 {k}/{of} = {100 * k / of:.2f}%  | ours: ΔNLL {avg_nll - v:+.4f}"
          for name, v, k, of in REFERENCE],
        f"ds4's Q2->Q4 gap: NLL {q2_nll - q4_nll:.4f}, top-1 {q4_top - q2_top:.1f} tokens. Ours from the mean Q2: "
        f"NLL {gap_nll:+.2f} of the gap, top-1 {gap_top:+.2f} of the gap (plus - worse than Q2).",
        f"Verdict by the proposed thresholds ({WARN_GAP} / {FAIL_GAP} of the gap): {verdict}"
        + ("" if full else "  - NOT the full set or not every token aligned: not comparable with the references"),
        f"API alternatives as our ids: by the tokenizer {stats['tok']}, through the vocabulary (not UTF-8) "
        f"{stats['vocab']}, not mapped {stats['unmapped']}",
    ]
    bad_render = [pc.id for pc in prep if not pc.render_same]
    usage_diff = [pc.id for pc in prep if isinstance(pc.case.usage_prompt, int)
                  and pc.case.usage_prompt != len(pc.prompt_ids)]
    unaligned = [f"{pc.id} ({pc.why})" for pc in prep if not pc.aligned]
    def listed(xs, sep=", "):
        return "yes" if not xs else f"NO in {len(xs)}: " + sep.join(xs[:10]) + (" …" if len(xs) > 10 else "")

    text.append(f"Piecewise render = whole render with special tokens: {listed(bad_render)}")
    text.append(f"Prompt length = the API's usage.prompt_tokens: {listed(usage_diff)}")
    text.append(f"API token bytes = our token boundaries: {listed(unaligned, '; ')}")
    again = positions.get(prep[0].id + REPEAT)
    if again is not None:
        first_rows = positions[prep[0].id]
        same = [a[:3] == b[:3] and a[5] == b[5] for a, b in zip(first_rows, again)]
        dmax = max(abs(a[1] - b[1]) for a, b in zip(first_rows, again))
        text.append(f"Repeat of {prep[0].id} at the end (another expert-cache state): "
                    + ("bit for bit the same" if all(same) and dmax == 0 else
                       f"DIFFERS: max |Δlogprob| {dmax:.3g}, positions with another argmax or top "
                       f"{sum(not s for s in same)}"))
    (out / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    result = {"summary": summary, "api_summary": api, "avg_nll": avg_nll, "nll_ci95": [lo, hi],
              "top1_match": tot["top1_match"], "top1_count": tot["top1_count"], "verdict": verdict,
              "gap_nll": gap_nll, "gap_top1": gap_top, "map_stats": stats, "unaligned": unaligned,
              "render_mismatch": bad_render, "usage_mismatch": usage_diff, "meta": meta or {}}
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(text))
    # ds4's validate_scores.py: next to the set in a clone of ds4 (gguf-tools/quality-testing/), or in a
    # quality-testing folder next to it in a copy of ds4's files
    validator = next((v for v in (fixture.parent / "validate_scores.py",
                                  fixture.parent / "quality-testing" / "validate_scores.py") if v.exists()), None)
    if full and validator is not None:
        # ds4's own check of the format and of the counters' consistency (our file against itself)
        r = subprocess.run([sys.executable, "-I", str(validator), "--manifest", str(fixture / "manifest.tsv"),
                            "--baseline", str(out / "scores.tsv"), "--candidate", str(out / "scores.tsv")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"ds4's validate_scores.py: exit code {r.returncode} {r.stderr.strip()[:300]}")
    return result


# ---------------------------------------------------------------- running

def child_env(extra):
    """The environment of llama-logits: none of the caller's LLAMA_* and GGML_* (lossy modes such as DROP_* must not
    leak into an "exact" run), OMP_WAIT_POLICY=PASSIVE as in the main mode, and the --env keys on top."""
    dropped = sorted(k for k in os.environ if k.upper().startswith(("LLAMA_", "GGML_")))
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    # The fork drops light drive misses by default (an expert in neither VRAM nor RAM with little routing weight is
    # not waited for), so the exact path is set explicitly; --env goes on top (--env LLAMA_MOE_STREAM_DROP_W=default
    # keeps the fork's default, as in harness.fork_env). A fork preset (--env
    # LLAMA_MOE_STREAM_PRESET=exact|default|fast) sets the mode itself - then the exact path is not filled in. The
    # fork also cuts the decoder in prefill by default (C7: DeepSeek's YoCo truncation, the decoder half computes only
    # the last 140 prompt positions) - the exact path turns that off too (the exact preset sets it to 0 itself)
    if "LLAMA_MOE_STREAM_PRESET" not in extra:
        env["LLAMA_MOE_STREAM_DROP_W"] = "0"
        env["LLAMA_DSV4_YOCO"] = "0"
    env.update(extra)
    if env.get("LLAMA_MOE_STREAM_DROP_W") == "default":
        del env["LLAMA_MOE_STREAM_DROP_W"]
    return env, dropped


def keep_awake(on):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
    except (AttributeError, OSError):
        pass


def preflight(force):
    """The GPU is there (a card can drop off the bus) and no other run is going (a 46 GiB RAM tier is pinned whole)."""
    if force:
        return
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = ""
    if not gpu:
        sys.exit(f"no NVIDIA GPU visible (nvidia-smi: {gpu!r}) - a cold reboot may bring it back; "
                 f"--force skips this check")
    tasks = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    busy = sorted({ln.split('","')[0].strip('"') for ln in tasks.splitlines()
                   if ln.lower().startswith(('"llama-', '"nsys', '"diskspd'))})
    if busy:
        sys.exit(f"already running: {', '.join(busy)} - wait for them to finish; --force skips this check")
    print(f"   GPU: {gpu}")


def run_tool(cmd, log, env, label):
    print(f"   {label}: {subprocess.list2cmdline(map(str, cmd))}", flush=True)
    t0 = time.time()
    last = ""
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=str(Path(cmd[0]).parent), env=env, stdout=lf,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            time.sleep(10)
            try:
                tail = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                        if x.startswith("score: ")]
            except OSError:
                tail = []
            if tail and tail[-1] != last:
                last = tail[-1]
                print("   " + last, flush=True)
    print(f"   {label}: exit code {proc.returncode}, {time.time() - t0:.0f} s", flush=True)
    return proc.returncode


def git_meta():
    if REPO is None:   # SL_LLAMA_SRC not set: the commit is not recorded
        return {"fork_head": None, "fork_dirty": None}
    # GIT_OPTIONAL_LOCKS=0: status does not rewrite the index of the source tree
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=60,
                                  env=env).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {"fork_head": git("rev-parse", "HEAD"), "fork_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def cmd_run(a):
    if not a.model or not a.bin:
        sys.exit(f"set SL_MODEL and SL_LLAMA_BIN (environment or {config.FILE}), or pass --model and --bin")
    fixture, out = fixture_dir(a.fixture), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = load_fixture(fixture, a.max_cases)
    chk = check(cases, fixture)
    if chk["problems"] and not a.force:
        sys.exit("the set has problems - see above (--force scores it anyway)")
    exe = Path(a.bin) / "llama-logits.exe"
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    env, dropped = child_env(extra_env)
    if dropped:
        print(f"   not passed on from the environment: {', '.join(dropped)}")

    jobs, _ = tokenize_jobs(cases)
    write_tokenize_jobs(out / "tok_jobs.tsv", jobs)
    tok_cmd = [exe, "-m", a.model, "--tokenize-file", out / "tok_jobs.tsv", "-o", out / "tok_out.tsv"]
    score_cmd = [exe, "-m", a.model, "-ngl", "99", "-t", a.threads, "-c", a.ctx, "-ub", a.ub,
                 "--moe-stream-direct", "--moe-stream-cache", f"{a.slots}s", "--moe-stream-l2", a.l2,
                 "--top", a.top, "--score-file", out / "score_jobs.tsv", "-o", out / "positions.tsv"]
    if a.io:
        score_cmd[-4:-4] = ["--moe-stream-io-threads", a.io]
    if a.batch:
        score_cmd[-4:-4] = ["--score-batch"]
    if a.extra:
        score_cmd[-6:-6] = shlex.split(a.extra)  # before --top

    meta = {"args": vars(a), "dropped_env": dropped, "env_extra": extra_env, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fixture_sha256": chk["fixture_sha256"], "exe": str(exe), **({} if a.dry_run else git_meta())}
    if exe.exists():
        st = exe.stat()
        meta["exe_size"], meta["exe_mtime"] = st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if a.dry_run:
        print("   --dry-run: the tokenizing jobs are written; the commands:")
        print("   " + subprocess.list2cmdline(map(str, tok_cmd)))
        print("   " + subprocess.list2cmdline(map(str, score_cmd)))
        return
    if not exe.exists():
        sys.exit(f"no {exe} - build the llama-logits target of the patched tree "
                 f"(cmake --build build --target llama-logits)")

    preflight(a.force)
    keep_awake(True)
    try:
        if run_tool(tok_cmd, out / "tokenize.log", env, "tokenize") != 0:
            sys.exit("tokenizing failed - see tokenize.log")
        toks, vocab = read_tokenize_out(out / "tok_out.tsv")
        prep, stats = prepare(cases, toks, vocab)
        write_score_jobs(out / "score_jobs.tsv", prep, repeat=not a.no_repeat)
        n_eval = sum(len(pc.prompt_ids) + len(pc.target_ids) - 1 for pc in prep)
        if not a.no_repeat:
            n_eval += len(prep[0].prompt_ids) + len(prep[0].target_ids) - 1
        print(f"   {n_eval} tokens to score: at {SEC_PER_TOKEN * 1000:.0f} ms a token "
              f"~{n_eval * SEC_PER_TOKEN / 60:.1f} min + ~15 s of loading and the cache warm-up", flush=True)
        if run_tool(score_cmd, out / "score.log", env, "score") != 0:
            sys.exit("scoring failed - see score.log")
    finally:
        keep_awake(False)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    report(fixture, out, prep, stats, read_positions(out / "positions.tsv"), meta)


def cmd_report(a):
    fixture, out = fixture_dir(a.fixture), Path(a.out).resolve()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    cases = load_fixture(fixture, (meta.get("args") or {}).get("max_cases") or 0)
    toks, vocab = read_tokenize_out(out / "tok_out.tsv")
    prep, stats = prepare(cases, toks, vocab)
    report(fixture, out, prep, stats, read_positions(out / "positions.tsv"), meta)


def cmd_selftest(a):
    """A model equal to the API: ids by the bytes of the API tokens, logits = the API's logprobs. Expected: the full
    model's NLL, top-1 2994/2994, logprob error 0, top-N recall and pair order 1. Checks the parsing, the alignment and
    the metrics without binaries (llama.cpp's tokenizing itself is not checked here)."""
    fixture, out = fixture_dir(a.fixture), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = load_fixture(fixture, a.max_cases)
    ids = {s.encode(): i for i, s in enumerate(SPECIALS)}

    def tid(b):
        return ids.setdefault(b, len(ids))

    lines = ["SPEC\t4\t" + ",".join(str(tid(s.encode())) for s in SPECIALS) + "\t"
             + ",".join(s.encode().hex() for s in SPECIALS)]
    for c in cases:
        p = tid(c.prompt)       # the prompt as one "token": the prompt length is not checked here
        lines.append(f"P:{c.id}\t1\t{p}\t{c.prompt.hex()}")
        r = [tid(s.encode()) for s in SPECIALS[:2]] + [p] + [tid(s.encode()) for s in SPECIALS[2:]]
        lines.append(f"R:{c.id}\t5\t{','.join(map(str, r))}\t" + ",".join("00" for _ in r))
        tb = [b for _, b, _ in c.api]
        lines.append(f"C:{c.id}\t{len(tb)}\t{','.join(str(tid(b)) for b in tb)}\t{','.join(b.hex() for b in tb)}")
    jobs, alt_tags = tokenize_jobs(cases)
    for ab, tag in alt_tags.items():
        lines.append(f"{tag}\t1\t{tid(ab)}\t{ab.hex()}")
    for c in cases:
        for _, _, al in c.api:
            for ab, _ in al:
                if ab and not is_utf8(ab):
                    tid(ab)
    lines += [f"VOCAB\t1\t{i}\t{b.hex()}" for b, i in ids.items()]
    (out / "tok_out.tsv").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    toks, vocab = read_tokenize_out(out / "tok_out.tsv")
    prep, stats = prepare(cases, toks, vocab)
    pos = []
    for pc in prep:
        for i, (lp, b, al) in enumerate(pc.case.api):
            mapped = sorted(((alp, tok) for (_, alp), tok in zip(al, pc.alt_ids[i]) if tok >= 0),
                            key=lambda x: (-x[0], x[1]))
            # the real llama-logits gives the top 64 of the whole vocabulary; here the missing ones are ids outside it
            top = [tok for _, tok in mapped] + [10**7 + k for k in range(TOP_CAP - len(mapped))]
            ex = ",".join(f"{tok}:{alp!r}" for alp, tok in mapped)
            pos.append(f"{pc.id}\t{i}\t{pc.target_ids[i]}\t{lp!r}\t{top[0]}\t{mapped[0][0]!r}\t0\t"
                       f"{','.join(map(str, top))}\t{ex}")
    (out / "positions.tsv").write_text("\n".join(pos) + "\n", encoding="ascii", newline="\n")
    res = report(fixture, out, prep, stats, read_positions(out / "positions.tsv"), {"selftest": True})
    print(f"selftest: mean NLL {res['avg_nll']:.6f} - must be the full model's NLL from check")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("fixture")
    for name in ("run", "report", "selftest"):
        s = sub.add_parser(name)
        s.add_argument("fixture")
        s.add_argument("out")
        if name != "report":
            s.add_argument("--max-cases", type=int, default=0)
        if name == "run":
            s.add_argument("--model", default=str(MODEL) if MODEL else None)
            s.add_argument("--bin", default=str(LLAMA_BIN) if LLAMA_BIN else None)
            s.add_argument("--ctx", default="4096")
            s.add_argument("--ub", default="1")
            s.add_argument("--batch", action="store_true")
            s.add_argument("--slots", default=str(slots_for(17)))   # 23 on cd, 25 on qd
            s.add_argument("--l2", default="46")
            s.add_argument("--io", default="")
            s.add_argument("--threads", default="6")
            s.add_argument("--top", default=str(TOP_CAP))
            s.add_argument("--env", action="append", default=[])
            # any llama-logits flags of a mode, e.g. --extra "--override-kv deepseek41.expert_used_count=int:5"
            # or --extra "-ctk q8_0 -ctv q8_0"; they go before --top, after the stream flags, so a repeated flag wins
            s.add_argument("--extra", default="")
            s.add_argument("--no-repeat", action="store_true")
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.cmd == "check":
        fixture = fixture_dir(a.fixture)
        res = check(load_fixture(fixture), fixture)
        sys.exit(1 if res["problems"] else 0)
    {"run": cmd_run, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
