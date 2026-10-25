"""Paths and settings of the benchmark scripts.

Every value comes from an environment variable or from bench/config.json (the environment wins), for example:

    {"SL_LLAMA_BIN": "D:/llama.cpp/build/bin",
     "SL_LLAMA_SRC": "D:/llama.cpp",
     "SL_MODEL": "D:/models/v41-cd/DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf",
     "SL_MIRRORS": "G:/models/v41-cd-copy"}

  SL_LLAMA_BIN   folder of the patched llama.cpp build (llama-server, llama-perplexity, llama-logits)
  SL_LLAMA_SRC   the patched llama.cpp source tree: long documents and agent turns are cut from its ggml sources
  SL_MODEL       the first GGUF part of the build (...-00001-of-00011.gguf)
  SL_BUILD       cd (default) or qd - the build in SL_MODEL; sets the VRAM expert slots
  SL_MIRRORS     ';'-separated folders with a copy of the expert parts on other drives (optional, up to three)
  SL_RESULTS     results and server logs (default: bench/results)
  SL_SLOTS_DIR   where the server saves KV slots for consult.py slot / restore (default: SL_RESULTS/slots)
  SL_REF         root of the quality anchors: continuations collected from the API by api_collect.py
  SL_NSYS        nsys.exe of Nsight Systems (profiling scripts only)
  SL_PORT        server port (default 8091)

The scripts are Windows-only as written (process groups, PowerShell counters, SetThreadExecutionState).
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = HERE / "config.json"
_CFG = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}


def get(key, default=None):
    v = os.environ.get(key)
    if v is None or v == "":
        v = _CFG.get(key, default)
    return v


def path(key, default=None):
    """A path setting, or None when it is not set."""
    v = get(key, default)
    return Path(v) if v else None


def need(key):
    """A path setting that must be set."""
    p = path(key)
    if p is None:
        raise SystemExit(f"{key} is not set: put it in the environment or in {FILE}")
    return p


def exe(name):
    """A binary of the patched build, e.g. exe('llama-server') -> <SL_LLAMA_BIN>/llama-server.exe (None if unset)."""
    d = path("SL_LLAMA_BIN")
    return d / (name + (".exe" if os.name == "nt" else "")) if d else None


BUILD = get("SL_BUILD", "cd")
MODEL = path("SL_MODEL")
MIRRORS = [Path(p) for p in (get("SL_MIRRORS", "") or "").split(";") if p.strip()]
RESULTS = path("SL_RESULTS") or HERE / "results"
LOGS = RESULTS / "logs"
SLOTS_DIR = path("SL_SLOTS_DIR") or RESULTS / "slots"
PORT = int(get("SL_PORT", 8091))
