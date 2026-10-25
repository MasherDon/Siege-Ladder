"""Assemble the weights of DeepSeek-V4.1-Flash-1017 (11 GGUF parts: the default "cd" build, 373 GB, or the fast
preset's "qd", 362 GB) from public Hugging Face repositories, following template/<build>/manifest.json.

    python tools/assemble.py OUT_DIR --plan              # what would be fetched and written; touches nothing
    python tools/assemble.py OUT_DIR --build qd          # the fast preset's lighter build instead of cd
    python tools/assemble.py OUT_DIR                     # fetch by HTTP range requests and write the 11 parts
    python tools/assemble.py OUT_DIR --sources DIR ...   # read whole source files from DIR(s) when present there
    python tools/assemble.py OUT_DIR --sources DIR --verify-sources   # first check those files' SHA-256
    python tools/assemble.py OUT_DIR --check             # compare the start and end of every tensor with its source
    python tools/assemble.py OUT_DIR --check --full      # SHA-256 of every part against our own build's

Every tensor is copied byte for byte from its source file (repository, pinned revision, file, offset - all in the
manifest); the GGUF headers and part 1 come from template/. Bytes between tensors are zeros. Resumable: progress is kept
in OUT_DIR/assemble.state.json, a part is written to NAME.tmp and renamed when complete; run the same command again
after an interruption. Set HF_TOKEN to send a Hugging Face token (the sources are public; a token raises rate limits;
it is sent to huggingface.co only, not to the storage it redirects to).

Needs ~373 GB free in OUT_DIR. No third-party packages.
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "template"
TEMPLATE = TEMPLATES / "cd"     # set by --build
CHUNK = 64 << 20
SAVE_EVERY = 20      # seconds between flushing the part and saving the progress state
PROBE = 64 << 10     # --check compares this many bytes at each end of every tensor


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


class Source:
    def __init__(self, d, dirs):
        self.repo, self.rev, self.path, self.size, self.sha256 = d["repo"], d["revision"], d["path"], d["size"], d["sha256"]
        self.name = self.path.rsplit("/", 1)[-1]
        self.url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{self.path}"
        self.local = None
        for base in dirs:
            for cand in (Path(base) / self.path, Path(base) / self.name):
                if cand.is_file() and cand.stat().st_size == self.size:
                    self.local = cand
                    break
            if self.local:
                break

    def _get(self, off, n):
        rng = {"Range": f"bytes={off}-{off + n - 1}", "User-Agent": "siege-ladder-assemble"}
        first = dict(rng)
        if os.environ.get("HF_TOKEN"):
            first["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        try:
            r = OPENER.open(urllib.request.Request(self.url, headers=first), timeout=120)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308) or not e.headers.get("Location"):
                raise
            loc = urllib.request.urljoin(self.url, e.headers["Location"])
            r = urllib.request.urlopen(urllib.request.Request(loc, headers=rng), timeout=120)
        with r:
            if r.status != 206:
                raise OSError(f"no range support (status {r.status})")
            return r.read()

    def read(self, off, n):
        if self.local:
            with open(self.local, "rb") as f:
                f.seek(off)
                b = f.read(n)
            if len(b) != n:
                raise OSError(f"{self.local}: short read at {off}")
            return b
        delay = 2
        for attempt in range(10):
            try:
                b = self._get(off, n)
                if len(b) != n:
                    raise OSError(f"got {len(b)} of {n} bytes")
                return b
            except (OSError, urllib.error.URLError) as e:
                if attempt == 9:
                    raise
                print(f"  {self.name} @{off}: {e}; retry in {delay} s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def chunks(part):
    """Deterministic work list of a part: (chunk id, tensor, offset in tensor, length)."""
    out = []
    for t in part["tensors"]:
        for o in range(0, t["size"], CHUNK):
            out.append((len(out), t, o, min(CHUNK, t["size"] - o)))
    return out


def gaps(part):
    """Byte ranges of a part that hold zeros: between tensors and after the last one."""
    pos, out = part["header_size"], []
    for t in sorted(part["tensors"], key=lambda t: t["offset"]):
        if t["offset"] > pos:
            out.append((pos, t["offset"] - pos))
        pos = t["offset"] + t["size"]
    if part["size"] > pos:
        out.append((pos, part["size"] - pos))
    return out


class State:
    """Chunks written per part. Saved only after the part file is flushed, so a saved chunk is on disk."""

    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.lock = threading.Lock()
        self.pending = {}

    def done(self, part):
        return set(self.data.get(part, []))

    def mark(self, part, cid):
        with self.lock:
            self.pending.setdefault(part, []).append(cid)

    def flush_and_save(self, part, tmp):
        with self.lock:
            fresh = self.pending.pop(part, [])
        with open(tmp, "r+b") as f:
            os.fsync(f.fileno())
        self.data.setdefault(part, []).extend(fresh)
        self.save()

    def forget(self, part):
        self.data.pop(part, None)
        self.pending.pop(part, None)
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        os.replace(tmp, self.path)


def write_part(part, sources, out, state, workers):
    final = out / part["file"]
    tmp = out / (part["file"] + ".tmp")
    work = chunks(part)
    if not tmp.exists():
        state.forget(part["file"])
        with open(tmp, "wb") as f:
            f.write((TEMPLATE / part["template"]).read_bytes())
            f.truncate(part["size"])
    done = state.done(part["file"])
    todo = [w for w in work if w[0] not in done]
    total = sum(w[3] for w in todo)
    print(f"{part['file']}: {human(part['size'])}, {len(part['tensors'])} tensors, "
          f"{len(work) - len(todo)} of {len(work)} chunks already written, {human(total)} to go", flush=True)
    q, errors, written = queue.Queue(maxsize=workers * 2), [], [0]
    t0 = time.time()

    def worker():
        with open(tmp, "r+b", buffering=0) as f:
            while True:
                item = q.get()
                if item is None:
                    return
                cid, t, o, n = item
                try:
                    b = sources[t["source"]].read(t["source_offset"] + o, n)
                    f.seek(t["offset"] + o)
                    f.write(b)
                    state.mark(part["file"], cid)
                    with state.lock:
                        written[0] += n
                except Exception as e:   # noqa: BLE001 - reported below; the part stays .tmp and can resume
                    errors.append(f"{t['name']} @{o}: {e}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for th in threads:
        th.start()
    last = time.time()
    try:
        for item in todo:
            if errors:
                break
            q.put(item)
            if time.time() - last > SAVE_EVERY:
                state.flush_and_save(part["file"], tmp)
                el = time.time() - t0
                print(f"  {human(written[0])} of {human(total)} ({human(written[0] / el)}/s)", flush=True)
                last = time.time()
    finally:
        for _ in threads:
            q.put(None)
        for th in threads:
            th.join()
        state.flush_and_save(part["file"], tmp)
    if errors:
        sys.exit(f"{part['file']}: {len(errors)} chunks failed, e.g. {errors[0]}; run again to resume")
    if len(state.done(part["file"])) != len(work):
        sys.exit(f"{part['file']}: {len(work) - len(state.done(part['file']))} chunks missing; run again")
    with open(tmp, "rb") as f:
        head = f.read(part["header_size"])
    if hashlib.sha256(head).hexdigest() != part["header_sha256"]:
        sys.exit(f"{tmp}: header differs from the template")
    os.replace(tmp, final)
    state.forget(part["file"])
    print(f"  done in {time.time() - t0:.0f} s", flush=True)


def check(m, sources, out, full=False):
    bad = 0
    for p in m["parts"]:
        f = out / p["file"]
        if not f.exists() or f.stat().st_size != p["size"]:
            print(f"{p['file']}: missing or wrong size")
            bad += 1
            continue
        if "tensors" not in p or (full and p.get("sha256")):
            ok = sha256_file(f) == p["sha256"]
            print(f"{p['file']}: {'ok (SHA-256)' if ok else 'DIFFERS from our build (SHA-256)'}", flush=True)
            bad += not ok
            continue
        errs = []
        with open(f, "rb") as fh:
            if hashlib.sha256(fh.read(p["header_size"])).hexdigest() != p["header_sha256"]:
                errs.append("header")
            for off, n in gaps(p):
                fh.seek(off)
                if fh.read(min(n, 1 << 20)).strip(b"\0"):
                    errs.append(f"non-zero gap at {off}")
            for t in p["tensors"]:
                for o in sorted({0, max(0, t["size"] - PROBE)}):
                    n = min(PROBE, t["size"] - o)
                    fh.seek(t["offset"] + o)
                    if fh.read(n) != sources[t["source"]].read(t["source_offset"] + o, n):
                        errs.append(f"{t['name']} @{o}")
        print(f"{p['file']}: {'ok' if not errs else 'BAD: ' + ', '.join(errs[:5])}", flush=True)
        bad += bool(errs)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--sources", nargs="*", default=[], help="folders with whole source files")
    ap.add_argument("--verify-sources", action="store_true", help="check SHA-256 of the local source files first")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--check", action="store_true", help="compare both ends of every tensor with its source")
    ap.add_argument("--full", action="store_true", help="with --check: SHA-256 of every part against our build's")
    ap.add_argument("--workers", type=int, default=8, help="parallel range requests or reads (default 8)")
    ap.add_argument("--build", default="cd", choices=sorted(d.name for d in TEMPLATES.iterdir() if d.is_dir()),
                    help="cd (default) or qd (the fast preset's build)")
    a = ap.parse_args()
    global TEMPLATE
    TEMPLATE = TEMPLATES / a.build
    m = json.loads((TEMPLATE / "manifest.json").read_text(encoding="utf-8"))
    sources = [Source(d, a.sources) for d in m["sources"]]
    if a.check:
        check(m, sources, a.out, a.full)
    need = {}
    for p in m["parts"]:
        for t in p.get("tensors", []):
            need[t["source"]] = need.get(t["source"], 0) + t["size"]
    print(f"{m['about']}\n")
    for i, s in enumerate(sources):
        where = f"local {s.local}" if s.local else "HTTP range requests"
        print(f"  {s.repo} @ {s.rev[:7]}  {s.path}: {human(need.get(i, 0))} of {human(s.size)} from {where}")
    total = sum(p["size"] for p in m["parts"])
    remote = sum(n for i, n in need.items() if not sources[i].local)
    print(f"\n{len(m['parts'])} parts, {human(total)} to write, {human(remote)} to download")
    if a.plan:
        return
    if a.verify_sources:
        for s in sources:
            if s.local:
                print(f"SHA-256 {s.local} ...", flush=True)
                if sha256_file(s.local) != s.sha256:
                    sys.exit(f"{s.local}: SHA-256 differs from {s.repo} @ {s.rev[:7]}")
    a.out.mkdir(parents=True, exist_ok=True)
    left = sum(p["size"] for p in m["parts"] if not (a.out / p["file"]).exists())
    if shutil.disk_usage(a.out).free < left:
        print(f"warning: {human(shutil.disk_usage(a.out).free)} free in {a.out}, the parts take {human(left)}",
              flush=True)
    state = State(a.out / "assemble.state.json")
    for p in m["parts"]:
        final = a.out / p["file"]
        if final.exists() and final.stat().st_size == p["size"]:
            print(f"{p['file']}: present")
            continue
        if "tensors" not in p:
            data = (TEMPLATE / p["template"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != p["sha256"]:
                sys.exit(f"template/{p['template']} is damaged")
            final.write_bytes(data)
            print(f"{p['file']}: copied from the template")
            continue
        write_part(p, sources, a.out, state, a.workers)
    print(f"\nall {len(m['parts'])} parts in {a.out}; point llama-server -m at {m['parts'][0]['file']}")


if __name__ == "__main__":
    main()
