"""Draw the README's charts from docs/data/*.csv into docs/img/ - a light and a dark SVG per chart.

    python tools/make_charts.py

Plain SVG, no dependencies. Thin marks, a 2px surface gap between stacked segments, rounded data ends, hairline
grid, system sans. Colours: a categorical palette validated for colour-vision deficiency on both surfaces
(blue / aqua / orange, light and dark steps) and a neutral grey for "waiting / idle". Every chart's numbers are in
its CSV file (docs/data/README.md says where they come from).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
IMG = ROOT / "docs" / "img"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", blue="#2a78d6", aqua="#1baf7a", orange="#eb6834", idle="#898781"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", blue="#3987e5", aqua="#199e70", orange="#d95926", idle="#898781"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
W = 760
LEFT, RIGHT = 64, 28
GAP = 2      # surface gap between stacked segments
R = 4        # rounded data end


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def f(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def text(x, y, s, size, fill, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}>'
            f"{esc(s)}</text>")


def rect(x, y, w, h, fill):
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="{fill}"/>'


def bar_end_right(x, y, w, h, fill, r=R):
    """A horizontal segment whose right end is the data end: square on the left, rounded on the right."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{f(x)},{f(y)} h{f(w - r)} a{r},{r} 0 0 1 {r},{r} v{f(h - 2 * r)} '
            f'a{r},{r} 0 0 1 {-r},{r} h{f(-(w - r))} z" fill="{fill}"/>')


def bar_end_top(x, y, w, h, fill, r=R):
    """A vertical segment whose top is the data end: square at the bottom, rounded at the top."""
    r = min(r, w / 2, h)
    return (f'<path d="M{f(x)},{f(y + h)} v{f(-(h - r))} a{r},{r} 0 0 1 {r},{-r} h{f(w - 2 * r)} '
            f'a{r},{r} 0 0 1 {r},{r} v{f(h - r)} z" fill="{fill}"/>')


def header(t, title, subtitle):
    out = [text(LEFT - 40, 30, title, 15, t["ink"], weight=600)]
    y = 50
    for line in subtitle:
        out.append(text(LEFT - 40, y, line, 12.5, t["ink2"]))
        y += 17
    return out, y


def legend(t, items, y, x0=None):
    """items: (colour, label); a swatch beside ink-coloured text, never coloured text."""
    out, x = [], LEFT - 40 if x0 is None else x0
    for colour, label in items:
        out.append(f'<rect x="{f(x)}" y="{f(y - 9)}" width="10" height="10" rx="2" fill="{colour}"/>')
        out.append(text(x + 15, y, label, 12, t["ink2"]))
        x += 15 + 6.6 * len(label) + 22
    return out


def svg(t, h, body, title, desc):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" '
        f'aria-labelledby="t d" font-family="{FONT}">',
        f'<title id="t">{esc(title)}</title>',
        f'<desc id="d">{esc(desc)}</desc>',
        f'<rect width="{W}" height="{h}" rx="8" fill="{t["surface"]}"/>',
        *body,
        "</svg>",
    ]) + "\n"


def read_csv(name):
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def decode_token(t):
    rows = read_csv("decode-token.csv")
    seg = {r["segment"]: float(r["ms"]) for r in rows}
    nvme = float(next(r["of_which_nvme_wait_ms"] for r in rows if r["of_which_nvme_wait_ms"]))
    parts = [("compute only", seg["compute only"], t["blue"]),
             ("compute and copies at once", seg["compute and copies at once"], t["aqua"]),
             ("expert copies only", seg["expert copies only"], t["orange"]),
             ("idle", seg["idle"], t["idle"])]
    total = sum(p[1] for p in parts)
    title = "Where a decode token's time goes"
    body, y = header(t, title, [
        f"GPU timeline of one token under Nsight: {total:.0f} ms (about 49-50 ms without the profiler).",
        "For half of it the GPU computes nothing and only copies experts from RAM over PCIe 4.0.",
    ])
    body += legend(t, [(c, n) for n, _, c in parts], y + 12)
    top = y + 36
    plot_w = W - LEFT - RIGHT
    scale = plot_w / 55.0
    x = LEFT
    for i, (name, ms, colour) in enumerate(parts):
        w = ms * scale - (GAP if i else 0)
        x0 = x + (GAP if i else 0)
        if i == len(parts) - 1:
            body.append(bar_end_right(x0, top, w, 24, colour))
        else:
            body.append(rect(x0, top, w, 24, colour))
        body.append(text(x0 + w / 2, top + 42, f"{ms:.1f} ms", 12, t["ink"], "middle", 600))
        x += ms * scale
    # axis
    ay = top + 62
    body.append(f'<line x1="{LEFT}" y1="{ay}" x2="{f(LEFT + plot_w)}" y2="{ay}" stroke="{t["axis"]}" '
                'stroke-width="1"/>')
    for v in range(0, 60, 10):
        xv = LEFT + v * scale
        body.append(f'<line x1="{f(xv)}" y1="{ay}" x2="{f(xv)}" y2="{ay + 4}" stroke="{t["axis"]}" '
                    'stroke-width="1"/>')
        body.append(text(xv, ay + 17, f"{v}", 11, t["muted"], "middle"))
    body.append(text(LEFT + plot_w, ay + 17, "ms", 11, t["muted"], "end"))
    idle_x = LEFT + (total - seg["idle"] / 2) * scale
    body.append(text(idle_x, top - 8, f"{nvme:.1f} ms of it: waiting for NVMe", 11, t["ink2"], "middle"))
    h = ay + 40
    body.append(text(LEFT - 40, h - 10, "Default preset, short context, 223 tokens; data: docs/data/decode-token.csv",
                     10.5, t["muted"]))
    desc = "; ".join(f"{n} {f(ms)} ms" for n, ms, _ in parts) + f"; idle includes {f(nvme)} ms waiting for NVMe."
    return svg(t, h, body, title, desc)


def turn_layers(t):
    rows = read_csv("turn-layers.csv")
    title = "A warm agent turn, layer by layer"
    enc = [r for r in rows if int(r["layer"]) < 20]
    dec = [r for r in rows if int(r["layer"]) >= 20]

    def total(rs):
        return sum(float(r["outside_waves_ms"]) + float(r["waves_ms"]) for r in rs) / 1000

    body, y = header(t, title, [
        "Reading 7,530 new tokens of an agent turn (context 16.8K to 24.3K): the main pass, per layer.",
        "Layers 20-39 run their experts only for the last 140 prompt positions, yet their waves still wait for loads.",
    ])
    series = [(t["blue"], "attention, dense layers, norms"), (t["orange"], "expert waves: compute and copies"),
              (t["idle"], "expert waves: waiting for loads")]
    body += legend(t, series, y + 12)
    top = y + 34
    ph = 220
    plot_w = W - LEFT - RIGHT
    ymax = 400.0
    sy = ph / ymax
    base = top + ph
    for v in range(0, 401, 100):
        yy = base - v * sy
        stroke = t["axis"] if v == 0 else t["grid"]
        body.append(f'<line x1="{LEFT}" y1="{f(yy)}" x2="{f(LEFT + plot_w)}" y2="{f(yy)}" stroke="{stroke}" '
                    'stroke-width="1"/>')
        body.append(text(LEFT - 8, yy + 4, f"{v}", 11, t["muted"], "end"))
    body.append(text(LEFT - 8, top - 10, "ms", 11, t["muted"], "end"))
    slot = plot_w / len(rows)
    bw = 12
    for r in rows:
        i = int(r["layer"])
        outside = float(r["outside_waves_ms"])
        wait = float(r["waves_waiting_ms"])
        work = float(r["waves_ms"]) - wait
        x = LEFT + i * slot + (slot - bw) / 2
        ycur = base
        segs = [(outside, t["blue"]), (work, t["orange"]), (wait, t["idle"])]
        for k, (v, colour) in enumerate(segs):
            hgt = v * sy - (GAP if k else 0)
            ytop = ycur - v * sy
            if k == len(segs) - 1:
                body.append(bar_end_top(x, ytop, bw, hgt, colour))
            else:
                body.append(rect(x, ytop, bw, hgt, colour))
            ycur = ytop
        if i % 5 == 0 or i == 39:
            body.append(text(x + bw / 2, base + 15, f"{i}", 11, t["muted"], "middle"))
    # encoder / decoder brackets
    by = base + 30
    for lo, hi, label in ((0, 19, f"layers 0-19 (encoder): {total(enc):.1f} s"),
                          (20, 39, f"layers 20-39 (decoder): {total(dec):.1f} s")):
        x1 = LEFT + lo * slot + 3
        x2 = LEFT + (hi + 1) * slot - 3
        body.append(f'<path d="M{f(x1)},{by - 5} v5 h{f(x2 - x1)} v-5" fill="none" stroke="{t["axis"]}" '
                    'stroke-width="1"/>')
        body.append(text((x1 + x2) / 2, by + 16, label, 12, t["ink2"], "middle"))
    l20 = next(r for r in rows if int(r["layer"]) == 20)
    x20 = LEFT + 20 * slot + slot / 2
    y20 = base - (float(l20["outside_waves_ms"]) + float(l20["waves_ms"])) * sy
    body.append(text(x20, y20 - 7, "layer 20", 11, t["ink2"], "middle"))
    h = by + 44
    body.append(text(LEFT - 40, h - 10, "Pass 1 of turn 7, default preset; data: docs/data/turn-layers.csv",
                     10.5, t["muted"]))
    desc = (f"Per-layer time of the main pass. Encoder layers 0-19 total {total(enc):.2f} s, decoder layers 20-39 "
            f"{total(dec):.2f} s; every fourth layer has a heavier attention part; layer 20 is the heaviest.")
    return svg(t, h, body, title, desc)


def agent_turns(t):
    rows = read_csv("agent-turns.csv")
    pts = []
    for r in rows:
        if r["turn"] == "1":
            continue        # the cold first turn is not part of the fit
        n = int(r["new_tokens"])
        read = (float(r["read_s_a"]) + float(r["read_s_b"])) / 2
        wait = (float(r["wave_wait_s_a"]) + float(r["wave_wait_s_b"])) / 2
        pts.append((n, read, wait))
    pts.sort()
    k = len(pts)
    mx = sum(p[0] for p in pts) / k
    my = sum(p[1] for p in pts) / k
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    a = my - b * mx
    title = "Reading an agent turn: a fixed cost plus a slope"
    body, y = header(t, title, [
        f"Turns 2-8 of an 8-turn agent session (warm RAM tier). Least squares: {a:.1f} s + {b * 1000:.2f} ms "
        "per new token;",
        "waiting for expert loads takes 2-3 s whatever the turn's size.",
    ])
    body += legend(t, [(t["idle"], "waiting for expert loads"),
                       (t["blue"], "everything else (compute, Engram rows, setup)")], y + 12)
    top = y + 34
    ph = 200
    plot_w = W - LEFT - RIGHT
    xmax, ymax = 8000.0, 15.0
    sx, sy = plot_w / xmax, ph / ymax
    base = top + ph
    for v in range(0, 16, 5):
        yy = base - v * sy
        stroke = t["axis"] if v == 0 else t["grid"]
        body.append(f'<line x1="{LEFT}" y1="{f(yy)}" x2="{f(LEFT + plot_w)}" y2="{f(yy)}" stroke="{stroke}" '
                    'stroke-width="1"/>')
        body.append(text(LEFT - 8, yy + 4, f"{v}", 11, t["muted"], "end"))
    body.append(text(LEFT - 8, top - 10, "s", 11, t["muted"], "end"))
    for v in range(0, 8001, 2000):
        xv = LEFT + v * sx
        body.append(text(xv, base + 17, "0" if v == 0 else f"{v // 1000}K", 11, t["muted"], "middle"))
    body.append(text(LEFT + plot_w, base + 34, "new tokens in the turn", 11, t["muted"], "end"))
    bw = 22
    for i, (n, read, wait) in enumerate(pts):
        x = LEFT + n * sx - bw / 2
        body.append(rect(x, base - wait * sy, bw, wait * sy, t["idle"]))
        rest = read - wait
        body.append(bar_end_top(x, base - read * sy, bw, rest * sy - GAP, t["blue"]))
        if i in (0, k - 1):
            body.append(text(x + bw / 2, base - read * sy - 8, f"{read:.1f} s", 11.5, t["ink"], "middle", 600))
    # the fit line
    x1, x2 = LEFT, LEFT + xmax * sx
    y1, y2 = base - a * sy, base - (a + b * xmax) * sy
    body.append(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{t["ink2"]}" '
                'stroke-width="1.5" stroke-linecap="round"/>')
    # the fit's label goes in the widest gap between columns, just under the line
    gaps = [(pts[i + 1][0] - pts[i][0], (pts[i][0] + pts[i + 1][0]) / 2) for i in range(k - 1)]
    nmid = max(gaps)[1]
    body.append(text(LEFT + nmid * sx, base - (a + b * nmid) * sy + 16, "least-squares fit", 11, t["ink2"],
                     "middle"))
    h = base + 56
    body.append(text(LEFT - 40, h - 10, "Mean of two runs, default preset; data: docs/data/agent-turns.csv",
                     10.5, t["muted"]))
    desc = ("Reading time of agent turns 2-8 against their new tokens: " +
            "; ".join(f"{n} tokens {read:.2f} s (waiting {wait:.2f} s)" for n, read, wait in pts) +
            f". Fit: {a:.2f} s + {b * 1000:.3f} ms per token.")
    return svg(t, h, body, title, desc)


def drives(t):
    rows = read_csv("drives.csv")
    configs = list(dict.fromkeys(r["config"] for r in rows))
    order = list(dict.fromkeys(r["drive"] for r in rows))
    colour = dict(zip(order, (t["blue"], t["aqua"], t["orange"])))
    title = "Expert reads from one, two and three NVMe drives"
    body, y = header(t, title, [
        "Unbuffered reads of one expert slab (3.4 MB) at a time, 12 / 4 / 4 in flight per drive (diskspd).",
        "The drives do not slow each other down; prompts read faster with each copy, decode waits for latency instead.",
    ])
    body += legend(t, [(colour[d], d) for d in order], y + 12)
    top = y + 34
    plot_w = W - LEFT - RIGHT - 60
    scale = plot_w / 24.0
    bh, gap_y = 22, 18
    label = {"F12": "one drive", "F12+E4": "two", "F12+E4+C4": "three"}
    for i, cfg in enumerate(configs):
        yy = top + i * (bh + gap_y)
        body.append(text(LEFT + 70, yy + bh / 2 + 4, label.get(cfg, cfg), 12, t["ink2"], "end"))
        x = LEFT + 80
        parts = [r for r in rows if r["config"] == cfg]
        for k, r in enumerate(parts):
            w = float(r["gb_s"]) * scale - (GAP if k else 0)
            x0 = x + (GAP if k else 0)
            if k == len(parts) - 1:
                body.append(bar_end_right(x0, yy, w, bh, colour[r["drive"]]))
            else:
                body.append(rect(x0, yy, w, bh, colour[r["drive"]]))
            x += float(r["gb_s"]) * scale
        total = sum(float(r["gb_s"]) for r in parts)
        body.append(text(x + 8, yy + bh / 2 + 4, f"{total:.1f} GB/s", 12, t["ink"], "start", 600))
    h = top + len(configs) * (bh + gap_y) + 20
    body.append(text(LEFT - 40, h - 10, "data: docs/data/drives.csv", 10.5, t["muted"]))
    desc = "; ".join(f"{label.get(c, c)}: " + ", ".join(f"{r['drive']} {r['gb_s']} GB/s" for r in rows if r["config"] == c)
                     for c in configs)
    return svg(t, h, body, title, desc)


CHARTS = {"decode-token": decode_token, "turn-layers": turn_layers, "agent-turns": agent_turns, "drives": drives}


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for name, draw in CHARTS.items():
        for mode, t in THEMES.items():
            (IMG / f"{name}-{mode}.svg").write_text(draw(t), encoding="utf-8", newline="\n")
    print(f"{len(CHARTS)} charts x {len(THEMES)} themes -> {IMG}")


if __name__ == "__main__":
    main()
