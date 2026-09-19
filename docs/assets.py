"""Regenerates the logo and figure. Text in the logo is outlined so it renders the same everywhere."""
import json
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTCollection

HERE = Path(__file__).resolve().parent


def outline(text, face="AvenirNext-DemiBold", size=48, tracking=-0.5):
    coll = TTCollection("/System/Library/Fonts/Avenir Next.ttc")
    font = next(f for f in coll.fonts if f["name"].getDebugName(6) == face)
    gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    upem = font["head"].unitsPerEm
    k = size / upem
    x, paths, lo, hi = 0.0, [], 1e9, -1e9
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(pen)
        bp = BoundsPen(gs)
        gs[g].draw(bp)
        if bp.bounds:
            lo, hi = min(lo, bp.bounds[1]), max(hi, bp.bounds[3])
        paths.append(f'<path transform="translate({x:.2f} 0) scale({k:.5f} {-k:.5f})" d="{pen.getCommands()}"/>')
        x += hmtx[g][0] * k + tracking
    return "".join(paths), x, lo * k, hi * k


# The logo is a supplied image (docs/brier.png, docs/brier-dark.png); it is not generated here.

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
# Ink and hairlines carry structure; exactly two chromatic series, validated for colour-vision separation:
# indigo = Brier / trained, amber = baseline / untrained.
INK, MUTED, FAINT, LINE, SURFACE = "#0F172A", "#5B6472", "#8B93A1", "#E4E7EC", "#FFFFFF"
BLUE, VIOLET = "#4F46E5", "#B45309"
PALE = "#FFFFFF"
ACCENT, BASE = BLUE, VIOLET


def figure():
    """Method schematic: state and questions in, one forward pass, a distribution per question out."""
    o = _svg(900, 400)
    _caps(o, 40, 34, "Method")
    _t(o, 40, 58, "One forward pass scores every allowed answer to every question", 16, INK, "700")
    _t(o, 40, 78, "No text is generated. A value outside the allowed set cannot occur.", 12, MUTED)

    # column 1: input
    _caps(o, 40, 120, "State + questions")
    _t(o, 40, 142, "“Webhooks fire twice since 3am.", 12.5, INK)
    _t(o, 40, 160, "Need this today.”", 12.5, INK)
    qs = (("category", "billing · bug · feature · account · other"), ("priority", "low · medium · high · urgent"), ("needs_human", "yes · no"))
    for i, (n, v) in enumerate(qs):
        y = 196 + i * 40
        _t(o, 40, y, n, 12, INK, "700")
        _t(o, 40, y + 15, v, 10.5, MUTED)
    # column 2: candidate rows into one pass
    x2 = 330
    _caps(o, x2, 120, "Batched candidates")
    rows = [("category", 5), ("priority", 4), ("needs_human", 2)]
    y = 138
    for name, n in rows:
        for k in range(n):
            _hair(o, x2, y, x2 + 130, y)
            o.append(f'<rect x="{x2 + 96}" y="{y - 5}" width="34" height="10" rx="2" fill="{ACCENT}" opacity="{0.35 + 0.65 * (k == 0)}"/>')
            y += 13
        y += 8
    _t(o, x2, y + 6, "11 rows · identical prompt · one candidate tail each", 10.5, MUTED)
    o.append(f'<line x1="{x2 + 140}" y1="{(138 + y) / 2 - 8}" x2="{x2 + 186}" y2="{(138 + y) / 2 - 8}" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#ar)"/>')
    # the pass
    x3 = 526
    o.append(f'<rect x="{x3}" y="150" width="112" height="110" fill="none" stroke="{INK}" stroke-width="1.4"/>')
    _t(o, x3 + 56, 196, "forward", 13, INK, "700", "middle")
    _t(o, x3 + 56, 214, "pass", 13, INK, "700", "middle")
    _t(o, x3 + 56, 238, "log P(answer | state)", 10, MUTED, "400", "middle")
    o.append(f'<line x1="{x3 + 118}" y1="205" x2="{x3 + 164}" y2="205" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#ar)"/>')
    # column 4: distributions
    x4 = 700
    _caps(o, x4, 120, "Distributions")
    dists = (("category", (("bug", 0.86), ("billing", 0.07), ("account", 0.05))), ("priority", (("high", 0.59), ("urgent", 0.28), ("medium", 0.12))), ("needs_human", (("no", 0.94), ("yes", 0.06))))
    y = 140
    for name, items in dists:
        _t(o, x4, y, name, 11.5, INK, "700")
        y += 8
        for lab, pv in items:
            _bar(o, x4, y, 120 * pv, 8, ACCENT if pv == items[0][1] else LINE)
            _t(o, x4 + 128, y + 8, f"{lab}  {pv:.2f}", 10.5, INK if pv == items[0][1] else MUTED)
            y += 15
        y += 12
    _hair(o, 40, 340, 860, 340)
    _t(o, 40, 366, "Latency is one pass over a short batch; it grows with the number of allowed answers, not with output length.", 11, MUTED)
    _t(o, 40, 384, "Questions are scored independently in the same pass. The backbone is any open causal language model.", 11, MUTED)
    o.append("</svg>")
    (HERE / "how-it-works.svg").write_text("\n".join(o))


def _rel_bins(tag, task="frustration", key="reliability_raw"):
    p = HERE.parent / "results" / f"cal_{task}_{tag}@eval.json"
    if not p.exists():
        return None
    qs = json.load(open(p))["questions"]
    q = qs.get(task) or next(iter(qs.values()))
    return q[key], q


def fig_results():
    """Hero: three measured panels. Latency per decision; raw calibration error on never-trained
    tasks before and after training; the reliability curve behind one of those numbers."""
    o = _svg(900, 330)
    # panel A: latency
    ax, ay = 40, 60
    _caps(o, ax, 34, "A · Latency per decision")
    models = (("0.6B", 44, 1108), ("1.7B", 51, 912), ("4B", 111, 1268))
    scale = 190 / 1300
    for i, (m, brier, gen) in enumerate(models):
        y = ay + i * 52
        _t(o, ax, y + 10, m, 11, INK, "700")
        _bar(o, ax + 40, y, brier * scale, 12, ACCENT)
        _t(o, ax + 40 + brier * scale + 6, y + 10, f"{brier} ms", 10.5, INK)
        _bar(o, ax + 40, y + 16, gen * scale, 12, BASE)
        _t(o, ax + 40 + gen * scale + 6, y + 26, f"{gen} ms", 10.5, MUTED)
    _hair(o, ax + 40, ay + 160, ax + 40 + 190, ay + 160)
    o.append(f'<rect x="{ax}" y="{ay + 176}" width="10" height="10" rx="2" fill="{ACCENT}"/>')
    _t(o, ax + 16, ay + 185, "Brier, one pass", 10.5, MUTED)
    o.append(f'<rect x="{ax + 120}" y="{ay + 176}" width="10" height="10" rx="2" fill="{BASE}"/>')
    _t(o, ax + 136, ay + 185, "same weights generating JSON", 10.5, MUTED)

    # panel B: raw ECE on never-trained tasks, untrained -> trained (dumbbells)
    bx = 340
    _caps(o, bx, 34, "B · Calibration error, unseen tasks")
    pairs = []
    for task, base_tag, tr_tag, label in (("frustration", "Qwen3-4B", "Qwen3-4B-brier-r16-s1200-u0.3-train", "frustration · 4B"),
                                           ("duplicates", "Qwen3-4B", "Qwen3-4B-brier-r16-s1200-u0.3-train", "duplicates · 4B"),
                                           ("frustration", "Phi-4-mini-instruct", "Phi-4-mini-r3", "frustration · Phi-4-mini"),
                                           ("banking", "Qwen3-0.6B", "Qwen3-0.6B-brier-r16-s1500-u0.3-train", "banking (real) · 0.6B")):
        b, t = _rel_bins(base_tag, task), _rel_bins(tr_tag, task)
        if b and t:
            pairs.append((label, b[1]["ece_raw"], t[1]["ece_raw"]))
    sx = 200 / 0.6
    for i, (label, e0, e1) in enumerate(pairs):
        y = ay + 14 + i * 44
        _t(o, bx, y - 14, label, 10.5, MUTED)
        _hair(o, bx, y, bx + 200, y)
        o.append(f'<line x1="{bx + e1 * sx:.1f}" y1="{y}" x2="{bx + e0 * sx:.1f}" y2="{y}" stroke="{INK}" stroke-width="1.5"/>')
        o.append(f'<circle cx="{bx + e0 * sx:.1f}" cy="{y}" r="5" fill="{BASE}" stroke="{SURFACE}" stroke-width="2"/>')
        o.append(f'<circle cx="{bx + e1 * sx:.1f}" cy="{y}" r="5" fill="{ACCENT}" stroke="{SURFACE}" stroke-width="2"/>')
        _t(o, bx + max(e0, e1) * sx + 12, y + 4, f"{e0:.2f} → {e1:.2f}", 10.5, INK)
    for v in (0, 0.3, 0.6):
        _t(o, bx + v * sx, ay + 14 + len(pairs) * 44 - 10, f"{v:.1f}", 9.5, FAINT, "400", "middle")
    _t(o, bx + 100, ay + 14 + len(pairs) * 44 + 6, "expected calibration error, lower is better", 10, FAINT, "400", "middle")

    # panel C: reliability, frustration 4B untrained vs trained
    cx, cy, S = 660, 78, 150
    _caps(o, cx, 34, "C · Reliability, frustration, 4B")
    _hair(o, cx, cy + S, cx + S, cy + S); _hair(o, cx, cy, cx, cy + S)
    o.append(f'<line x1="{cx}" y1="{cy + S}" x2="{cx + S}" y2="{cy}" stroke="{FAINT}" stroke-width="1" stroke-dasharray="3 3"/>')
    for tag, color in (("Qwen3-4B", BASE), ("Qwen3-4B-brier-r16-s1200-u0.3-train", ACCENT)):
        r = _rel_bins(tag)
        if not r:
            continue
        total = sum(n for lo, hi, n, conf, acc in r[0])
        for lo, hi, n, conf, acc in r[0]:
            if n < 8:
                continue
            rad = 3 + 9 * (n / total) ** 0.5
            o.append(f'<circle cx="{cx + S * conf:.1f}" cy="{cy + S - S * acc:.1f}" r="{rad:.1f}" fill="{color}" fill-opacity="0.85" stroke="{SURFACE}" stroke-width="2"/>')
    for v in (0, 0.5, 1):
        _t(o, cx + S * v, cy + S + 14, f"{v:.1f}", 9.5, FAINT, "400", "middle")
        _t(o, cx - 6, cy + S - S * v + 3, f"{v:.1f}", 9.5, FAINT, "400", "end")
    _t(o, cx + S / 2, cy + S + 30, "stated confidence", 10, FAINT, "400", "middle")
    o.append(f'<text transform="translate({cx - 26} {cy + S / 2}) rotate(-90)" font-size="10" fill="{FAINT}" text-anchor="middle">observed accuracy</text>')
    o.append(f'<circle cx="{cx + 6}" cy="{cy + S + 52}" r="4" fill="{BASE}"/>')
    _t(o, cx + 16, cy + S + 56, "untrained", 10.5, MUTED)
    o.append(f'<circle cx="{cx + 86}" cy="{cy + S + 52}" r="4" fill="{ACCENT}"/>')
    _t(o, cx + 96, cy + S + 56, "trained", 10.5, MUTED)
    _t(o, 40, 318, "Panel A: tickets task, same weights, one pass versus generating JSON. B and C: raw probabilities, no post-hoc correction; dot area ∝ decisions in the bin. 300 held-out examples per task; results/.", 9.5, FAINT)
    o.append("</svg>")
    (HERE / "fig-results.svg").write_text("\n".join(o))





def _svg(w, h):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
         f'<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{MUTED}"/></marker></defs>',
         f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>']
    return o


def _caps(o, x, y, text, fill=None, anchor="start"):
    """Small-caps section label, the only decorative element in the system."""
    o.append(f'<text x="{x}" y="{y}" font-size="10" font-weight="700" letter-spacing="1.4" fill="{fill or MUTED}" text-anchor="{anchor}">{text.upper()}</text>')


def _hair(o, x1, y1, x2, y2):
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1"/>')


def _bar(o, x, y, w, h, color, horizontal=True):
    """A thin bar with a 4px rounded data end and a square baseline end."""
    if w <= 0 and horizontal or h <= 0 and not horizontal:
        return
    if horizontal:
        r = min(4, w / 2)
        o.append(f'<path d="M{x} {y} H{x + w - r} a{r} {r} 0 0 1 {r} {r} V{y + h - r} a{r} {r} 0 0 1 -{r} {r} H{x} Z" fill="{color}"/>')
    else:
        r = min(4, h / 2)
        o.append(f'<path d="M{x} {y + h} V{y + r} a{r} {r} 0 0 1 {r} -{r} H{x + w - r} a{r} {r} 0 0 1 {r} {r} V{y + h} Z" fill="{color}"/>')


def _t(o, x, y, s, size=12, fill=INK, weight="400", anchor="start"):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def _write(o, name):
    o.append("</svg>")
    (HERE / f"fig-{name}.svg").write_text("\n".join(o))


def fig_next_token():
    o = _svg(900, 300)
    _t(o, 450, 34, "A language model computes one thing: a probability for every possible next token", 15, INK, "700", "middle")
    toks = ["The", " ticket", " should", " go", " to", " the"]
    x = 60
    for t in toks:
        w = 14 + len(t) * 8
        o.append(f'<rect x="{x}" y="70" width="{w}" height="30" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
        _t(o, x + w / 2, 90, t.strip(), 13, INK, "400", "middle")
        x += w + 6
    o.append(f'<rect x="{x}" y="70" width="60" height="30" rx="3" fill="{PALE}" stroke="{BLUE}" stroke-dasharray="4 3"/>')
    _t(o, x + 30, 90, "?", 15, BLUE, "700", "middle")
    _t(o, 60, 130, "context (tokens so far)", 11, MUTED)
    o.append(f'<line x1="{x + 30}" y1="104" x2="{x + 30}" y2="150" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    rows = [(" billing", 0.42), (" bug", 0.31), (" account", 0.14), (" other", 0.06), (" feature", 0.04), ("…150,000 more", 0.03)]
    for i, (t, p) in enumerate(rows):
        y = 168 + i * 20
        _t(o, 340, y + 11, t.strip(), 12, INK if i < 5 else FAINT, "400", "end")
        o.append(f'<rect x="350" y="{y}" width="300" height="14" rx="3" fill="#EEF0F3"/>')
        o.append(f'<rect x="350" y="{y}" width="{300 * p:.0f}" height="14" rx="3" fill="{BLUE if i < 5 else FAINT}"/>')
        _t(o, 660, y + 11, f"{p:.2f}", 12, MUTED)
    _t(o, 720, 190, "This vector is the whole", 12, MUTED)
    _t(o, 720, 206, "output of one forward pass.", 12, MUTED)
    _t(o, 720, 230, "Generation = pick one, append,", 12, MUTED)
    _t(o, 720, 246, "run the pass again. Brier never", 12, MUTED)
    _t(o, 720, 262, "does the second step.", 12, MUTED)
    _write(o, "next-token")


def fig_scoring():
    o = _svg(900, 320)
    _t(o, 450, 34, "Scoring a candidate: the log-probability the model gives to its tokens, in context", 15, INK, "700", "middle")
    _t(o, 40, 76, "prompt (input + schema + instructions)", 11, MUTED)
    o.append(f'<rect x="40" y="84" width="330" height="30" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
    _t(o, 50, 104, "…Fields: category: one of billing, bug, … ", 12, INK)
    _t(o, 380, 76, "field header", 11, MUTED)
    o.append(f'<rect x="380" y="84" width="130" height="30" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
    _t(o, 445, 104, '{"category": "', 12, INK, "400", "middle")
    _t(o, 520, 76, "candidate tokens", 11, MUTED)
    for i, (t, lp) in enumerate((("account", "-0.62"),)):
        o.append(f'<rect x="520" y="84" width="90" height="30" rx="3" fill="{PALE}" stroke="{BLUE}"/>')
        _t(o, 565, 104, t, 12, BLUE, "700", "middle")
    _t(o, 620, 104, "log p = −0.62", 12, INK, "700")
    _t(o, 40, 150, "Each field's candidates share the same prompt and header; only the tail differs:", 12, MUTED)
    cands = (("billing", -3.10), ("bug", -1.05), ("feature", -6.40), ("account", -0.62), ("other", -4.20))
    for i, (c, lp) in enumerate(cands):
        y = 172 + i * 26
        o.append(f'<rect x="40" y="{y}" width="470" height="18" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
        _t(o, 48, y + 13, '…{"category": "', 11, FAINT)
        o.append(f'<rect x="520" y="{y}" width="90" height="18" rx="3" fill="{PALE}" stroke="{BLUE}"/>')
        _t(o, 565, y + 13, c, 11, BLUE, "700", "middle")
        _t(o, 630, y + 13, f"{lp:+.2f}", 12, INK, "700" if c == "account" else "400")
        if c == "account":
            _t(o, 690, y + 13, "← highest score wins", 11, MUTED)
    _t(o, 40, 306, "A multi-token candidate is scored by summing its tokens' log-probabilities. Nothing is ever generated.", 11, FAINT)
    _write(o, "scoring")


def fig_batch():
    o = _svg(900, 330)
    _t(o, 450, 34, "One batched forward pass: every candidate of every field is a row", 15, INK, "700", "middle")
    rows = [("category", "billing", 5), ("category", "bug", 5), ("category", "feature", 5), ("category", "account", 5), ("category", "other", 5),
            ("priority", "low", 4), ("priority", "medium", 4), ("priority", "high", 4), ("priority", "urgent", 4),
            ("needs_human", "yes", 2), ("needs_human", "no", 2)]
    colors = {"category": "#3557C4", "priority": "#5B8DEF", "needs_human": "#9DB9F5"}
    y = 60
    for i, (f, c, n) in enumerate(rows):
        col = colors[f]
        o.append(f'<rect x="40" y="{y}" width="330" height="16" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
        _t(o, 48, y + 12, "prompt (identical in every row)", 10, FAINT)
        o.append(f'<rect x="372" y="{y}" width="110" height="16" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
        _t(o, 427, y + 12, f'{{"{f}": "', 10, MUTED, "400", "middle")
        cw = 20 + len(c) * 7
        o.append(f'<rect x="484" y="{y}" width="{cw}" height="16" rx="3" fill="{col}"/>')
        _t(o, 484 + cw / 2, y + 12, c, 10, "#FFFFFF", "700", "middle")
        px = 484 + cw + 2
        if px < 600:
            o.append(f'<rect x="{px}" y="{y}" width="{600 - px}" height="16" rx="3" fill="#F1F3F6"/>')
            _t(o, (px + 600) / 2, y + 12, "pad", 9, FAINT, "400", "middle")
        y += 21
    _t(o, 620, 72, "Only the candidate positions are read", 12, INK, "700")
    _t(o, 620, 90, "The model produces a next-token", 11, MUTED)
    _t(o, 620, 105, "distribution at every position, but", 11, MUTED)
    _t(o, 620, 120, "Brier looks up just the candidate", 11, MUTED)
    _t(o, 620, 135, "tokens' log-probabilities. That keeps", 11, MUTED)
    _t(o, 620, 150, "memory small (the vocabulary is 150k).", 11, MUTED)
    _t(o, 620, 182, "Rows are padded to equal length with", 11, MUTED)
    _t(o, 620, 197, "an attention mask, so padding is", 11, MUTED)
    _t(o, 620, 212, "invisible to the model.", 11, MUTED)
    _t(o, 620, 244, "11 rows, one call, ~100 ms on a 4090", 12, BLUE, "700")
    _t(o, 620, 262, "for a 4B model. The cost grows with", 11, MUTED)
    _t(o, 620, 277, "the number of allowed values, not", 11, MUTED)
    _t(o, 620, 292, "with output length.", 11, MUTED)
    _write(o, "batch")


def fig_softmax():
    o = _svg(900, 260)
    _t(o, 450, 34, "Softmax per field turns scores into a distribution that sums to one", 15, INK, "700", "middle")
    cands = (("billing", -3.10, 0.07), ("bug", -1.05, 0.34), ("feature", -6.40, 0.00), ("account", -0.62, 0.52), ("other", -4.20, 0.02))
    _t(o, 60, 70, "score (log p)", 11, MUTED)
    _t(o, 330, 70, "exp(score − max), normalised", 11, MUTED)
    _t(o, 620, 70, "probability", 11, MUTED)
    for i, (c, lp, p) in enumerate(cands):
        y = 90 + i * 26
        _t(o, 60, y + 12, c, 12, INK, "700" if c == "account" else "400")
        _t(o, 180, y + 12, f"{lp:+.2f}", 12, INK, "400", "end")
        o.append(f'<line x1="200" y1="{y + 8}" x2="240" y2="{y + 8}" stroke="#5B6B7C" stroke-width="1.5" marker-end="url(#ar)"/>')
        o.append(f'<rect x="330" y="{y}" width="240" height="16" rx="3" fill="#EEF0F3"/>')
        o.append(f'<rect x="330" y="{y}" width="{240 * p:.0f}" height="16" rx="3" fill="{BLUE}"/>')
        _t(o, 620, y + 12, f"{p:.2f}", 12, INK, "700" if c == "account" else "400")
    _t(o, 700, 102, "The top value is decoded to", 11, MUTED)
    _t(o, 700, 117, "its Python type (str or bool).", 11, MUTED)
    _t(o, 700, 141, "The whole distribution is kept,", 11, MUTED)
    _t(o, 700, 156, "so code can act on 0.52 rather", 11, MUTED)
    _t(o, 700, 171, "than on 'account' alone.", 11, MUTED)
    _t(o, 60, 240, "It is impossible for the answer to be anything other than one of the five names on the left. That is the type guarantee.", 11, FAINT)
    _write(o, "softmax")


def fig_reliability():
    o = _svg(900, 360)
    _t(o, 450, 34, "Calibration: when the model says 0.9, is it right 90% of the time?", 15, INK, "700", "middle")
    ox, oy, s = 80, 300, 240
    o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + s}" y2="{oy}" stroke="{MUTED}"/>')
    o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - s}" stroke="{MUTED}"/>')
    o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + s}" y2="{oy - s}" stroke="{FAINT}" stroke-dasharray="4 4"/>')
    _t(o, ox + s / 2, oy + 30, "confidence the model reported", 11, MUTED, "400", "middle")
    o.append(f'<text transform="translate({ox - 30} {oy - s / 2}) rotate(-90)" font-size="11" fill="{MUTED}" text-anchor="middle">fraction actually correct</text>')
    for v in (0, 0.5, 1):
        _t(o, ox + s * v, oy + 14, f"{v:.1f}", 10, FAINT, "400", "middle")
        _t(o, ox - 8, oy - s * v + 4, f"{v:.1f}", 10, FAINT, "400", "end")
    pts = [(0.55, 0.30), (0.65, 0.36), (0.75, 0.42), (0.85, 0.50), (0.95, 0.59), (0.99, 0.63)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{ox + s * c:.0f} {oy - s * a:.0f}" for i, (c, a) in enumerate(pts))
    o.append(f'<path d="{path}" fill="none" stroke="{VIOLET}" stroke-width="2.5"/>')
    for c, a in pts:
        o.append(f'<circle cx="{ox + s * c:.0f}" cy="{oy - s * a:.0f}" r="4" fill="{VIOLET}"/>')
    pts2 = [(0.35, 0.33), (0.45, 0.44), (0.55, 0.53), (0.65, 0.66), (0.75, 0.73), (0.85, 0.84)]
    path2 = " ".join(f"{'M' if i == 0 else 'L'}{ox + s * c:.0f} {oy - s * a:.0f}" for i, (c, a) in enumerate(pts2))
    o.append(f'<path d="{path2}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for c, a in pts2:
        o.append(f'<circle cx="{ox + s * c:.0f}" cy="{oy - s * a:.0f}" r="4" fill="{BLUE}"/>')
    _t(o, 380, 90, "dashed diagonal: perfect calibration", 12, FAINT)
    _t(o, 380, 120, "raw log-likelihood confidence", 12, VIOLET, "700")
    _t(o, 380, 136, "Qwen3-4B on tickets reports a mean confidence of 0.95", 11, MUTED)
    _t(o, 380, 151, "while being right 59% of the time. Points sit far below", 11, MUTED)
    _t(o, 380, 166, "the diagonal: overconfident. ECE = 0.37.", 11, MUTED)
    _t(o, 380, 196, "after one temperature per field", 12, BLUE, "700")
    _t(o, 380, 212, "Scores are divided by a constant fitted on held-out data.", 11, MUTED)
    _t(o, 380, 227, "Rankings do not change, so accuracy is untouched, but the", 11, MUTED)
    _t(o, 380, 242, "confidences move toward the diagonal. ECE = 0.16.", 11, MUTED)
    _t(o, 380, 272, "ECE (expected calibration error) is the average gap", 11, FAINT)
    _t(o, 380, 287, "between the two axes, weighted by how many decisions", 11, FAINT)
    _t(o, 380, 302, "land in each confidence bin. Zero is perfect.", 11, FAINT)
    _write(o, "reliability")


def fig_temperature():
    o = _svg(900, 250)
    _t(o, 450, 34, "Temperature scaling: one number per field that flattens or sharpens every distribution", 15, INK, "700", "middle")
    sets = (("T = 1 (raw)", (0.86, 0.09, 0.03, 0.02), "over-sure"), ("T = 3.75 (fitted)", (0.52, 0.27, 0.13, 0.08), "honest"), ("T = 10", (0.31, 0.26, 0.22, 0.21), "too flat"))
    for j, (title, ps, note) in enumerate(sets):
        x0 = 60 + j * 280
        _t(o, x0, 76, title, 12, INK, "700")
        _t(o, x0 + 200, 76, note, 11, VIOLET if j != 1 else BLUE, "700", "end")
        for i, p in enumerate(ps):
            y = 90 + i * 24
            _t(o, x0, y + 12, ("bug", "account", "billing", "other")[i], 11, MUTED)
            o.append(f'<rect x="{x0 + 60}" y="{y}" width="140" height="14" rx="3" fill="#EEF0F3"/>')
            o.append(f'<rect x="{x0 + 60}" y="{y}" width="{140 * p:.0f}" height="14" rx="3" fill="{BLUE}"/>')
            _t(o, x0 + 208, y + 11, f"{p:.2f}", 11, INK)
    _t(o, 60, 214, "probability = softmax(score / T). T is chosen to minimise negative log-likelihood on half the data and evaluated on the other half.", 11, FAINT)
    _t(o, 60, 230, "The argmax never changes, so accuracy is identical at every T. Only honesty changes.", 11, FAINT)
    _write(o, "temperature")


def fig_training():
    o = _svg(900, 420)
    _t(o, 450, 34, "How the backbone was trained before Brier, and how Brier trains on top of it", 15, INK, "700", "middle")
    stages = (("pretraining", "predict the next token", "trillions of tokens of text", "→ general knowledge, reading"),
              ("instruction tuning", "imitate helpful answers", "curated prompt/response pairs", "→ follows instructions, JSON"),
              ("preference tuning", "reward preferred answers", "human or model preferences", "→ confident, agreeable tone"))
    for i, (name, obj, data, out) in enumerate(stages):
        x = 40 + i * 275
        o.append(f'<rect x="{x}" y="60" width="255" height="96" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
        _t(o, x + 14, 84, name, 13, INK, "700")
        _t(o, x + 14, 104, f"objective: {obj}", 11, MUTED)
        _t(o, x + 14, 120, f"data: {data}", 11, MUTED)
        _t(o, x + 14, 140, out, 11, BLUE)
        if i < 2:
            o.append(f'<line x1="{x + 257}" y1="108" x2="{x + 273}" y2="108" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    _t(o, 40, 182, "published open weights (Qwen3, Phi-4-mini, SmolLM2). The untrained scorer uses them exactly as released: good at ranking answers, bad at stating probabilities.", 11, FAINT)

    o.append(f'<line x1="40" y1="200" x2="860" y2="200" stroke="{LINE}"/>')
    o.append(f'<rect x="40" y="220" width="200" height="170" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
    _t(o, 54, 244, "data with ground truth", 13, INK, "700")
    for i, s in enumerate(("synthetic tasks, held-out templates", "(the generator chose the label)", "30% ambiguous, soft labels", "= the generator's own posterior", "contrastive claims", "(the cue is present, the answer is no)")):
        _t(o, 54, 266 + i * 16, s, 11, MUTED if i % 2 == 0 else FAINT)
    o.append(f'<line x1="242" y1="305" x2="268" y2="305" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<rect x="270" y="220" width="220" height="170" rx="3" fill="{PALE}" stroke="{BLUE}" stroke-width="1.5"/>')
    _t(o, 284, 244, "backbone + LoRA + typed head", 13, INK, "700")
    _t(o, 284, 266, "base weights frozen", 11, MUTED)
    _t(o, 284, 282, "LoRA rank 16 on q, k, v, o,", 11, MUTED)
    _t(o, 284, 298, "gate, up, down in every block", 11, MUTED)
    _t(o, 284, 322, "typed head over hidden states:", 11, BLUE)
    _t(o, 284, 338, "input-conditional temperature", 11, BLUE)
    _t(o, 284, 354, "+ per-candidate residual", 11, BLUE)
    _t(o, 284, 376, "zero-initialised = plain scorer", 11, MUTED)
    o.append(f'<line x1="492" y1="305" x2="518" y2="305" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<rect x="520" y="220" width="340" height="170" rx="3" fill="#FFFFFF" stroke="{LINE}"/>')
    _t(o, 534, 244, "objective: calibrated decisions", 13, INK, "700")
    _t(o, 534, 266, "1. cross-entropy over the answer set, hard or soft", 11, MUTED)
    _t(o, 534, 281, "   targets: a proper scoring rule, honest at its minimum", 11, MUTED)
    _t(o, 534, 301, "2. outcome reward, no labels: policy gradient on the", 11, MUTED)
    _t(o, 534, 316, "   environment's result + Brier score of the stated p", 11, MUTED)
    _t(o, 534, 336, "3. gate: raw ECE on never-trained tasks must beat", 11, MUTED)
    _t(o, 534, 351, "   the untrained model after temperature scaling", 11, MUTED)
    _t(o, 534, 375, "no preference model, no pass/fail reward", 11, BLUE)
    _write(o, "training")


def fig_components():
    o = _svg(900, 300)
    _t(o, 450, 34, "An ask() call through the code", 15, INK, "700", "middle")
    boxes = (("your code", "state + questions", "#FFFFFF", LINE), ("brier/questions.py", "Choice, Score, Claim → candidates", "#FFFFFF", LINE),
             ("brier/scorer.py", "prompt → batch → softmax", PALE, BLUE), ("brier/questions.py", "Answer: value, probabilities", "#FFFFFF", LINE))
    for i, (name, what, fill, stroke) in enumerate(boxes):
        x = 40 + i * 215
        o.append(f'<rect x="{x}" y="70" width="195" height="70" rx="3" fill="{fill}" stroke="{stroke}"/>')
        _t(o, x + 12, 96, name, 12, INK, "700")
        _t(o, x + 12, 118, what, 11, MUTED)
        if i < 3:
            o.append(f'<line x1="{x + 197}" y1="105" x2="{x + 213}" y2="105" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<line x1="40" y1="170" x2="860" y2="170" stroke="{LINE}"/>')
    _t(o, 40, 196, "measurement, not on the request path", 12, MUTED, "700")
    items = (("tasks/", "eight tasks with computable or human labels"), ("brier/calibrate.py", "temperature, held-out ECE, decision policy"),
             ("train.py · rl.py", "supervised and outcome-reward training"), ("pod.py", "GPU pod with a hard time-to-live"))
    for i, (n, d) in enumerate(items):
        x = 40 + (i % 2) * 420
        y = 222 + (i // 2) * 34
        _t(o, x, y, n, 12, INK, "700")
        _t(o, x + 130, y, d, 11, MUTED)
    _write(o, "components")


for fn in (fig_next_token, fig_scoring, fig_batch, fig_softmax, fig_reliability, fig_temperature, fig_training, fig_components):
    fn()
print("figures written")


# ---- primer figures ---------------------------------------------------------

import math as _m


def fig_distribution():
    o = _svg(900, 300)
    _t(o, 450, 34, "A probability distribution: one number per outcome, all non-negative, summing to one", 15, INK, "700", "middle")
    outcomes = (("billing", 0.07), ("bug", 0.34), ("feature", 0.00), ("account", 0.52), ("other", 0.07))
    for i, (c, p) in enumerate(outcomes):
        x = 70 + i * 150
        h = 160 * p
        o.append(f'<rect x="{x}" y="{230 - h:.0f}" width="90" height="{h:.0f}" rx="3" fill="{BLUE}"/>')
        _t(o, x + 45, 250, c, 12, INK, "400", "middle")
        _t(o, x + 45, 222 - h, f"{p:.2f}", 12, INK, "700", "middle")
    o.append(f'<line x1="60" y1="230" x2="820" y2="230" stroke="{MUTED}"/>')
    _t(o, 830, 120, "Σ = 1.00", 13, INK, "700")
    _t(o, 450, 282, "Reading it: 'account' is the most likely answer, but 'bug' has a third of the mass. The argmax hides that; the distribution shows it.", 11, FAINT, "400", "middle")
    _write(o, "distribution")


def fig_entropy():
    o = _svg(900, 300)
    _t(o, 450, 34, "Entropy: how spread out a distribution is. Confidence in Brier is the top probability; entropy is the fuller picture.", 14, INK, "700", "middle")
    sets = (("certain", (0.97, 0.01, 0.01, 0.01)), ("leaning", (0.52, 0.34, 0.07, 0.07)), ("uniform", (0.25, 0.25, 0.25, 0.25)))
    for j, (name, ps) in enumerate(sets):
        x0 = 70 + j * 280
        H = -sum(p * _m.log(p) for p in ps if p > 0)
        _t(o, x0 + 90, 76, name, 13, INK, "700", "middle")
        for i, p in enumerate(ps):
            h = 120 * p
            o.append(f'<rect x="{x0 + i * 46}" y="{210 - h:.0f}" width="36" height="{h:.0f}" rx="3" fill="{BLUE}"/>')
        o.append(f'<line x1="{x0 - 6}" y1="210" x2="{x0 + 186}" y2="210" stroke="{MUTED}"/>')
        _t(o, x0 + 90, 234, f"top p = {max(ps):.2f}   H = {H:.2f} nats", 11, MUTED, "400", "middle")
    _t(o, 450, 274, "H = −Σ p·log p. Zero when one outcome has all the mass; log(4) = 1.39 when four outcomes are equally likely.", 11, FAINT, "400", "middle")
    _write(o, "entropy")


def fig_softmax_curve():
    o = _svg(900, 320)
    _t(o, 450, 34, "Softmax maps any real-valued scores to a distribution; temperature T controls how sharp", 15, INK, "700", "middle")
    scores = (-0.62, -1.05, -3.10, -4.20)
    labels = ("account", "bug", "billing", "other")
    Ts = (0.5, 1, 2, 4, 8)
    ox, oy, w, h = 150, 250, 460, 170
    o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + w}" y2="{oy}" stroke="{MUTED}"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - h}" stroke="{MUTED}"/>')
    _t(o, ox + w / 2, oy + 28, "temperature T", 11, MUTED, "400", "middle")
    o.append(f'<text transform="translate({ox - 34} {oy - h / 2}) rotate(-90)" font-size="11" fill="{MUTED}" text-anchor="middle">probability</text>')
    cols = (BLUE, "#5B8DEF", "#9DB9F5", VIOLET)
    for li, (lab, col) in enumerate(zip(labels, cols)):
        pts = []
        for ti, T in enumerate(Ts):
            z = [_m.exp((s - max(scores)) / T) for s in scores]
            p = z[li] / sum(z)
            pts.append((ox + 40 + ti * (w - 80) / (len(Ts) - 1), oy - h * p))
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
        _t(o, pts[0][0] - 10, pts[0][1] + 4, lab, 11, col, "700", "end")
    for ti, T in enumerate(Ts):
        _t(o, ox + 40 + ti * (w - 80) / (len(Ts) - 1), oy + 14, str(T), 10, FAINT, "400", "middle")
    for v in (0, 0.5, 1):
        _t(o, ox - 8, oy - h * v + 3, f"{v:.1f}", 10, FAINT, "400", "end")
    _t(o, 700, 110, "p_i = exp(s_i / T) / Σ_j exp(s_j / T)", 12, INK, "700")
    _t(o, 700, 134, "T → 0: all mass on the top score", 11, MUTED)
    _t(o, 700, 150, "T = 1: the raw model distribution", 11, MUTED)
    _t(o, 700, 166, "T → ∞: uniform", 11, MUTED)
    _t(o, 700, 196, "The order of the curves never", 11, MUTED)
    _t(o, 700, 211, "crosses: the argmax is the same", 11, MUTED)
    _t(o, 700, 226, "at every T.", 11, MUTED)
    _write(o, "softmax-curve")


def fig_scoring_rules():
    o = _svg(900, 320)
    _t(o, 450, 34, "Proper scoring rules: the loss you pay for stating probability p when the outcome happens", 15, INK, "700", "middle")
    ox, oy, w, h = 90, 250, 520, 170
    o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + w}" y2="{oy}" stroke="{MUTED}"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - h}" stroke="{MUTED}"/>')
    _t(o, ox + w / 2, oy + 28, "stated probability p of the outcome that occurred", 11, MUTED, "400", "middle")
    o.append(f'<text transform="translate({ox - 34} {oy - h / 2}) rotate(-90)" font-size="11" fill="{MUTED}" text-anchor="middle">loss</text>')
    for v in (0, 0.5, 1):
        _t(o, ox + w * v, oy + 14, f"{v:.1f}", 10, FAINT, "400", "middle")
    n = 60
    brier_pts = [(ox + w * i / n, oy - h * ((1 - i / n) ** 2)) for i in range(n + 1)]
    log_pts = [(ox + w * i / n, oy - h * min(1.0, -_m.log(max(i / n, 1e-3)) / 3)) for i in range(2, n + 1)]
    o.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(brier_pts)) + f'" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
    o.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(log_pts)) + f'" fill="none" stroke="{VIOLET}" stroke-width="2.2"/>')
    _t(o, 700, 100, "Brier: (1 − p)²", 12, BLUE, "700")
    _t(o, 700, 118, "bounded, gentle near p = 0", 11, MUTED)
    _t(o, 700, 148, "log loss: −log p", 12, VIOLET, "700")
    _t(o, 700, 166, "unbounded: p = 0 on the truth", 11, MUTED)
    _t(o, 700, 181, "costs infinity, so it punishes", 11, MUTED)
    _t(o, 700, 196, "confident misses hardest", 11, MUTED)
    _t(o, 700, 226, "Both are minimised in expectation", 11, INK)
    _t(o, 700, 241, "only by stating your true belief.", 11, INK)
    _t(o, 700, 256, "That is what 'proper' means.", 11, INK)
    _write(o, "scoring-rules")


def fig_transformer():
    o = _svg(900, 360)
    _t(o, 450, 34, "Inside the backbone: what one forward pass does to a sequence of tokens", 15, INK, "700", "middle")
    stages = (("tokens", "ids from the tokenizer", "#FFFFFF"), ("embeddings", "each id → a vector", "#FFFFFF"),
              ("N × blocks", "attention, then MLP", PALE), ("hidden states", "one vector per position", "#FFFFFF"),
              ("LM head", "vector → 150k logits", "#FFFFFF"), ("softmax", "next-token distribution", "#FFFFFF"))
    for i, (name, what, fill) in enumerate(stages):
        x = 30 + i * 143
        o.append(f'<rect x="{x}" y="70" width="128" height="72" rx="3" fill="{fill}" stroke="{BLUE if fill == PALE else LINE}"/>')
        _t(o, x + 64, 98, name, 11.5, INK, "700", "middle")
        _t(o, x + 64, 118, what, 9.5, MUTED, "400", "middle")
        if i < 5:
            o.append(f'<line x1="{x + 129}" y1="106" x2="{x + 142}" y2="106" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    _t(o, 30, 182, "Inside one block", 12, INK, "700")
    _t(o, 30, 202, "attention:  for each position, a weighted average of every earlier position's vector, weights = softmax(query · keyᵀ / √d).", 11, MUTED)
    _t(o, 30, 218, "                   This is how 'account' at the end can look back at 'can't log in' at the start.", 11, MUTED)
    _t(o, 30, 240, "MLP:  the same small two-layer network applied to every position independently.", 11, MUTED)
    _t(o, 30, 262, "Each block adds its result to its input (a residual connection), so N blocks refine the same vectors N times.", 11, MUTED)
    _t(o, 30, 292, "Where Brier reads:", 12, BLUE, "700")
    _t(o, 30, 312, "today, the softmax output at the candidate positions (milestone 1 and 2). LoRA (milestone 2) adds small trainable matrices inside", 11, MUTED)
    _t(o, 30, 328, "the attention and MLP weights, leaving the original weights frozen. A future typed head would read the hidden states directly.", 11, MUTED)
    _write(o, "transformer")


def fig_lora():
    o = _svg(900, 260)
    _t(o, 450, 34, "LoRA: fine-tune a big matrix by adding a small low-rank one next to it", 15, INK, "700", "middle")
    o.append(f'<rect x="80" y="70" width="140" height="140" rx="3" fill="#F1F3F6" stroke="{LINE}"/>')
    _t(o, 150, 145, "W", 22, MUTED, "700", "middle")
    _t(o, 150, 230, "frozen, d × d (e.g. 2560 × 2560)", 10, MUTED, "400", "middle")
    _t(o, 260, 145, "+", 26, INK, "700", "middle")
    o.append(f'<rect x="310" y="70" width="140" height="22" rx="3" fill="{BLUE}"/>')
    _t(o, 380, 86, "A  (r × d)", 11, "#FFFFFF", "700", "middle")
    o.append(f'<rect x="310" y="100" width="22" height="110" rx="3" fill="{BLUE}"/>')
    o.append(f'<text transform="translate(325 155) rotate(-90)" font-size="11" fill="#FFFFFF" font-weight="700" text-anchor="middle">B (d × r)</text>')
    _t(o, 380, 160, "trained", 12, BLUE, "700", "middle")
    _t(o, 380, 178, "r = 16", 11, MUTED, "400", "middle")
    _t(o, 520, 110, "W' = W + B·A", 14, INK, "700")
    _t(o, 520, 134, "B·A has the same shape as W but only 2·d·r numbers", 11, MUTED)
    _t(o, 520, 150, "to learn instead of d²: for d = 2560, r = 16 that is", 11, MUTED)
    _t(o, 520, 166, "82k parameters per matrix instead of 6.5M (1.3%).", 11, MUTED)
    _t(o, 520, 194, "The base model is untouched and can be restored by", 11, MUTED)
    _t(o, 520, 210, "deleting the adapter. Fits a 4B model on one 24GB GPU.", 11, MUTED)
    _write(o, "lora")


def fig_gradient():
    o = _svg(900, 280)
    _t(o, 450, 34, "Training is one loop: score, compare to the label, nudge the weights downhill, repeat", 15, INK, "700", "middle")
    steps = (("example", "state, questions, labels"), ("forward pass", "scores per candidate"), ("softmax per question", "probabilities"),
             ("loss", "−log p(correct answer)"), ("backward pass", "gradient w.r.t. LoRA weights"), ("optimizer step", "weights −= lr · gradient"))
    for i, (name, what) in enumerate(steps):
        x = 30 + i * 143
        o.append(f'<rect x="{x}" y="70" width="128" height="72" rx="3" fill="{PALE if i == 3 else "#FFFFFF"}" stroke="{BLUE if i == 3 else LINE}"/>')
        _t(o, x + 64, 98, name, 11.5, INK, "700", "middle")
        _t(o, x + 64, 118, what, 9.5, MUTED, "400", "middle")
        if i < 5:
            o.append(f'<line x1="{x + 129}" y1="106" x2="{x + 142}" y2="106" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<path d="M 830 150 C 830 190, 100 190, 94 150" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#ar)"/>')
    _t(o, 450, 198, "1500 steps × 4 examples each, about fifteen minutes for a 0.6B model on a 4090", 11, FAINT, "400", "middle")
    _t(o, 30, 236, "The loss is the log scoring rule from Part III of the architecture doc. Minimising it makes the correct answer likely AND makes the", 11, MUTED)
    _t(o, 30, 252, "stated probabilities honest, because a proper scoring rule cannot be minimised by overclaiming.", 11, MUTED)
    _write(o, "gradient")


for fn in (fig_distribution, fig_entropy, fig_softmax_curve, fig_scoring_rules, fig_transformer, fig_lora, fig_gradient):
    fn()
print("primer figures written")

figure()
fig_results()
print("assets written")
