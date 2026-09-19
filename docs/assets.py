"""Regenerates the logo and figure. Text in the logo is outlined so it renders the same everywhere."""
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
INK, MUTED, FAINT, BLUE, PALE, VIOLET, LINE = "#1B1F24", "#5B6B7C", "#8A94A0", "#3557C4", "#EEF3FF", "#8A2BE2", "#D8DEE6"


def figure():
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520" font-family="{FONT}">')
    a('<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#5B6B7C"/></marker></defs>')
    a(f'<rect width="900" height="520" rx="14" fill="#FFFFFF" stroke="{LINE}"/>')

    a(f'<text x="450" y="36" text-anchor="middle" font-size="16" font-weight="700" fill="{INK}">Every field, every allowed value, one forward pass</text>')

    # input card
    a(f'<rect x="36" y="64" width="222" height="216" rx="10" fill="#F6F8FB" stroke="{LINE}"/>')
    a(f'<text x="52" y="90" font-size="12" font-weight="700" fill="{MUTED}" letter-spacing="1">INPUT</text>')
    a(f'<text x="52" y="112" font-size="12.5" fill="{INK}">“Webhooks fire twice since</text>')
    a(f'<text x="52" y="129" font-size="12.5" fill="{INK}">3am. Need this today.”</text>')
    a(f'<text x="52" y="164" font-size="12" font-weight="700" fill="{MUTED}" letter-spacing="1">SCHEMA</text>')
    for i, (name, vals) in enumerate((("category", "billing · bug · feature · account · other"), ("priority", "low · medium · high · urgent"), ("needs_human", "yes · no"))):
        y = 186 + i * 32
        a(f'<text x="52" y="{y}" font-size="12.5" font-weight="700" fill="{BLUE}">{name}</text>')
        a(f'<text x="52" y="{y + 15}" font-size="11" fill="{MUTED}">{vals}</text>')

    a('<line x1="260" y1="172" x2="292" y2="172" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    cx0 = 300
    groups = ((5, "#3557C4"), (4, "#5B8DEF"), (2, "#9DB9F5"))
    y = 78
    for n, c in groups:
        for _ in range(n):
            a(f'<rect x="{cx0}" y="{y}" width="92" height="9" rx="2" fill="{c}" opacity="0.9"/>')
            a(f'<line x1="{cx0 + 92}" y1="{y + 4.5}" x2="{cx0 + 120}" y2="172" stroke="{c}" stroke-width="1" opacity="0.5"/>')
            y += 13
        y += 6
    a(f'<text x="{cx0 + 46}" y="{y + 8}" text-anchor="middle" font-size="11" fill="{MUTED}">11 candidates</text>')
    a(f'<text x="{cx0 + 46}" y="{y + 22}" text-anchor="middle" font-size="11" fill="{MUTED}">= 5 + 4 + 2</text>')

    # model block
    a(f'<rect x="{cx0 + 122}" y="120" width="150" height="104" rx="12" fill="{PALE}" stroke="{BLUE}" stroke-width="1.5"/>')
    a(f'<text x="{cx0 + 197}" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">any open LM</text>')
    a(f'<text x="{cx0 + 197}" y="180" text-anchor="middle" font-size="12" fill="{BLUE}" font-weight="700">1 batched pass</text>')
    a(f'<text x="{cx0 + 197}" y="198" text-anchor="middle" font-size="11" fill="{MUTED}">log-prob of each value</text>')
    a(f'<line x1="{cx0 + 274}" y1="172" x2="{cx0 + 318}" y2="172" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')

    # decision card
    dx = 642
    a(f'<rect x="{dx}" y="64" width="222" height="216" rx="10" fill="#F6F8FB" stroke="{LINE}"/>')
    a(f'<text x="{dx + 16}" y="90" font-size="12" font-weight="700" fill="{MUTED}" letter-spacing="1">TYPED DECISION</text>')
    for i, (name, val, p) in enumerate((("category", "bug", 0.86), ("priority", "high", 0.59), ("needs_human", "no", 0.94))):
        y = 116 + i * 56
        a(f'<text x="{dx + 16}" y="{y}" font-size="12.5" font-weight="700" fill="{INK}">{name}</text>')
        a(f'<text x="{dx + 206}" y="{y}" text-anchor="end" font-size="12.5" font-weight="700" fill="{BLUE}">{p:.2f}</text>')
        a(f'<rect x="{dx + 16}" y="{y + 8}" width="190" height="9" rx="4.5" fill="#E3E8EF"/>')
        a(f'<rect x="{dx + 16}" y="{y + 8}" width="{190 * p:.0f}" height="9" rx="4.5" fill="{BLUE}"/>')
        a(f'<text x="{dx + 16}" y="{y + 32}" font-size="12" fill="{MUTED}">{val}</text>')

    # measured strip
    a(f'<line x1="36" y1="308" x2="864" y2="308" stroke="{LINE}"/>')
    a(f'<text x="36" y="336" font-size="12" font-weight="700" fill="{MUTED}" letter-spacing="1">MEASURED</text>')
    a(f'<text x="864" y="336" text-anchor="end" font-size="11" fill="{FAINT}">Qwen3-4B · RTX 4090 · 200 tickets · 3 fields per decision</text>')
    scale = 620 / 1268
    x0, w_json, w_brier = 200, 1268 * scale, 111 * scale
    a(f'<text x="36" y="372" font-size="12.5" font-weight="700" fill="{VIOLET}">generate JSON</text>')
    a(f'<text x="36" y="388" font-size="11" fill="{MUTED}">token by token, then parse</text>')
    a(f'<rect x="{x0}" y="360" width="{w_json:.0f}" height="22" rx="4" fill="#EFE6FB" stroke="{VIOLET}" stroke-width="1"/>')
    tx = x0 + 4
    while tx < x0 + w_json - 8:
        a(f'<rect x="{tx:.0f}" y="364" width="9" height="14" rx="1.5" fill="{VIOLET}" opacity="0.55"/>')
        tx += 13
    a(f'<text x="{x0 + w_json + 8:.0f}" y="376" font-size="12.5" font-weight="700" fill="{VIOLET}">1268 ms</text>')
    a(f'<text x="36" y="420" font-size="12.5" font-weight="700" fill="{BLUE}">Brier</text>')
    a(f'<text x="36" y="436" font-size="11" fill="{MUTED}">one pass, typed</text>')
    a(f'<rect x="{x0}" y="408" width="{w_brier:.0f}" height="22" rx="4" fill="{BLUE}"/>')
    a(f'<text x="{x0 + w_brier + 8:.0f}" y="424" font-size="12.5" font-weight="700" fill="{BLUE}">111 ms</text>')
    a(f'<text x="{x0 + w_brier + 70:.0f}" y="424" font-size="12" fill="{INK}">11× faster · same accuracy (0.59 vs 0.59) · 0 type errors, by construction</text>')

    a(f'<text x="450" y="484" text-anchor="middle" font-size="11" fill="{FAINT}">Fields are scored independently in the same pass. No training yet: probabilities are the model’s own log-likelihoods, softmaxed per field.</text>')
    a('</svg>')
    (HERE / "how-it-works.svg").write_text("\n".join(o))


figure()
print("assets written")


def _svg(w, h):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
         '<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#5B6B7C"/></marker></defs>',
         f'<rect width="{w}" height="{h}" rx="14" fill="#FFFFFF" stroke="{LINE}"/>']
    return o


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
        o.append(f'<rect x="{x}" y="70" width="{w}" height="30" rx="5" fill="#F6F8FB" stroke="{LINE}"/>')
        _t(o, x + w / 2, 90, t.strip(), 13, INK, "400", "middle")
        x += w + 6
    o.append(f'<rect x="{x}" y="70" width="60" height="30" rx="5" fill="{PALE}" stroke="{BLUE}" stroke-dasharray="4 3"/>')
    _t(o, x + 30, 90, "?", 15, BLUE, "700", "middle")
    _t(o, 60, 130, "context (tokens so far)", 11, MUTED)
    o.append(f'<line x1="{x + 30}" y1="104" x2="{x + 30}" y2="150" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    rows = [(" billing", 0.42), (" bug", 0.31), (" account", 0.14), (" other", 0.06), (" feature", 0.04), ("…150,000 more", 0.03)]
    for i, (t, p) in enumerate(rows):
        y = 168 + i * 20
        _t(o, 340, y + 11, t.strip(), 12, INK if i < 5 else FAINT, "400", "end")
        o.append(f'<rect x="350" y="{y}" width="300" height="14" rx="3" fill="#E3E8EF"/>')
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
    o.append(f'<rect x="40" y="84" width="330" height="30" rx="5" fill="#F6F8FB" stroke="{LINE}"/>')
    _t(o, 50, 104, "…Fields: category: one of billing, bug, … ", 12, INK)
    _t(o, 380, 76, "field header", 11, MUTED)
    o.append(f'<rect x="380" y="84" width="130" height="30" rx="5" fill="#F6F8FB" stroke="{LINE}"/>')
    _t(o, 445, 104, '{"category": "', 12, INK, "400", "middle")
    _t(o, 520, 76, "candidate tokens", 11, MUTED)
    for i, (t, lp) in enumerate((("account", "-0.62"),)):
        o.append(f'<rect x="520" y="84" width="90" height="30" rx="5" fill="{PALE}" stroke="{BLUE}"/>')
        _t(o, 565, 104, t, 12, BLUE, "700", "middle")
    _t(o, 620, 104, "log p = −0.62", 12, INK, "700")
    _t(o, 40, 150, "Each field's candidates share the same prompt and header; only the tail differs:", 12, MUTED)
    cands = (("billing", -3.10), ("bug", -1.05), ("feature", -6.40), ("account", -0.62), ("other", -4.20))
    for i, (c, lp) in enumerate(cands):
        y = 172 + i * 26
        o.append(f'<rect x="40" y="{y}" width="470" height="18" rx="3" fill="#F6F8FB" stroke="{LINE}"/>')
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
        o.append(f'<rect x="40" y="{y}" width="330" height="16" rx="3" fill="#F6F8FB" stroke="{LINE}"/>')
        _t(o, 48, y + 12, "prompt (identical in every row)", 10, FAINT)
        o.append(f'<rect x="372" y="{y}" width="110" height="16" rx="3" fill="#F6F8FB" stroke="{LINE}"/>')
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
        o.append(f'<rect x="330" y="{y}" width="240" height="16" rx="3" fill="#E3E8EF"/>')
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
            o.append(f'<rect x="{x0 + 60}" y="{y}" width="140" height="14" rx="3" fill="#E3E8EF"/>')
            o.append(f'<rect x="{x0 + 60}" y="{y}" width="{140 * p:.0f}" height="14" rx="3" fill="{BLUE}"/>')
            _t(o, x0 + 208, y + 11, f"{p:.2f}", 11, INK)
    _t(o, 60, 214, "probability = softmax(score / T). T is chosen to minimise negative log-likelihood on half the data and evaluated on the other half.", 11, FAINT)
    _t(o, 60, 230, "The argmax never changes, so accuracy is identical at every T. Only honesty changes.", 11, FAINT)
    _write(o, "temperature")


def fig_training():
    o = _svg(900, 420)
    _t(o, 450, 34, "How the backbone was trained (today) and how Brier will train on top of it (milestone 2)", 15, INK, "700", "middle")
    stages = (("pretraining", "predict the next token", "trillions of tokens of text", "→ general knowledge, reading"),
              ("instruction tuning", "imitate helpful answers", "curated prompt/response pairs", "→ follows instructions, JSON"),
              ("preference tuning", "reward preferred answers", "human or model preferences", "→ confident, agreeable tone"))
    for i, (name, obj, data, out) in enumerate(stages):
        x = 40 + i * 275
        o.append(f'<rect x="{x}" y="60" width="255" height="96" rx="10" fill="#F6F8FB" stroke="{LINE}"/>')
        _t(o, x + 14, 84, name, 13, INK, "700")
        _t(o, x + 14, 104, f"objective: {obj}", 11, MUTED)
        _t(o, x + 14, 120, f"data: {data}", 11, MUTED)
        _t(o, x + 14, 140, out, 11, BLUE)
        if i < 2:
            o.append(f'<line x1="{x + 257}" y1="108" x2="{x + 273}" y2="108" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    _t(o, 40, 182, "published open weights (e.g. Qwen3-4B). Milestone 1 uses them exactly as released: good at ranking values, bad at stating probabilities.", 11, FAINT)

    o.append(f'<line x1="40" y1="200" x2="860" y2="200" stroke="{LINE}"/>')
    o.append(f'<rect x="40" y="220" width="200" height="170" rx="10" fill="#F6F8FB" stroke="{LINE}"/>')
    _t(o, 54, 244, "data with ground truth", 13, INK, "700")
    for i, s in enumerate(("synthetic environments", "(answer is computable)", "distilled labels from a", "frontier model, parsed to types", "logs from real software", "with a checkable outcome")):
        _t(o, 54, 266 + i * 16, s, 11, MUTED if i % 2 == 0 else FAINT)
    o.append(f'<line x1="242" y1="305" x2="268" y2="305" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<rect x="270" y="220" width="220" height="170" rx="10" fill="{PALE}" stroke="{BLUE}" stroke-width="1.5"/>')
    _t(o, 284, 244, "backbone + typed heads", 13, INK, "700")
    _t(o, 284, 266, "the LM head is dropped", 11, MUTED)
    _t(o, 284, 282, "one head per field type:", 11, MUTED)
    _t(o, 284, 298, "enum → softmax over values", 11, BLUE)
    _t(o, 284, 314, "bool → one logit", 11, BLUE)
    _t(o, 284, 330, "number → bucketed distribution", 11, BLUE)
    _t(o, 284, 346, "select → pointer over input", 11, BLUE)
    _t(o, 284, 370, "LoRA on the backbone, heads full", 11, MUTED)
    o.append(f'<line x1="492" y1="305" x2="518" y2="305" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<rect x="520" y="220" width="340" height="170" rx="10" fill="#F6F8FB" stroke="{LINE}"/>')
    _t(o, 534, 244, "objective: calibrated decisions", 13, INK, "700")
    _t(o, 534, 266, "1. cross-entropy on typed labels: a proper scoring", 11, MUTED)
    _t(o, 534, 281, "   rule, minimised only by honest probabilities", 11, MUTED)
    _t(o, 534, 301, "2. outcome reward on environments: Brier score of", 11, MUTED)
    _t(o, 534, 316, "   the stated probability, penalising confident misses", 11, MUTED)
    _t(o, 534, 336, "3. gate: ECE on held-out data must beat the", 11, MUTED)
    _t(o, 534, 351, "   temperature-scaled milestone 1 number", 11, MUTED)
    _t(o, 534, 375, "no preference model, no pass/fail reward", 11, BLUE)
    _write(o, "training")


def fig_components():
    o = _svg(900, 300)
    _t(o, 450, 34, "A decide() call through the code", 15, INK, "700", "middle")
    boxes = (("your code", "Schema + input", "#F6F8FB", LINE), ("brier/schema.py", "Enum, Bool → candidates", "#F6F8FB", LINE),
             ("brier/scorer.py", "prompt → batch → softmax", PALE, BLUE), ("brier/schema.py", "Decision: value, prob, dist", "#F6F8FB", LINE))
    for i, (name, what, fill, stroke) in enumerate(boxes):
        x = 40 + i * 215
        o.append(f'<rect x="{x}" y="70" width="195" height="70" rx="10" fill="{fill}" stroke="{stroke}"/>')
        _t(o, x + 12, 96, name, 12, INK, "700")
        _t(o, x + 12, 118, what, 11, MUTED)
        if i < 3:
            o.append(f'<line x1="{x + 197}" y1="105" x2="{x + 213}" y2="105" stroke="#5B6B7C" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<line x1="40" y1="170" x2="860" y2="170" stroke="{LINE}"/>')
    _t(o, 40, 196, "measurement, not on the request path", 12, MUTED, "700")
    items = (("tasks/", "synthetic inputs with computable labels"), ("bench.py", "runs scorer and JSON baseline, prints the table"),
             ("brier/metrics.py", "accuracy, ECE, Brier, latency, temperature fit"), ("pod.py", "GPU pod with a hard time-to-live"))
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
        o.append(f'<rect x="{x}" y="{230 - h:.0f}" width="90" height="{h:.0f}" rx="4" fill="{BLUE}"/>')
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
    stages = (("tokens", "ids from the tokenizer", "#F6F8FB"), ("embeddings", "each id → a vector", "#F6F8FB"),
              ("N × blocks", "attention, then MLP", PALE), ("hidden states", "one vector per position", "#F6F8FB"),
              ("LM head", "vector → 150k logits", "#F6F8FB"), ("softmax", "next-token distribution", "#F6F8FB"))
    for i, (name, what, fill) in enumerate(stages):
        x = 30 + i * 143
        o.append(f'<rect x="{x}" y="70" width="128" height="72" rx="10" fill="{fill}" stroke="{BLUE if fill == PALE else LINE}"/>')
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
    o.append(f'<rect x="80" y="70" width="140" height="140" rx="8" fill="#F1F3F6" stroke="{LINE}"/>')
    _t(o, 150, 145, "W", 22, MUTED, "700", "middle")
    _t(o, 150, 230, "frozen, d × d (e.g. 2560 × 2560)", 10, MUTED, "400", "middle")
    _t(o, 260, 145, "+", 26, INK, "700", "middle")
    o.append(f'<rect x="310" y="70" width="140" height="22" rx="4" fill="{BLUE}"/>')
    _t(o, 380, 86, "A  (r × d)", 11, "#FFFFFF", "700", "middle")
    o.append(f'<rect x="310" y="100" width="22" height="110" rx="4" fill="{BLUE}"/>')
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
        o.append(f'<rect x="{x}" y="70" width="128" height="72" rx="10" fill="{PALE if i == 3 else "#F6F8FB"}" stroke="{BLUE if i == 3 else LINE}"/>')
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
