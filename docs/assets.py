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


def logo(name, text_fill, ring, bars):
    glyphs, width, ymin, ymax = outline("Brier")
    mark = 64
    cy = mark / 2 + 6
    baseline = cy + (ymax + ymin) / 2
    bar_svg = "".join(
        f'<rect x="{18 + i * 12}" y="{54 - h}" width="8" height="{h}" rx="2" fill="{c}"/>'
        for i, (h, c) in enumerate(zip((12, 22, 32), bars))
    )
    total_w = mark + 22 + width + 8
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="76" viewBox="0 0 {total_w:.0f} 76">
  <rect x="6" y="6" width="{mark}" height="{mark}" rx="15" fill="none" stroke="{ring}" stroke-width="3"/>
  {bar_svg}
  <path d="M18 56 L58 16" stroke="{text_fill}" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="0.1 5"/>
  <g fill="{text_fill}" transform="translate({mark + 22} {baseline:.2f})">{glyphs}</g>
</svg>'''
    (HERE / f"{name}.svg").write_text(svg)


logo("brier", "#1B1F24", "#3557C4", ("#9DB9F5", "#5B8DEF", "#3557C4"))
logo("brier-dark", "#F5F7FA", "#7FA6F0", ("#9DB9F5", "#5B8DEF", "#7FA6F0"))

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
