"""Reliability diagrams as SVG, one panel per question."""
from .calibrate import Calibration

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
INK, MUTED, FAINT, BLUE, VIOLET, LINE = "#1B1F24", "#5B6B7C", "#8A94A0", "#3557C4", "#8A2BE2", "#D8DEE6"


def _curve(rows, ox, oy, s, color):
    pts = [(ox + s * conf, oy - s * acc) for lo, hi, n, conf, acc in rows if n > 0]
    if not pts:
        return ""
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>' for x, y in pts)
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>{dots}'


def reliability_svg(cal: Calibration, title="") -> str:
    qs = list(cal.questions.values())
    pw, ph, s = 280, 300, 170
    W = 40 + pw * len(qs)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{ph + 40}" viewBox="0 0 {W} {ph + 40}" font-family="{FONT}">',
         f'<rect width="{W}" height="{ph + 40}" rx="14" fill="#FFFFFF" stroke="{LINE}"/>']
    o.append(f'<text x="{W / 2}" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="{INK}">{title or "reliability, held-out folds"}</text>')
    o.append(f'<text x="{W - 20}" y="28" text-anchor="end" font-size="11" fill="{FAINT}"><tspan fill="{VIOLET}">●</tspan> raw   <tspan fill="{BLUE}">●</tspan> calibrated</text>')
    for i, q in enumerate(qs):
        ox, oy = 60 + i * pw, 60 + s
        o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + s}" y2="{oy}" stroke="{MUTED}"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - s}" stroke="{MUTED}"/>')
        o.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + s}" y2="{oy - s}" stroke="{FAINT}" stroke-dasharray="4 4"/>')
        for v in (0, 0.5, 1):
            o.append(f'<text x="{ox + s * v}" y="{oy + 14}" font-size="9" fill="{FAINT}" text-anchor="middle">{v:.1f}</text>')
            o.append(f'<text x="{ox - 6}" y="{oy - s * v + 3}" font-size="9" fill="{FAINT}" text-anchor="end">{v:.1f}</text>')
        o.append(_curve(q.reliability_raw, ox, oy, s, VIOLET))
        o.append(_curve(q.reliability_cal, ox, oy, s, BLUE))
        o.append(f'<text x="{ox + s / 2}" y="{oy + 30}" text-anchor="middle" font-size="12" font-weight="700" fill="{INK}">{q.id}</text>')
        o.append(f'<text x="{ox + s / 2}" y="{oy + 46}" text-anchor="middle" font-size="10" fill="{MUTED}">acc {q.accuracy:.2f} · ECE {q.ece_raw:.2f} → {q.ece_cal:.2f} · T {q.T:.1f}</text>')
        auto = next((t for t in q.tiers if t.name == "auto" and t.min_confidence is not None), None)
        if auto:
            o.append(f'<text x="{ox + s / 2}" y="{oy + 60}" text-anchor="middle" font-size="10" fill="{BLUE}">auto ≥ {auto.min_confidence:.2f} covers {auto.coverage:.0%} at {auto.precision:.2f}</text>')
    o.append("</svg>")
    return "\n".join(o)
