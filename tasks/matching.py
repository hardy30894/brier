"""Contrastive claim for training: does an invoice line match the order line? Negatives share
most surface details on purpose (same product, different quantity; same total, different
product), so a model cannot learn 'shared cues mean yes'."""
import random

from brier.questions import Claim

QUESTIONS = [Claim("matches", "the invoice line is the same item, quantity and unit price as the order line")]

PRODUCTS = ["steel bracket 40mm", "steel bracket 60mm", "brass hinge", "nylon washer pack", "M6 bolt box", "M8 bolt box", "LED panel 600", "LED panel 1200", "cable tray 2m", "cable tray 3m"]


def _line(rng, product=None):
    p = product or rng.choice(PRODUCTS)
    qty = rng.choice([1, 2, 5, 10, 12, 24, 50, 100])
    price = rng.choice([1.2, 2.5, 4.0, 7.5, 12.0, 19.99, 45.0, 80.0])
    return {"item": p, "qty": qty, "unit_price": price, "total": round(qty * price, 2)}


def _same(a, rng):
    b = dict(a)
    kind = rng.choice(["case", "spacing", "unit", "order"])
    if kind == "case":
        b["item"] = a["item"].upper()
    elif kind == "spacing":
        b["item"] = a["item"].replace(" ", "  ")
    elif kind == "unit":
        b["unit_price"] = f"${a['unit_price']:.2f}"
    else:
        b = {"total": a["total"], "item": a["item"], "unit_price": a["unit_price"], "qty": a["qty"]}
    return b


def _near(a, rng):
    b = dict(a)
    kind = rng.choice(["qty", "price", "sibling", "same_total"])
    if kind == "qty":
        b["qty"] = a["qty"] * 2
        b["total"] = round(b["qty"] * a["unit_price"], 2)
    elif kind == "price":
        b["unit_price"] = round(a["unit_price"] * 1.1, 2)
        b["total"] = round(a["qty"] * b["unit_price"], 2)
    elif kind == "sibling":
        p = a["item"]
        sib = [x for x in PRODUCTS if x != p and x.split()[0] == p.split()[0]]
        b["item"] = sib[0] if sib else rng.choice([x for x in PRODUCTS if x != p])
    else:
        b["qty"], b["unit_price"] = a["unit_price"], a["qty"]
        b["item"] = a["item"]
    return b


def generate(n, seed=0, split="all", uncertain=0.0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = _line(rng)
        if rng.random() < uncertain:
            b = {"item": a["item"], "total": a["total"]}
            label = {True: 0.5, False: 0.5}
        elif rng.random() < 0.5:
            b, label = _same(a, rng), True
        else:
            b, label = _near(a, rng), False
        out.append({"state": {"order_line": a, "invoice_line": b}, "questions": QUESTIONS, "labels": {"matches": label}})
    return out
