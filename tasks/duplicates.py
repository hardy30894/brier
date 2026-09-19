"""Duplicate detection: do two contact records describe the same person? A Claim with computable
ground truth, held out from training as a second unseen task."""
import random

from brier.questions import Claim

QUESTIONS = [Claim("same", "the two records describe the same person")]

FIRST = ["Priya", "Marcus", "Dana", "Tomás", "Aisha", "Jonas", "Mei", "Oluwaseun", "Elena", "Rahul"]
LAST = ["Raman", "Hale", "Whitfield", "Ortega", "Bello", "Lindqvist", "Chen", "Adeyemi", "Petrova", "Mehta"]
COMPANY = ["Northwind", "Acme Analytics", "Bluefin Labs", "Kestrel Health", "Orbit Freight"]
DOMAIN = {"Northwind": "northwind.io", "Acme Analytics": "acme-analytics.com", "Bluefin Labs": "bluefin.dev", "Kestrel Health": "kestrel.health", "Orbit Freight": "orbitfreight.com"}


def _person(rng):
    f, l, c = rng.choice(FIRST), rng.choice(LAST), rng.choice(COMPANY)
    phone = f"+1 {rng.randint(200, 989)} {rng.randint(200, 999)} {rng.randint(1000, 9999)}"
    return {"name": f"{f} {l}", "email": f"{f.lower()}.{l.lower()}@{DOMAIN[c]}", "phone": phone, "company": c}


def _variant(p, rng):
    q = dict(p)
    f, l = p["name"].split()
    kind = rng.choice(["initial", "email_fmt", "phone_fmt", "typo", "company_short", "missing"])
    if kind == "initial":
        q["name"] = f"{f[0]}. {l}"
    elif kind == "email_fmt":
        q["email"] = f"{f[0].lower()}{l.lower()}@{DOMAIN[p['company']]}"
    elif kind == "phone_fmt":
        d = p["phone"].replace("+1 ", "").replace(" ", "")
        q["phone"] = f"({d[:3]}) {d[3:6]}-{d[6:]}"
    elif kind == "typo":
        q["name"] = f"{f} {l[:-1]}{'n' if l[-1] != 'n' else 'm'}"
    elif kind == "company_short":
        q["company"] = p["company"].split()[0]
    else:
        q.pop(rng.choice(["email", "phone"]))
    return q


def _near_miss(p, rng):
    q = _person(rng)
    kind = rng.choice(["same_company", "same_last", "same_first_company"])
    if kind == "same_company":
        q["company"] = p["company"]
        q["email"] = q["email"].split("@")[0] + "@" + DOMAIN[p["company"]]
    elif kind == "same_last":
        f = rng.choice([x for x in FIRST if x != p["name"].split()[0]])
        q["name"] = f"{f} {p['name'].split()[1]}"
        q["email"] = f"{f.lower()}.{p['name'].split()[1].lower()}@{DOMAIN[q['company']]}"
    else:
        q["name"] = f"{p['name'].split()[0]} {rng.choice([x for x in LAST if x != p['name'].split()[1]])}"
        q["company"] = p["company"]
        q["email"] = f"{q['name'].split()[0].lower()}.{q['name'].split()[1].lower()}@{DOMAIN[p['company']]}"
    return q


def generate(n, seed=0, split="all", uncertain=0.0):
    """Ambiguous examples share only a name, with everything else missing on one side: 0.5."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = _person(rng)
        if rng.random() < uncertain:
            b, label = {"name": a["name"]}, {True: 0.5, False: 0.5}
        elif rng.random() < 0.5:
            b, label = _variant(a, rng), True
        else:
            b, label = _near_miss(a, rng), False
        out.append({"state": {"record_a": a, "record_b": b}, "questions": QUESTIONS, "labels": {"same": label}})
    return out
