"""Generated question types.

Every other task in this repository is a question type with many examples: one question, one
set of allowed answers, a thousand situations. A model trained on those learns those questions.
This generator inverts that. It invents the question type itself, thousands of times, and shows
each one a handful of times: a new domain, new attribute names, new options with new wording, a
new rule connecting them. Nothing recurs, so nothing can be memorised, and the only thing that
transfers between examples is the skill of reading a question and its options and applying them
to what the situation says.

Correctness is by construction. The generator draws a rule first, then a situation, then reads
the answer off the rule, so every label is exact and free. Where a rule leaves genuine doubt the
label is a distribution, not a guess.

The rule is always stated in the options themselves. An option is not a bare word but a word and
the condition it stands for, so the question is answerable by a reader who has never seen this
domain, and unanswerable by a model that is only pattern-matching on names it saw in training.
"""
import random

from brier.questions import Pick, Grade, Claim

MAX_STATE_CHARS = 1200


def _fmt(v):
    """Numbers as a person would write them: no scientific notation, thousands separated, and
    decimals only where they carry information. A model comparing 1.46701e+06 with a threshold
    is being tested on notation rather than on reading."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{round(v):,}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _unit(a):
    u = a.get("unit", "")
    if not u:
        return ""
    return u if u.startswith(("%", "°")) else " " + u


def _an(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def draw_record(rng, domain, n_attrs):
    """A situation: a subset of the domain's attributes with values drawn for this record."""
    attrs = rng.sample(domain["attributes"], min(n_attrs, len(domain["attributes"])))
    rec = {}
    for a in attrs:
        if a["kind"] == "categorical":
            rec[a["key"]] = rng.choice(a["values"])
        elif a["kind"] == "number":
            lo, hi = a["range"]
            rec[a["key"]] = round(rng.uniform(lo, hi), 2 if hi - lo < 10 else 0)
        elif a["kind"] == "bool":
            rec[a["key"]] = rng.random() < 0.4
        else:
            rec[a["key"]] = rng.choice(a.get("phrases", ["not recorded"]))
    return attrs, rec


def render_record(domain, attrs, rec, style):
    """The same facts as a table, as JSON, or as prose, so the model cannot rely on one layout."""
    items = [(a["label"], rec[a["key"]]) for a in attrs]
    if style == "json":
        return {a["key"]: rec[a["key"]] for a in attrs}
    if style == "lines":
        return "\n".join(f"{k}: {_fmt(v)}" for k, v in items)
    body = "; ".join(f"the {k.lower()} is {_fmt(v)}" for k, v in items)
    return f"This {domain['entity']} record states that {body}."


def _cond_categorical(rng, a):
    vals = rng.sample(a["values"], min(len(a["values"]), rng.randint(1, max(1, len(a["values"]) // 2))))
    text = f"the {a['label'].lower()} is " + (" or ".join(f"'{v}'" for v in vals))
    return text, (lambda rec, k=a["key"], vs=set(vals): rec.get(k) in vs)


def _cond_number(rng, a):
    lo, hi = a["range"]
    t = round(rng.uniform(lo + (hi - lo) * 0.2, lo + (hi - lo) * 0.8), 2 if hi - lo < 10 else 0)
    over = rng.random() < 0.5
    text = f"the {a['label'].lower()} is {'above' if over else 'below'} {_fmt(t)}{_unit(a)}"
    return text, (lambda rec, k=a["key"], t=t, over=over: (rec.get(k, 0) > t) if over else (rec.get(k, 0) < t))


def _cond_bool(rng, a):
    want = rng.random() < 0.5
    text = f"the {a['label'].lower()} is {'yes' if want else 'no'}"
    return text, (lambda rec, k=a["key"], w=want: bool(rec.get(k)) is w)


# What a condition tests. Matching a named value is reading; comparing two numbers is
# arithmetic, and a small encoder with no pretraining for it will spend all its capacity there
# and learn nothing about reading a question. Numbers stay in the mixture, as a minority.
KIND_WEIGHTS = {"categorical": 0.55, "bool": 0.30, "number": 0.15, "text": 0.0}


def draw_condition(rng, attrs):
    """One readable condition over the record, and the function that decides it."""
    usable = [a for a in attrs if KIND_WEIGHTS.get(a["kind"], 0) > 0]
    if usable:
        w = [KIND_WEIGHTS[a["kind"]] for a in usable]
        a = rng.choices(usable, weights=w, k=1)[0]
    else:
        a = rng.choice(attrs)
    if a["kind"] == "categorical" and len(a["values"]) > 1:
        return _cond_categorical(rng, a)
    if a["kind"] == "number":
        return _cond_number(rng, a)
    if a["kind"] == "bool":
        return _cond_bool(rng, a)
    return _cond_categorical(rng, a) if a.get("values") else _cond_bool(rng, a)


def draw_conjunction(rng, attrs, k):
    """k conditions joined by 'and', stated as one sentence."""
    parts = [draw_condition(rng, attrs) for _ in range(k)]
    text = " and ".join(p[0] for p in parts)
    fns = [p[1] for p in parts]
    return text, (lambda rec, fns=fns: all(f(rec) for f in fns))


def option_names(rng, domain, attrs, n, vocab):
    """Half the time the option labels are arbitrary phrases, which forces the model to read the
    rule rather than guess from the word; the other half they are words from this domain, which
    is what a real question looks like and lets the model learn to use a label that means
    something. A model shown only one of the two learns only half the skill."""
    if rng.random() < 0.5:
        return vocab.label_words(rng, n)
    pool = []
    for a in attrs:
        if a["kind"] == "categorical":
            pool += [str(v) for v in a["values"]]
    pool = [w for w in dict.fromkeys(pool) if len(w) < 40]
    if len(pool) >= n:
        return rng.sample(pool, n)
    return vocab.label_words(rng, n)


def make_pick(rng, domain, attrs, rec, vocab):
    """A Pick whose options are mutually exclusive conditions, tried in order, with a fallback.
    The option text carries its own rule, so the question is self-contained."""
    n = rng.randint(3, 6)
    names = option_names(rng, domain, attrs, n, vocab)
    options, tests = {}, []
    for i in range(n - 1):
        text, fn = (draw_conjunction(rng, attrs, 2) if rng.random() < 0.3 else draw_condition(rng, attrs))
        options[names[i]] = text
        tests.append((names[i], fn))
    options[names[-1]] = "none of the above applies"
    what = rng.choice(vocab.NOUNS)
    q = Pick(f"q_{rng.randrange(10**6)}", rng.choice(vocab.INSTRUCTION_FRAMES).format(what=what), options)
    hit = [nm for nm, fn in tests if fn(rec)]
    if len(hit) == 1:
        label = hit[0]
    elif not hit:
        label = names[-1]
    else:
        label = {nm: 1.0 / len(hit) for nm in hit}   # genuinely ambiguous: a distribution, not a guess
    return q, label, (lambda r, tests=tests, fb=names[-1]: [nm for nm, fn in tests if fn(r)] or [fb])


def make_grade(rng, domain, attrs, rec, vocab):
    """A Grade over a numeric attribute cut into named bands; the bands are stated in the options."""
    nums = [a for a in attrs if a["kind"] == "number"]
    if not nums:
        return None
    a = rng.choice(nums)
    scale = list(rng.choice(vocab.SCALE_NAMES))
    k = len(scale)
    lo, hi = a["range"]
    # Evenly spaced bands, lightly jittered. Uniformly random cuts leave some levels far wider
    # than others, and the narrow ones almost never occur.
    span = (hi - lo) / k
    cuts = [round(lo + span * (j + 1) + rng.uniform(-0.25, 0.25) * span, 2 if hi - lo < 10 else 0) for j in range(k - 1)]
    cuts = sorted(cuts)
    levels, bounds = [], [lo] + cuts + [hi]
    for i, name in enumerate(scale):
        levels.append(f"{name}: {a['label'].lower()} from {_fmt(bounds[i])} to {_fmt(bounds[i + 1])}{_unit(a)}")
    v = rec.get(a["key"], lo)
    idx = sum(1 for c in cuts if v > c)
    q = Grade(f"q_{rng.randrange(10**6)}", rng.choice(vocab.INSTRUCTION_FRAMES).format(what=rng.choice(vocab.NOUNS)), tuple(levels))
    return q, levels[idx], (lambda r, key=a["key"], cuts=cuts, lv=levels: lv[sum(1 for c in cuts if r.get(key, 0) > c)])


def make_claim(rng, domain, attrs, rec, vocab):
    """A Claim whose statement is a condition over the record.

    A condition drawn at random is false far more often than true, which would hand the model a
    free 64% by always answering no. The record is redrawn until it matches the target answer,
    which is chosen by a coin, so yes and no are equally likely and the prior is worth nothing."""
    text, fn = (draw_conjunction(rng, attrs, 2) if rng.random() < 0.4 else draw_condition(rng, attrs))
    want = rng.random() < 0.5
    if fn(rec) is not want:
        for _ in range(40):
            _, cand = draw_record(rng, domain, len(attrs))
            cand = {a["key"]: cand.get(a["key"], rec.get(a["key"])) for a in attrs}
            if fn(cand) is want:
                rec.clear()
                rec.update(cand)
                break
    q = Claim(f"q_{rng.randrange(10**6)}", rng.choice(vocab.CLAIM_FRAMES).format(what=text))
    return q, ("yes" if fn(rec) else "no"), fn


MIX = (("pick", 0.40), ("claim", 0.30), ("grade", 0.20), ("hard", 0.10))


def _kind_at(i, mix):
    """Which kind of question index i gets, by a fixed repeating pattern, so a mixture is
    deterministic and every shard of the stream holds the same proportions."""
    period = 100
    slot, acc = i % period, 0
    for kind, share in mix:
        acc += share * period
        if slot < acc:
            return kind
    return mix[-1][0]


def example_at(i, seed=0, split="all", mix=MIX, kinds=None):
    """The i-th example, and nothing else: a pure function of its index.

    This is the property that makes the generator scale. No shared state, no sequence, no
    skipped indices. Worker r of w takes i = r, r+w, r+2w and can never collide with another
    worker or repeat its work; a run is reproducible from (seed, split, i) alone; and a corpus
    of any size is just a range of integers, which can be produced on as many cores or machines
    as are available and in any order."""
    from . import vocab
    rng = random.Random((seed * 1_000_003) ^ (0 if split == "train" else 0x5EED) ^ (i * 2_654_435_761))
    kind = rng.choice(kinds) if kinds else _kind_at(i, mix)
    hard = kind == "hard"
    if hard:
        kind = rng.choice(("pick", "claim", "grade"))
    lo, hi = (9, 14) if hard else (5, 10)
    domain = rng.choice(vocab.DOMAINS)
    style = rng.choice(("json", "lines", "prose"))
    maker = {"pick": make_pick, "grade": make_grade, "claim": make_claim}[kind]
    made = None
    for _ in range(6):
        attrs, rec = draw_record(rng, domain, rng.randint(lo, hi))
        made = maker(rng, domain, attrs, rec, vocab)
        if made:
            break
    if not made:                                  # a Grade needs a numeric attribute; fall back
        attrs, rec = draw_record(rng, domain, rng.randint(lo, hi))
        made = make_claim(rng, domain, attrs, rec, vocab)
    q, label, check = made
    state = render_record(domain, attrs, rec, style)
    if isinstance(state, str) and len(state) > MAX_STATE_CHARS:
        state = state[:MAX_STATE_CHARS] + " …"
    ex = {"state": state, "questions": [q], "labels": {q.id: label}, "domain": domain["name"], "kind": kind, "index": i}
    ex["_verify"] = (check, dict(rec))   # the rule itself, for the audit; training never reads it
    return ex


def stream(seed=0, split="all", start=0, step=1, limit=None, **kw):
    """An endless supply, sharded by (start, step). One worker of w passes start=rank, step=w."""
    i, made = start, 0
    while limit is None or made < limit:
        yield example_at(i, seed, split, **kw)
        i += step
        made += 1


def generate(n, seed=0, split="all", uncertain=0.0, kinds=None, **kw):
    """`n` examples, each a freshly invented question type. The training and evaluation halves
    draw from disjoint seeds, so evaluation is always on question types never trained on."""
    return [example_at(i, seed, split, kinds=kinds, **kw) for i in range(n)]


SCHEMA = {"schema": generate,
          "schema_pick": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("pick",)),
          "schema_grade": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("grade",)),
          "schema_claim": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("claim",))}
