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

MAX_STATE_CHARS = 1600

# Five difficulty levels, sampled across every draw so a corpus spans easy and hard rather than
# sitting at one setting. Each level says how much there is to read, how many answers to weigh,
# how tangled the conditions are, how much of the record is irrelevant, and how many questions
# ride on one situation.
LEVELS = {
    1: dict(attrs=(4, 6), options=(2, 3), conj=0.05, neg=0.15, noise=0, kinds=("cat", "bool"), qs=(1, 1)),
    2: dict(attrs=(5, 8), options=(3, 4), conj=0.25, neg=0.30, noise=1, kinds=("cat", "bool", "num"), qs=(1, 2)),
    3: dict(attrs=(7, 10), options=(4, 5), conj=0.45, neg=0.35, noise=2, kinds=("cat", "bool", "num", "range"), qs=(1, 3)),
    4: dict(attrs=(9, 12), options=(4, 6), conj=0.60, neg=0.40, noise=3, kinds=("cat", "bool", "num", "range", "compare", "either"), qs=(2, 4)),
    5: dict(attrs=(11, 14), options=(5, 8), conj=0.70, neg=0.45, noise=4, kinds=("cat", "bool", "num", "range", "compare", "either", "atleast"), qs=(2, 5)),
}
LEVEL_MIX = (1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5)   # every level appears; the middle is fattest

# How many answers a question offers. Two is a straight either/or; fifty is an intent or a team
# out of dozens, which is where a fixed per-answer budget breaks and where this head should not.
WIDTHS = ((2, 0.06), (3, 0.16), (4, 0.18), (5, 0.16), (6, 0.12), (10, 0.12), (16, 0.08), (30, 0.07), (50, 0.05))


def held_out_domains(vocab, k=15):
    """Domains reserved for evaluation. Without this the two halves differ only in their
    questions, and a model could be learning the vocabulary of these subjects rather than how to
    read a question; evaluation on unseen subject matter is the point."""
    names = sorted(d["name"] for d in vocab.DOMAINS)
    step = max(1, len(names) // k)
    return set(names[::step][:k])


def domains_for(vocab, split):
    held = held_out_domains(vocab)
    if split == "train":
        return [d for d in vocab.DOMAINS if d["name"] not in held]
    return [d for d in vocab.DOMAINS if d["name"] in held]

# A generated statement is a clause ("the breed is 'brown swiss'"), so it needs frames that take
# a clause. The general-purpose frames expect a noun and produce "The the breed is ...".
CLAIM_CLAUSE_FRAMES = (
    "Is it true of this record that {what}?",
    "Decide whether {what}.",
    "Does this record show that {what}?",
    "State whether it holds that {what}.",
    "Answer yes if {what}, and no otherwise.",
    "Judge the claim: {what}.",
    "Confirm or deny: {what}.",
    "Is the following correct about the record: {what}?",
    "Say whether the record supports this: {what}.",
    "Verify against the record that {what}.",
)


def _val(v):
    """The value as it belongs in a structured field: a whole number is an integer, not 203929.0."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return v


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


def _field(a):
    """The field as it reads inside 'the ... is ...'. Labels that already begin with an article
    would otherwise produce 'the a cancellation has been requested'."""
    lab = a["label"].lower()
    for art in ("a ", "an ", "the "):
        if lab.startswith(art):
            return lab[len(art):]
    return lab


def draw_record(rng, domain, n_attrs):
    """A situation: a subset of the domain's attributes with values drawn for this record."""
    attrs = rng.sample(domain["attributes"], min(n_attrs, len(domain["attributes"])))
    rec = {}
    for a in attrs:
        if a["kind"] == "categorical":
            rec[a["key"]] = rng.choice(a["values"])
        elif a["kind"] == "number":
            lo, hi = a["range"]
            whole = float(lo).is_integer() and float(hi).is_integer() and (hi - lo) >= 3
            rec[a["key"]] = float(rng.randint(int(lo), int(hi))) if whole else round(rng.uniform(lo, hi), 2)
        elif a["kind"] == "bool":
            rec[a["key"]] = rng.random() < 0.4
        else:
            rec[a["key"]] = rng.choice(a.get("phrases", ["not recorded"]))
    return attrs, rec


NOISE_LINES = (
    "logged by the duty operator at the end of the shift",
    "this entry was migrated from the previous system and the formatting may differ",
    "a colleague flagged it for a second look but did not say why",
    "printed for the file; the original is held centrally",
    "reference numbers below are internal and not shown to the customer",
    "the earlier version of this note was corrected for a typing error",
    "attached photographs are stored separately",
    "the handler was covering two desks that day",
)


def render_record(domain, attrs, rec, style, rng=None, noise=0):
    """The same facts as a table, as JSON, or as prose, so the model cannot rely on one layout."""
    items = [(a["label"], rec[a["key"]]) for a in attrs]
    extra = []
    if noise and rng is not None:
        extra = [rng.choice(NOISE_LINES) for _ in range(noise)]
    if style == "json":
        d = {a["label"]: _val(rec[a["key"]]) for a in attrs}
        for j, line in enumerate(extra):
            d[f"note {j + 1}"] = line
        return d
    if style == "nested":
        # a document with sections, the shape a real payload usually arrives in
        half = max(1, len(items) // 2)
        d = {"summary": {k: _val(rec[a["key"]]) for a, (k, _) in zip(attrs[:half], items[:half])},
             "detail": {k: _val(v) for k, v in items[half:]}}
        if extra:
            d["notes"] = extra
        return d
    if style == "table":
        w = max(len(k) for k, _ in items)
        body = "\n".join(f"| {k.ljust(w)} | {_fmt(_val(v))}" for k, v in items)
        return body + ("\n" + "\n".join(f"| note{' ' * (w - 4)} | {e}" for e in extra) if extra else "")
    if style == "thread":
        who = ("operator", "supervisor", "customer", "system")
        lines = []
        for j, (k, v) in enumerate(items):
            lines.append(f"[{who[j % len(who)]}] the {k.lower()} is {_fmt(_val(v))}")
        lines += [f"[system] {e}" for e in extra]
        return "\n".join(lines)
    if style == "lines":
        return "\n".join(f"{k}: {_fmt(_val(v))}" for k, v in items) + ("\n" + "\n".join(f"note: {e}" for e in extra) if extra else "")
    body = "; ".join(f"the {k.lower()} is {_fmt(_val(v))}" for k, v in items)
    if extra:
        body += ". " + " ".join(e.capitalize() + "." for e in extra)
    ent = domain["entity"]
    lead = f"This {ent}" if ent.split()[-1] in ("record", "case", "report", "visit", "session", "order", "ticket") else f"This {ent} record"
    return f"{lead} states that {body}."


def _cond_categorical(rng, a, vals=None, negate=None):
    """Either 'the field is x or y' or 'the field is not x or y'.

    The negation matters more than it looks. A true 'is' rule always names the value the record
    holds, so an option that merely mentions something the record contains is usually the answer,
    and the question can be won by looking for the overlap. A 'is not' rule that names the
    record's own value is false, which inverts that signal, so the overlap stops being an answer
    and the field and the sense of the test have to be read."""
    vals = vals or rng.sample(a["values"], min(len(a["values"]), rng.randint(1, max(1, len(a["values"]) // 2))))
    neg = rng.random() < NEG_P[0] if negate is None else negate
    text = f"the {_field(a)} is {'not ' if neg else ''}" + (" or ".join(f"'{v}'" for v in vals))
    return text, (lambda rec, k=a["key"], vs=set(vals), n=neg: (rec.get(k) not in vs) if n else (rec.get(k) in vs)), a["key"]


def _cond_number(rng, a):
    lo, hi = a["range"]
    t = round(rng.uniform(lo + (hi - lo) * 0.2, lo + (hi - lo) * 0.8), 2 if hi - lo < 10 else 0)
    over = rng.random() < 0.5
    text = f"the {_field(a)} is {'above' if over else 'below'} {_fmt(t)}{_unit(a)}"
    return text, (lambda rec, k=a["key"], t=t, over=over: (rec.get(k, 0) > t) if over else (rec.get(k, 0) < t)), a["key"]


def _cond_bool(rng, a):
    want = rng.random() < 0.5
    text = f"the {_field(a)} is {'yes' if want else 'no'}"
    return text, (lambda rec, k=a["key"], w=want: bool(rec.get(k)) is w), a["key"]


# What a condition tests. Matching a named value is reading; comparing two numbers is
# arithmetic, and a small encoder with no pretraining for it will spend all its capacity there
# and learn nothing about reading a question. Numbers stay in the mixture, as a minority.
def _cond_range(rng, a):
    """'the X is between a and b' — true inside the band, which cannot be read off a single
    mention of a value."""
    lo, hi = a["range"]
    p, q = sorted(round(rng.uniform(lo, hi), 2 if hi - lo < 10 else 0) for _ in range(2))
    if p == q:
        q = p + max(1, (hi - lo) / 10)
    text = f"the {_field(a)} is between {_fmt(p)} and {_fmt(q)}{_unit(a)}"
    return text, (lambda rec, k=a["key"], p=p, q=q: p <= rec.get(k, p - 1) <= q), a["key"]


def _cond_compare(rng, nums):
    """'the X is greater than the Y' — two fields weighed against each other, so neither field
    alone decides it."""
    a, b = rng.sample(nums, 2)
    over = rng.random() < 0.5
    text = f"the {_field(a)} is {'greater' if over else 'less'} than the {_field(b)}"
    fn = (lambda rec, ka=a["key"], kb=b["key"], o=over: (rec.get(ka, 0) > rec.get(kb, 0)) if o else (rec.get(ka, 0) < rec.get(kb, 0)))
    return text, fn, a["key"]


def _cond_either(rng, attrs, draw):
    """Two tests joined by 'or': one of them holding is enough, so both have to be read."""
    parts, used = [], set()
    for _ in range(2):
        left = [x for x in attrs if x["key"] not in used]
        if not left:
            break
        c = draw(rng, left)
        used.add(c[2])
        parts.append(c)
    if len(parts) < 2:
        return parts[0] if parts else None
    text = f"either {parts[0][0]}, or {parts[1][0]}"
    fns = [p[1] for p in parts]
    return text, (lambda rec, fns=fns: any(f(rec) for f in fns)), None


def _cond_atleast(rng, attrs, draw):
    """'at least two of the following hold' — counting, which no single field answers."""
    parts, used = [], set()
    for _ in range(3):
        left = [x for x in attrs if x["key"] not in used]
        if not left:
            break
        c = draw(rng, left)
        used.add(c[2])
        parts.append(c)
    if len(parts) < 3:
        return None
    k = rng.choice((2, 2, 3))
    text = f"at least {'two' if k == 2 else 'three'} of these hold: " + "; ".join(p[0] for p in parts)
    fns = [p[1] for p in parts]
    return text, (lambda rec, fns=fns, k=k: sum(1 for f in fns if f(rec)) >= k), None


KIND_WEIGHTS = {"categorical": 0.55, "bool": 0.30, "number": 0.15, "text": 0.0}
NEG_P = [0.35]      # set per level while a question is being drawn


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


def draw_any(rng, attrs, level):
    """One condition, drawn from the forms this difficulty level allows."""
    forms = list(level["kinds"])
    nums = [a for a in attrs if a["kind"] == "number"]
    if "compare" in forms and len(nums) < 2:
        forms.remove("compare")
    if "range" in forms and not nums:
        forms.remove("range")
    pick = rng.choice(forms)
    if pick == "range":
        return _cond_range(rng, rng.choice(nums))
    if pick == "compare":
        return _cond_compare(rng, nums)
    if pick == "either":
        got = _cond_either(rng, attrs, draw_condition)
        return got or draw_condition(rng, attrs)
    if pick == "atleast":
        got = _cond_atleast(rng, attrs, draw_condition)
        return got or draw_condition(rng, attrs)
    return draw_condition(rng, attrs)


def draw_conjunction(rng, attrs, k):
    """k conditions joined by 'and', over distinct attributes: repeating one attribute produces
    'the cancellation is yes and the cancellation is yes', which is noise."""
    pool, parts, used = list(attrs), [], set()
    for _ in range(k):
        left = [a for a in pool if a["key"] not in used]
        if not left:
            break
        c = draw_condition(rng, left)
        used.add(c[2])
        parts.append(c[:2])
    if len(parts) == 1:
        return parts[0][0], parts[0][1], None
    text = " and ".join(p[0] for p in parts)
    fns = [p[1] for p in parts]
    return text, (lambda rec, fns=fns: all(f(rec) for f in fns)), None


def make_wide_pick(rng, domain, attrs, rec, vocab, n_opts):
    """A question with many answers: ten, thirty, fifty.

    Real decisions routinely choose one of dozens, an intent from seventy-seven or a team from
    forty, and that is where a head that gives each answer a slice of a fixed budget falls apart.
    Options here are the cross-product of two categorical fields, so a large answer set comes
    with exact rules and exactly one of them can hold."""
    cats = [a for a in attrs if a["kind"] == "categorical" and len(a["values"]) >= 3]
    if len(cats) < 2:
        return None
    a, b = rng.sample(cats, 2)
    truth = (rec.get(a["key"]), rec.get(b["key"]))
    every = [(x, y) for x in a["values"] for y in b["values"]]
    if len(every) < 6:
        return None
    # Distractors that share one half of the true pair. An answer set where only the correct
    # option names anything the record contains can be found by looking for the overlap; here
    # most of the wrong options name a value the record holds too, on one side or the other, so
    # both halves of every pair have to be checked.
    near = [c for c in every if c != truth and (c[0] == truth[0] or c[1] == truth[1])]
    far = [c for c in every if c != truth and c not in near]
    rng.shuffle(near)
    rng.shuffle(far)
    need = max(5, n_opts - 1)
    combos = (near[: max(2, int(need * 0.6))] + far)[:need]
    include = rng.random() < 0.85          # sometimes the real combination is not offered at all
    if include:
        combos.append(truth)
    rng.shuffle(combos)
    names = vocab.label_words(rng, len(combos) + 1)
    options, tests = {}, []
    for (x, y), nm in zip(combos, names):
        # Some options negate their second half. Without this an option that names two values the
        # record holds is always the answer, and a wide question reduces to spotting the overlap;
        # with it, an option can name both and still be false, so the sense of each test counts.
        neg = rng.random() < 0.4 and (x, y) != truth
        options[nm] = f"the {_field(a)} is '{x}' and the {_field(b)} is {'not ' if neg else ''}'{y}'"
        tests.append((nm, (lambda r, ka=a["key"], kb=b["key"], x=x, y=y, n=neg:
                           r.get(ka) == x and ((r.get(kb) != y) if n else (r.get(kb) == y)))))
    options[names[len(combos)]] = "none of the above applies"
    q = Pick(f"q_{rng.randrange(10**6)}", rng.choice(vocab.INSTRUCTION_FRAMES).format(what=rng.choice(vocab.NOUNS)), options)
    hit = [nm for nm, fn in tests if fn(rec)]
    # Negated options can hold together, so the answer follows the same rule as everywhere else:
    # one match is the answer, none falls back, several is a genuine tie and is labelled as one.
    label = hit[0] if len(hit) == 1 else (names[len(combos)] if not hit else {nm: 1.0 / len(hit) for nm in hit})
    return q, label, (lambda r, tests=tests, fb=names[len(combos)]: [nm for nm, fn in tests if fn(r)] or [fb])


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
    if len(pool) >= n * 2:                       # keep a margin so names and tested values differ
        return rng.sample(pool, n)
    return vocab.label_words(rng, n)


def make_pick(rng, domain, attrs, rec, vocab, lv=None, width=None):
    """A Pick whose options are mutually exclusive conditions, tried in order, with a fallback.
    The option text carries its own rule, so the question is self-contained."""
    lv = lv or LEVELS[3]
    n = width if width and width < 9 else rng.randint(*lv["options"])
    n = max(2, n)
    names = option_names(rng, domain, attrs, n, vocab)
    options, tests = {}, []
    # Most of the time every option tests the SAME field, with a different set of values. If each
    # option named a different field, only the correct one would mention anything the state
    # actually contains, and the question could be answered by looking for the overlap without
    # evaluating anything. Sharing the field removes that shortcut: every option looks equally
    # relevant and the model has to check which value the record holds.
    shared = None
    cats = [a for a in attrs if a["kind"] == "categorical" and len(a["values"]) >= n]
    if cats and rng.random() < 0.35:
        shared = rng.choice(cats)
        pool = list(shared["values"])
        rng.shuffle(pool)
    for i in range(n - 1):
        if shared is not None:
            take = pool[i * max(1, len(pool) // n): (i + 1) * max(1, len(pool) // n)] or [pool[i % len(pool)]]
            # Partition the field's values between the options, and never negate here: a negated
            # test on a shared field is true whenever the record holds anything else, so several
            # options would hold at once and the question would collapse into a tie.
            text, fn, _ = _cond_categorical(rng, shared, vals=take, negate=False)
        else:
            text, fn, _ = (draw_conjunction(rng, attrs, 2) if rng.random() < 0.3 else draw_condition(rng, attrs))
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
        levels.append(f"{name}: {_field(a)} from {_fmt(bounds[i])} to {_fmt(bounds[i + 1])}{_unit(a)}")
    v = rec.get(a["key"], lo)
    idx = sum(1 for c in cuts if v > c)
    q = Grade(f"q_{rng.randrange(10**6)}", rng.choice(vocab.INSTRUCTION_FRAMES).format(what=rng.choice(vocab.NOUNS)), tuple(levels))
    return q, levels[idx], (lambda r, key=a["key"], cuts=cuts, lv=levels: lv[sum(1 for c in cuts if r.get(key, 0) > c)])


def make_claim(rng, domain, attrs, rec, vocab, lv=None):
    """A Claim whose statement is a condition over the record.

    A condition drawn at random is false far more often than true, which would hand the model a
    free 64% by always answering no. The record is redrawn until it matches the target answer,
    which is chosen by a coin, so yes and no are equally likely and the prior is worth nothing."""
    lv = lv or LEVELS[3]
    text, fn, _ = (draw_conjunction(rng, attrs, 2) if rng.random() < lv["conj"] else draw_any(rng, attrs, lv))
    # Balance yes and no by choosing a different statement, never by changing the record: the
    # record is shared by every question about this situation, and rewriting it here silently
    # invalidated the answers already computed for the others.
    want = rng.random() < 0.5
    if bool(fn(rec)) is not want:
        for _ in range(30):
            cand = (draw_conjunction(rng, attrs, 2) if rng.random() < lv["conj"] else draw_any(rng, attrs, lv))
            if cand and bool(cand[1](rec)) is want:
                text, fn = cand[0], cand[1]
                break
    frame = rng.choice(CLAIM_CLAUSE_FRAMES)
    q = Claim(f"q_{rng.randrange(10**6)}", frame.format(what=text))
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


def _weighted(rng, pairs):
    r, acc = rng.random(), 0.0
    for v, w in pairs:
        acc += w
        if r < acc:
            return v
    return pairs[-1][0]


def example_at(i, seed=0, split="all", mix=MIX, kinds=None, level=None):
    """The i-th example, and nothing else: a pure function of its index.

    One situation, one to five typed questions about it, at a difficulty drawn from the whole
    range. No shared state and no sequence, so worker r of w takes i = r, r+w, r+2w and can never
    collide or repeat, a run is reproducible from (seed, split, i) alone, and a corpus of any
    size is a range of integers."""
    from . import vocab
    rng = random.Random((seed * 1_000_003) ^ (0 if split == "train" else 0x5EED) ^ (i * 2_654_435_761))
    lv = LEVELS[level or rng.choice(LEVEL_MIX)]
    NEG_P[0] = lv["neg"]
    domain = rng.choice(domains_for(vocab, split))
    style = rng.choice(("json", "lines", "prose", "nested", "table", "thread"))
    attrs, rec = draw_record(rng, domain, rng.randint(*lv["attrs"]))

    questions, labels, checks = [], {}, {}
    want = rng.randint(*lv["qs"])
    for _ in range(want):
        kind = rng.choice(kinds) if kinds else _kind_at(rng.randrange(100), mix)
        if kind == "hard":
            kind = rng.choice(("pick", "claim", "grade"))
        made = None
        if kind == "pick":
            width = _weighted(rng, WIDTHS)
            if width >= 10:
                made = make_wide_pick(rng, domain, attrs, rec, vocab, width)
            if made is None:
                made = make_pick(rng, domain, attrs, rec, vocab, lv, width)
        elif kind == "grade":
            made = make_grade(rng, domain, attrs, rec, vocab)
        if made is None:
            made = make_claim(rng, domain, attrs, rec, vocab, lv)
        q, label, check = made
        if q.id in labels or any(x.instructions == q.instructions for x in questions):
            continue
        questions.append(q)
        labels[q.id] = label
        checks[q.id] = check
    if not questions:
        return example_at(i + 1_000_003, seed, split, mix, kinds, level)

    state = render_record(domain, attrs, rec, style, rng, lv["noise"])
    if isinstance(state, str) and len(state) > MAX_STATE_CHARS:
        state = state[:MAX_STATE_CHARS] + " …"
    return {"state": state, "questions": questions, "labels": labels, "domain": domain["name"],
            "kind": questions[0].__class__.__name__.lower(), "level": level or 0, "index": i,
            "_verify": (checks, dict(rec))}


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
          **{f"schema_L{L}": (lambda n, seed=0, split="all", uncertain=0.0, L=L: [example_at(i, seed, split, level=L) for i in range(n)]) for L in LEVELS},
          "schema_pick": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("pick",)),
          "schema_grade": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("grade",)),
          "schema_claim": lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, kinds=("claim",))}
