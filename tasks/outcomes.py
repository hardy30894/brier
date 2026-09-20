"""Environments for outcome-reward training on text decisions. The model never sees a label;
it acts, and the environment returns what happened, as a number in [0, 1]. Where the generator
built a genuinely ambiguous case with a soft label, the environment's hidden truth is drawn from
that posterior once, so the outcome carries the same irreducible uncertainty the world would."""
import random

from brier.questions import Grade, Claim


def _truth(label, rng):
    if isinstance(label, dict):
        keys, weights = zip(*label.items())
        return rng.choices(keys, weights=weights, k=1)[0]
    return label


def _name(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def tickets(ex, answers, rng):
    """Routing: a wrong team bounces the ticket (0). Priority: an under-prioritised ticket
    breaches its SLA, half a point per level; over-prioritising wastes a fast lane, a fifth of a
    point per level. Human hand-off: a missed request for a person is the worst outcome (0); an
    unneeded hand-off costs agent time (0.7)."""
    labels = ex["labels"]
    out = {}
    for q in ex["questions"]:
        if q.id not in labels:
            continue
        truth = _name(_truth(labels[q.id], rng))
        chosen = answers[q.id]
        if isinstance(q, Grade):
            levels = list(q.labels)
            gap = levels.index(chosen) - levels.index(truth)
            out[q.id] = max(0.0, 1.0 - (0.5 * -gap if gap < 0 else 0.2 * gap))
        elif isinstance(q, Claim):
            if truth == "yes":
                out[q.id] = 1.0 if chosen == "yes" else 0.0
            else:
                out[q.id] = 1.0 if chosen == "no" else 0.7
        else:
            out[q.id] = 1.0 if chosen == truth else 0.0
    return out


def toolcall(ex, answers, rng):
    """The chosen tool is called: it succeeds (1) or fails (0). Nothing else is revealed."""
    truth = _name(_truth(ex["labels"]["tool"], rng))
    return {"tool": 1.0 if answers["tool"] == truth else 0.0}


def typed(ex, answers, rng):
    """The public benchmark as bandit feedback: the sampled answer is marked against the
    benchmark's hard label (1 or 0; for an ordered scale, 1 minus the distance in levels over the
    scale), and nothing else is revealed. The gold distribution is never used."""
    out = {}
    for q in ex["questions"]:
        g = ex["gold"][q.id]["label"]
        g = {"true": "yes", "false": "no"}.get(str(g).lower(), str(g))
        chosen = answers[q.id]
        if isinstance(q, Grade):
            levels = list(q.labels)
            out[q.id] = 1.0 - abs(levels.index(chosen) - levels.index(g)) / max(1, len(levels) - 1)
        else:
            out[q.id] = 1.0 if chosen == g else 0.0
    return out


ENVS = {"tickets": tickets, "toolcall": toolcall, "typed": typed}
