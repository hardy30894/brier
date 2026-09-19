"""Customer frustration on an ordered scale: a Grade question."""
import random

from brier.questions import Grade

LEVELS = {"calm": "neutral or friendly, no complaint", "annoyed": "mild irritation, still polite", "angry": "clearly upset, demanding", "furious": "hostile, threats to leave or escalate"}
QUESTIONS = [Grade("frustration", "how frustrated is the customer", LEVELS)]

TEMPLATES = {
    "calm": ["Quick question: how do I change my billing email? Thanks!", "Loving the new dashboard. Is there a keyboard shortcut list somewhere?", "Could you point me to the docs for webhooks when you get a chance?", "Just checking whether the {feature} feature is on the roadmap."],
    "annoyed": ["The {feature} page has been slow all week, it's getting a bit tedious.", "Second time I've had to re-enter my card details. Can this be fixed?", "Your docs for {feature} are out of date, wasted an hour on that.", "Not thrilled that the export failed again, please look into it."],
    "angry": ["This is the third outage this month. What exactly are we paying for?", "I've emailed twice about {feature} and nobody has replied. Unacceptable.", "Your update deleted my saved {feature} settings. I want an explanation.", "Stop charging me for a plan I downgraded from two weeks ago."],
    "furious": ["I am done. Cancel everything and refund every cent or I'm going to my bank and my lawyer.", "You people have wasted weeks of my time with {feature}. I'll be telling everyone I know to avoid you.", "Absolutely disgraceful support. Escalate this to someone who actually cares, NOW.", "If this isn't fixed by tomorrow I'm filing a complaint and posting every email publicly."],
}
FEATURES = ["analytics", "billing", "SSO", "reports", "webhooks"]
NOISE = ["Sent from my phone.", "Account ref 4471.", "We're on the Team plan."]


def _pool(items, split):
    """Template hold-out: train and eval draw from disjoint halves of every template list."""
    if split == "train":
        return items[0::2]
    if split == "eval":
        return items[1::2]
    return list(items)


def generate(n, seed=0, split="all", uncertain=0.0):
    """Ambiguous examples join sentences from two adjacent levels; the label is 0.5 each."""
    rng = random.Random(seed)
    out = []
    levels = list(LEVELS)
    for _ in range(n):
        i = rng.randrange(len(levels))
        lvl = levels[i]
        text = rng.choice(_pool(TEMPLATES[lvl], split)).replace("{feature}", rng.choice(FEATURES))
        label = lvl
        if rng.random() < uncertain:
            j = i + 1 if i + 1 < len(levels) else i - 1
            text = text + " " + rng.choice(_pool(TEMPLATES[levels[j]], split)).replace("{feature}", rng.choice(FEATURES))
            label = {lvl: 0.5, levels[j]: 0.5}
        if rng.random() < 0.3:
            text += " " + rng.choice(NOISE)
        out.append({"state": text, "questions": QUESTIONS, "labels": {"frustration": label}})
    return out
