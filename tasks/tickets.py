import random

from brier.schema import Schema, Enum, Bool

SCHEMA = Schema((
    Enum("category", ("billing", "bug", "feature", "account", "other"), "which team should handle this"),
    Enum("priority", ("low", "medium", "high", "urgent"), "how fast it must be handled"),
    Bool("needs_human", "does this require a human agent rather than an automated reply"),
))

CATEGORY_TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month.",
        "My invoice for {month} shows the wrong amount, it should be {amount} not {amount2}.",
        "Why did my card get billed after I downgraded to the free tier?",
        "The receipt I got doesn't include our VAT number, our finance team needs it.",
        "I switched to annual billing but I'm still seeing monthly charges.",
        "There's a charge on my statement from you that I don't recognize.",
    ],
    "bug": [
        "The dashboard crashes every time I open the {page} page on Safari.",
        "Exporting to CSV returns a 500 error since the update yesterday.",
        "Search results are empty even though I can see the items in the list.",
        "The mobile app logs me out randomly and loses my unsaved changes.",
        "Webhooks are firing twice for every event since around {time}.",
        "Dark mode makes the chart labels unreadable, they render white on white.",
    ],
    "feature": [
        "It would be great if we could export reports as PDF as well as CSV.",
        "Any plans to add a Slack integration for {page} notifications?",
        "Please consider adding keyboard shortcuts for the editor.",
        "Would love a way to bulk-edit tags across many items at once.",
        "A dark mode for the email digests would be really nice.",
        "Could you add support for SSO with Okta?",
    ],
    "account": [
        "I can't log in, it says my password is wrong but I just reset it.",
        "I lost my phone and can't get past two-factor authentication.",
        "How do I change the email address on my account?",
        "My teammate can't see the workspace I invited them to.",
        "I want to transfer ownership of the organization to {name}.",
        "The verification email never arrives, I've checked spam.",
    ],
    "other": [
        "We're a podcast and would like to interview someone from your team.",
        "Do you have a partner program for agencies?",
        "Where can I find your privacy policy and data processing agreement?",
        "I'm a student, is there an academic discount or program?",
        "Is there a public roadmap I can follow?",
        "What's your office address for sending a gift to your support team?",
    ],
}

PRIORITY_PHRASES = {
    "urgent": [
        "This is blocking our entire team right now.",
        "We are losing customers every minute this stays broken.",
        "Production is down for us because of this.",
    ],
    "high": [
        "I need this sorted today if at all possible.",
        "This is pretty important for a launch we have tomorrow.",
        "Please prioritize, it's affecting several of our users.",
    ],
    "medium": [
        "Would be good to get this resolved this week.",
        "Not an emergency but it's getting annoying.",
        "Let me know when you have an update.",
    ],
    "low": [
        "No rush at all, just flagging it.",
        "Whenever you get a chance, it's a minor thing.",
        "Low priority, just curious.",
    ],
}

HUMAN_PHRASES = [
    "I'd like to speak to an actual person about this.",
    "I've already been through the bot twice, I need a human.",
    "Can a real support agent call me back?",
    "Please have someone from your team reply personally, not an auto-response.",
    "I'd rather explain this to a person than keep typing into a form.",
]

NOISE = [
    "Thanks in advance.",
    "By the way, love the new logo.",
    "I'm on the {plan} plan if that matters.",
    "Sent from my phone, sorry for typos.",
    "We've been customers since 2023.",
]

FILL = {
    "plan": ["Pro", "Team", "Starter", "Enterprise"],
    "month": ["March", "July", "August", "September"],
    "amount": ["$49", "$120", "$300"],
    "amount2": ["$98", "$240", "$600"],
    "page": ["analytics", "settings", "billing", "reports"],
    "time": ["noon", "3am UTC", "last Tuesday"],
    "name": ["Priya", "Marcus", "Dana"],
}


def _pool(items, split):
    """Template hold-out: train and eval draw from disjoint halves of every template list."""
    if split == "train":
        return items[0::2]
    if split == "eval":
        return items[1::2]
    return list(items)


def _fill(s, rng):
    for k, vs in FILL.items():
        s = s.replace("{" + k + "}", rng.choice(vs))
    return s


PRIORITY_WEIGHTS = {"low": 3, "medium": 4, "high": 2, "urgent": 1}


def generate(n, seed=0, split="all", uncertain=0.0):
    """`uncertain` is the fraction of examples built to be genuinely ambiguous, with soft labels
    equal to the generator's own posterior: two categories' sentences mixed (0.5 each), or no
    priority sentence at all (the prior over priorities)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        cat = rng.choice(list(CATEGORY_TEMPLATES))
        pri = rng.choices(list(PRIORITY_WEIGHTS), weights=list(PRIORITY_WEIGHTS.values()))[0]
        human = rng.random() < 0.3
        parts = [_fill(rng.choice(_pool(CATEGORY_TEMPLATES[cat], split)), rng)]
        labels = {"category": cat, "priority": pri, "needs_human": "yes" if human else "no"}
        if rng.random() < uncertain:
            if rng.random() < 0.5:
                cat2 = rng.choice([c for c in CATEGORY_TEMPLATES if c != cat])
                parts.append(_fill(rng.choice(_pool(CATEGORY_TEMPLATES[cat2], split)), rng))
                labels["category"] = {cat: 0.5, cat2: 0.5}
                parts.append(rng.choice(_pool(PRIORITY_PHRASES[pri], split)))
            else:
                z = sum(PRIORITY_WEIGHTS.values())
                labels["priority"] = {k: v / z for k, v in PRIORITY_WEIGHTS.items()}
        else:
            parts.append(rng.choice(_pool(PRIORITY_PHRASES[pri], split)))
        if human:
            parts.append(rng.choice(_pool(HUMAN_PHRASES, split)))
        if rng.random() < 0.35:
            parts.append(_fill(rng.choice(NOISE), rng))
        rng.shuffle(parts)
        out.append({"context": " ".join(parts), "labels": labels})
    return out


from brier.questions import Choice, Score, Claim  # noqa: E402

QUESTIONS = [
    Choice("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
    Score("priority", "how fast this must be handled", ("low", "medium", "high", "urgent")),
    Claim("needs_human", "the customer is asking for a human agent rather than an automated reply"),
]


def generate_q(n, seed=0, split="all", uncertain=0.0):
    return [{"state": ex["context"], "questions": QUESTIONS, "labels": ex["labels"]} for ex in generate(n, seed, split, uncertain)]
