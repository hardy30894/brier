"""Function-call selection: given a user request and a list of tools, which tool should be called?"""
import random

from brier.questions import Choice

TOOLS = {
    "search_docs": ("Search the product documentation for a topic", ["how do I {verb} a {noun}", "where is the setting for {noun}s", "is there a way to {verb} {noun}s in bulk", "docs on {noun} permissions"]),
    "create_ticket": ("Open a support ticket with the team", ["{noun} {broken} and I need someone to look at it", "please open a ticket, my {noun} {broken}", "report: {noun} {broken} since this morning"]),
    "get_invoice": ("Fetch a billing invoice by month", ["send me the invoice for {month}", "I need the {month} receipt for accounting", "download {month} bill"]),
    "cancel_subscription": ("Cancel the current subscription", ["cancel my plan effective {month}", "I want to stop paying, cancel the subscription", "end my {plan} subscription"]),
    "schedule_call": ("Book a call with a human agent", ["can someone call me {when}", "book a call with support for {when}", "I'd like to talk to a person {when}"]),
    "reset_password": ("Send a password reset email", ["I forgot my password", "reset my login, I can't get in", "send a password reset to my email"]),
    "export_data": ("Export account data as a file", ["export all my {noun}s to csv", "I need a full data export before {month}", "give me a download of my {noun}s"]),
    "none": ("No tool applies; reply conversationally", ["thanks, that fixed it", "just saying hi, no issue", "great product, keep it up", "never mind, sorted it myself"]),
}
FILL = {"verb": ["rename", "archive", "share", "duplicate", "tag"], "noun": ["project", "report", "dashboard", "workspace", "webhook"],
        "broken": ["is broken", "keeps crashing", "won't load", "is throwing errors"], "month": ["March", "June", "August", "September"],
        "plan": ["Pro", "Team", "Starter"], "when": ["tomorrow", "this afternoon", "next week", "as soon as possible"]}
NOISE = ["Thanks!", "Sorry for the bother.", "I'm on the {plan} plan.", "It's fairly urgent.", "No rush."]


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


def questions(tool_names):
    return [Choice("tool", "which tool should be called to handle this request", {t: TOOLS[t][0] for t in tool_names})]


def generate(n, seed=0, k=5, split="all", uncertain=0.0):
    """Ambiguous examples join two tools' requests in one message; the label is 0.5 each."""
    rng = random.Random(seed)
    out = []
    names = list(TOOLS)
    for _ in range(n):
        tool = rng.choice(names)
        others = rng.sample([t for t in names if t != tool], k - 1)
        offered = others + [tool]
        rng.shuffle(offered)
        text = _fill(rng.choice(_pool(TOOLS[tool][1], split)), rng)
        label = tool
        if rng.random() < uncertain:
            tool2 = rng.choice([t for t in others if t != "none"] or others)
            text = text + " Also, " + _fill(rng.choice(_pool(TOOLS[tool2][1], split)), rng).lower()
            label = {tool: 0.5, tool2: 0.5}
        if rng.random() < 0.4:
            text = text + " " + _fill(rng.choice(NOISE), rng)
        out.append({"state": {"request": text, "tools": {t: TOOLS[t][0] for t in offered}}, "questions": questions(offered), "labels": {"tool": label}})
    return out
