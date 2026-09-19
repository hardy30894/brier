"""A real dataset: customer banking queries with human intent labels (PolyAI banking77),
restricted to ten intents so a Pick can list them. Real text, real label noise."""
import random

from brier.questions import Pick

INTENTS = {
    "card_arrival": "asking when a new card will arrive",
    "lost_or_stolen_card": "reporting a card lost or stolen",
    "declined_card_payment": "a card payment was declined",
    "exchange_rate": "asking about currency exchange rates",
    "top_up_failed": "a top-up did not go through",
    "transfer_not_received_by_recipient": "a transfer the recipient has not received",
    "pin_blocked": "the card PIN is blocked",
    "cancel_transfer": "wants to cancel a transfer",
    "atm_support": "trouble using an ATM",
    "edit_personal_details": "wants to change name, address or other personal details",
}
QUESTIONS = [Pick("intent", "what the customer wants help with", INTENTS)]


URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"


def _rows(split):
    import csv
    import io
    import os
    import urllib.request
    cache = os.path.expanduser(f"~/.cache/brier/banking77_{split}.csv")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with urllib.request.urlopen(URL.format(split=split), timeout=60) as r:
            open(cache, "wb").write(r.read())
    with open(cache, newline="", encoding="utf-8") as f:
        return [(row["text"], row["category"]) for row in csv.DictReader(f)]


def generate(n, seed=0, split="all", uncertain=0.0):
    rows = [(t, c) for t, c in _rows("test" if split != "train" else "train") if c in INTENTS]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [{"state": text, "questions": QUESTIONS, "labels": {"intent": intent}} for text, intent in rows[:n]]
