"""A real dataset: customer banking queries with human intent labels (PolyAI banking77),
restricted to ten intents so a Choice can list them. Real text, real label noise."""
import random

from brier.questions import Choice

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
QUESTIONS = [Choice("intent", "what the customer wants help with", INTENTS)]


def generate(n, seed=0, split="all", uncertain=0.0):
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", split="test" if split != "train" else "train")
    names = ds.features["label"].names
    keep = {names.index(k): k for k in INTENTS}
    rows = [(r["text"], keep[r["label"]]) for r in ds if r["label"] in keep]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [{"state": text, "questions": QUESTIONS, "labels": {"intent": intent}} for text, intent in rows[:n]]
