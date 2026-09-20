"""Every task exposes examples of the form {state, questions, labels}."""
from . import tickets, gridworld, toolcall, passages, frustration, duplicates, banking, matching
from .public import PUBLIC
from .typed import TYPED

REGISTRY = {
    "tickets": tickets.generate_q,
    "gridworld": gridworld.generate_q,
    "toolcall": toolcall.generate,
    "passages": passages.generate,
    "frustration": frustration.generate,
    "duplicates": duplicates.generate,
    "banking": banking.generate,
    "matching": matching.generate,
    **PUBLIC,
    **TYPED,
}


def examples(name, n, seed=0, split="all", uncertain=0.0):
    return REGISTRY[name](n, seed, split=split, uncertain=uncertain)
