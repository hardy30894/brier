"""Every task exposes examples of the form {state, questions, labels}."""
from . import tickets, gridworld, toolcall, passages, frustration

REGISTRY = {
    "tickets": tickets.generate_q,
    "gridworld": gridworld.generate_q,
    "toolcall": toolcall.generate,
    "passages": passages.generate,
    "frustration": frustration.generate,
}


def examples(name, n, seed=0):
    return REGISTRY[name](n, seed)
