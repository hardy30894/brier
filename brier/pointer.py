"""The pointer head: one forward pass per request, every question and every option scored from
the prompt's own hidden states.

The prompt lists each question with its options. For a question, the hidden state at the end
of its line is the query; the hidden state at the last token of each listed option is that
option's key; the score is a scaled bilinear form of the two. No candidate rows, no vocabulary
projection: the cost of a request is one pass over the prompt, whatever the number of questions
or options. The head only scores options that were listed, so the type guarantee is kept."""
import json
import time
from dataclasses import dataclass

import torch

from .questions import render_state, Pick, Grade, Claim, Answer, Answers, confidence_of
from .scorer import softmax


class PointerHead(torch.nn.Module):
    def __init__(self, d: int, dh: int = 256):
        super().__init__()
        self.q = torch.nn.Linear(d, dh, bias=False)
        self.k = torch.nn.Linear(d, dh, bias=False)
        self.scale = dh ** -0.5
        self.dh = dh

    def forward(self, h_query, h_keys):
        """h_query: (d,), h_keys: (n, d) -> (n,) scores."""
        return (self.k(h_keys.float()) @ self.q(h_query.float())) * self.scale


@dataclass
class Layout:
    text: str
    query_char: list[int]            # per question: char index (exclusive end) of the question line
    option_chars: list[list[int]]    # per question, per option: char index (exclusive end) of the option mention


def render(state, questions) -> Layout:
    """The same prompt the row scorer uses, but with the character position of every option and
    of every question's end recorded as it is written."""
    head = ("Read the state and answer every question with exactly one of its allowed answers. "
            "Reply with a single JSON object keyed by question id and nothing else.\n\n"
            f"State:\n{render_state(state)}\n\nQuestions:\n")
    parts, qc, oc = [head], [], []
    pos = len(head)
    for q in questions:
        if isinstance(q, Claim):
            clar = "" if (q.true, q.false) == ("yes", "no") else f" (yes means: {q.true}; no means: {q.false})"
            line = f'- "{q.id}" ({q.instructions}) '
            ends = []
            for i, lab in enumerate(("yes", "no")):
                line += lab
                ends.append(pos + len(line))
                line += " or " if i == 0 else ""
            line += clar
        else:
            lead = f'- "{q.id}" ({q.instructions}) ' + ("one of, from lowest to highest: " if isinstance(q, Grade) else "one of: ")
            line = lead
            ends = []
            labels = q.labels
            for i, lab in enumerate(labels):
                line += lab
                ends.append(pos + len(line))
                if i < len(labels) - 1:
                    line += ", "
        line += "\n"
        parts.append(line)
        oc.append(ends)
        qc.append(pos + len(line))
        pos += len(line)
    return Layout("".join(parts), qc, oc)


def char_to_token(offsets, char_end):
    """Index of the last token whose span ends at or before char_end."""
    best = 0
    for i, (a, b) in enumerate(offsets):
        if b <= char_end and b > 0:
            best = i
        elif a >= char_end:
            break
    return best


def encode(scorer, state, questions):
    lay = render(state, questions)
    full = scorer.chat_prefix(lay.text)
    base = full.find(lay.text)
    enc = scorer.tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    qpos = [char_to_token(offs, base + c) for c in lay.query_char]
    opos = [[char_to_token(offs, base + c) for c in row] for row in lay.option_chars]
    return ids, qpos, opos


def decoder(model):
    """The transformer body without the vocabulary head, through PEFT and DDP wrappers."""
    m = model
    for _ in range(6):
        if hasattr(m, "embed_tokens"):
            return m
        m = getattr(m, "module", None) or getattr(m, "base_model", None) or getattr(m, "model", None) or m
        if m is None:
            break
    raise RuntimeError("no decoder body found")


def pointer_scores(scorer, head, state, questions, grad=False):
    """One forward pass over the prompt, through the transformer body only (no vocabulary
    projection); returns a list (per question) of score tensors (per option)."""
    ids, qpos, opos = encode(scorer, state, questions)
    x = torch.tensor([ids], device=scorer.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = decoder(scorer.model)(input_ids=x).last_hidden_state[0]
        return [head(h[qp], h[torch.tensor(op, device=scorer.device)]) for qp, op in zip(qpos, opos)]


def ask_pointer(scorer, head, state, questions, temperatures=None) -> Answers:
    t0 = time.perf_counter()
    raw = pointer_scores(scorer, head, state, questions)
    temperatures = temperatures or {}
    out = {}
    for q, s in zip(questions, raw):
        s = s.tolist()
        probs = softmax(s, temperatures.get(q.id, 1.0))
        dist = dict(zip(q.labels, probs))
        conf, margin = confidence_of(probs)
        if isinstance(q, Claim):
            out[q.id] = Answer(q.id, "claim", dist["yes"], dist, conf, margin, dict(zip(q.labels, s)))
        else:
            best = max(dist, key=dist.get)
            exp = sum(i * p for i, p in enumerate(probs)) if isinstance(q, Grade) else None
            out[q.id] = Answer(q.id, "grade" if isinstance(q, Grade) else "pick", best, dist, conf, margin, dict(zip(q.labels, s)), expected=exp)
    return Answers(out, (time.perf_counter() - t0) * 1000)
