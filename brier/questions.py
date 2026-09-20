"""The question interface: Pick, Grade and Claim, answered together in one pass."""
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from .scorer import Scorer, softmax


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=1, ensure_ascii=False)


@dataclass(frozen=True)
class Pick:
    """Which of these options? `options` maps a label to an optional description."""
    id: str
    instructions: str
    options: tuple[str, ...] | dict[str, str]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.options) if isinstance(self.options, dict) else self.options

    def render(self) -> str:
        if isinstance(self.options, dict):
            opts = "; ".join(f"{k}: {v}" for k, v in self.options.items())
        else:
            opts = ", ".join(self.options)
        return f'- "{self.id}" ({self.instructions}) one of: {opts}'


@dataclass(frozen=True)
class Grade:
    """Which level on an ordered scale? `levels` is lowest to highest; labels may carry a description."""
    id: str
    instructions: str
    levels: tuple[str, ...] | dict[str, str]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.levels) if isinstance(self.levels, dict) else self.levels

    def render(self) -> str:
        return f'- "{self.id}" ({self.instructions}) one of, from lowest to highest: {", ".join(self.labels)}'


@dataclass(frozen=True)
class Claim:
    """Is this statement true of the state? Answered as a probability."""
    id: str
    instructions: str
    true: str = "yes"
    false: str = "no"

    @property
    def labels(self) -> tuple[str, ...]:
        return ("yes", "no")

    def render(self) -> str:
        clar = "" if (self.true, self.false) == ("yes", "no") else f" (yes means: {self.true}; no means: {self.false})"
        return f'- "{self.id}" ({self.instructions}) yes or no{clar}'


Question = Pick | Grade | Claim


@dataclass
class Answer:
    id: str
    kind: str
    value: Any
    probabilities: dict[str, float]
    confidence: float
    margin: float
    scores: dict[str, float] = field(default_factory=dict)
    expected: float | None = None
    tier: str | None = None

    def top(self, k=3):
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Answers:
    answers: dict[str, Answer]
    latency_ms: float

    def __getitem__(self, k) -> Answer:
        return self.answers[k]

    def __iter__(self):
        return iter(self.answers.values())


def confidence_of(probs: list[float]) -> tuple[float, float]:
    """Confidence is the top probability; margin is top minus runner-up. Both are stated, not hidden."""
    s = sorted(probs, reverse=True)
    return s[0], s[0] - (s[1] if len(s) > 1 else 0.0)


def answer_from_scores(q: Question, scores: list[float], T: float = 1.0) -> Answer:
    labels = q.labels
    probs = softmax(scores, T)
    dist = dict(zip(labels, probs))
    conf, margin = confidence_of(probs)
    if isinstance(q, Claim):
        return Answer(q.id, "claim", dist["yes"], dist, conf, margin, dict(zip(labels, scores)))
    best = max(dist, key=dist.get)
    if isinstance(q, Grade):
        expected = sum(i * p for i, p in enumerate(probs))
        return Answer(q.id, "grade", best, dist, conf, margin, dict(zip(labels, scores)), expected=expected)
    return Answer(q.id, "pick", best, dist, conf, margin, dict(zip(labels, scores)))


def build_prompt(scorer: Scorer, state: Any, questions: list[Question]) -> str:
    qs = "\n".join(q.render() for q in questions)
    return scorer.chat_prefix(
        "Read the state and answer every question with exactly one of its allowed answers. "
        "Reply with a single JSON object keyed by question id and nothing else.\n\n"
        f"State:\n{render_state(state)}\n\nQuestions:\n{qs}"
    )


def groups_for(questions: list[Question]):
    return [(f'{{"{q.id}": "', list(q.labels)) for q in questions]


def ask(scorer: Scorer, state: Any, questions: list[Question], temperatures: dict[str, float] | None = None) -> Answers:
    """Every question, every allowed answer, one pass. When the scorer carries a pointer head,
    Pick and Claim questions go through it (one row for all of them) and Grade questions
    through the row scorer, which keeps the language prior that ordered scales need.
    Questions are independent either way."""
    if getattr(scorer, "pointer", None) is not None:
        from .pointer import ask_pointer
        all_pointer = scorer.pointer_grade or getattr(scorer, "encoder", False)   # an encoder has no row scorer
        fast = questions if all_pointer else [q for q in questions if not isinstance(q, Grade)]
        slow = [] if all_pointer else [q for q in questions if isinstance(q, Grade)]
        t0 = time.perf_counter()
        out = {}
        if fast:
            out.update(ask_pointer(scorer, scorer.pointer, state, fast, temperatures).answers)
        if slow:
            out.update(_ask_rows(scorer, state, slow, temperatures).answers)
        return Answers({q.id: out[q.id] for q in questions}, (time.perf_counter() - t0) * 1000)
    return _ask_rows(scorer, state, questions, temperatures)


def _ask_rows(scorer: Scorer, state: Any, questions: list[Question], temperatures: dict[str, float] | None = None) -> Answers:
    t0 = time.perf_counter()
    raw = scorer.score_groups(build_prompt(scorer, state, questions), groups_for(questions))
    temperatures = temperatures or {}
    out = {q.id: answer_from_scores(q, s, temperatures.get(q.id, 1.0)) for q, s in zip(questions, raw)}
    return Answers(out, (time.perf_counter() - t0) * 1000)
