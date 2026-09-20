"""The calibration kit: fit one temperature per question on labelled examples, measure honesty
on held-out folds, and derive the confidence thresholds that reach a target precision."""
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .metrics import ece, brier as brier_score
from .questions import Question, Claim, ask, answer_from_scores
from .scorer import softmax


@dataclass
class Tier:
    name: str
    target_precision: float
    min_confidence: float | None
    precision: float | None
    coverage: float | None


@dataclass
class QuestionCalibration:
    id: str
    T: float
    n: int
    accuracy: float
    ece_raw: float
    ece_cal: float
    brier_raw: float
    brier_cal: float
    mean_conf_raw: float
    mean_conf_cal: float
    reliability_raw: list
    reliability_cal: list
    tiers: list[Tier]

    def tier_of(self, confidence: float) -> str:
        for t in self.tiers:
            if t.min_confidence is not None and confidence >= t.min_confidence:
                return t.name
        return "human"


@dataclass
class Calibration:
    questions: dict[str, QuestionCalibration]
    model: str = ""
    created: float = field(default_factory=time.time)

    @property
    def temperatures(self) -> dict[str, float]:
        return {k: v.T for k, v in self.questions.items()}

    def apply(self, answers):
        for a in answers:
            qc = self.questions.get(a.id)
            if qc is not None:
                a.tier = qc.tier_of(a.confidence)
        return answers

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"model": self.model, "created": self.created, "questions": {k: asdict(v) for k, v in self.questions.items()}}, f, indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        qs = {}
        for k, v in d["questions"].items():
            v["tiers"] = [Tier(**t) for t in v["tiers"]]
            qs[k] = QuestionCalibration(**v)
        return cls(qs, d.get("model", ""), d.get("created", 0))


def acceptable(q: Question, label: Any) -> set[int]:
    """Indices of answers that count as correct. A dict label (soft target) is not evaluable
    this way; evaluation sets carry hard labels only."""
    labels = q.labels
    if isinstance(q, Claim):
        if isinstance(label, bool):
            label = "yes" if label else "no"
    if isinstance(label, dict):
        top = max(label.values())
        name = lambda k: ("yes" if k else "no") if isinstance(k, bool) else str(k)
        return {labels.index(name(k)) for k, v in label.items() if v == top}
    if isinstance(label, (tuple, list, set)):
        return {labels.index(str(l)) for l in label}
    return {labels.index(str(label))}


def target(q: Question, label: Any) -> list[float] | None:
    """A soft target distribution over q.labels when the label is a dict, else None."""
    if not isinstance(label, dict):
        return None
    labels = q.labels
    t = [0.0] * len(labels)
    for k, v in label.items():
        if isinstance(k, bool):
            k = "yes" if k else "no"
        t[labels.index(str(k))] = float(v)
    z = sum(t)
    return [x / z for x in t]


def collect(scorer, questions: list[Question], examples: list[tuple[Any, dict]], log_every=50):
    """Run every example once at T=1 and keep the raw scores. Returns {qid: [(scores, acceptable_idx)]}."""
    records = {q.id: [] for q in questions}
    t0 = time.perf_counter()
    for i, (state, labels) in enumerate(examples):
        ans = ask(scorer, state, questions)
        for q in questions:
            if q.id in labels:
                records[q.id].append(([ans[q.id].scores[l] for l in q.labels], acceptable(q, labels[q.id])))
        if log_every and (i + 1) % log_every == 0:
            print(f"  collected {i + 1}/{len(examples)} ({time.perf_counter() - t0:.0f}s)")
    return records


def nll(records, T):
    total = 0.0
    for scores, acc in records:
        p = softmax(scores, T)
        total -= math.log(max(sum(p[i] for i in acc), 1e-12))
    return total


def fit_T(records, grid=None) -> float:
    grid = grid if grid is not None else np.exp(np.linspace(math.log(0.05), math.log(20), 160))
    return float(min(grid, key=lambda T: nll(records, T)))


def confidences(records, T):
    conf, ok = [], []
    for scores, acc in records:
        p = softmax(scores, T)
        top = int(np.argmax(p))
        conf.append(p[top])
        ok.append(float(top in acc))
    return np.asarray(conf), np.asarray(ok)


def thresholds(conf, ok, targets) -> list[Tier]:
    """For each target precision, the lowest confidence cutoff whose precision above it meets the target."""
    order = np.argsort(-conf)
    c, o = conf[order], ok[order]
    cum = np.cumsum(o) / np.arange(1, len(o) + 1)
    tiers = []
    for name, target in targets:
        best = None
        for k in range(len(c) - 1, -1, -1):
            if cum[k] >= target and (k + 1) >= max(5, len(c) * 0.02):
                best = k
                break
        if best is None:
            tiers.append(Tier(name, target, None, None, None))
        else:
            tiers.append(Tier(name, target, float(c[best]), float(cum[best]), float((best + 1) / len(c))))
    return tiers


def calibrate_question(q: Question, records, targets, folds=5) -> QuestionCalibration:
    n = len(records)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    parts = np.array_split(idx, min(folds, max(2, n // 10)))
    conf_cal, ok_cal, conf_raw, ok_raw = [], [], [], []
    for held in parts:
        held_set = set(held.tolist())
        train = [records[i] for i in range(n) if i not in held_set]
        T = fit_T(train)
        test = [records[i] for i in held]
        c, o = confidences(test, T)
        conf_cal += c.tolist()
        ok_cal += o.tolist()
        c, o = confidences(test, 1.0)
        conf_raw += c.tolist()
        ok_raw += o.tolist()
    conf_cal, ok_cal, conf_raw, ok_raw = map(np.asarray, (conf_cal, ok_cal, conf_raw, ok_raw))
    e_raw, rel_raw = ece(conf_raw, ok_raw)
    e_cal, rel_cal = ece(conf_cal, ok_cal)
    return QuestionCalibration(
        id=q.id, T=fit_T(records), n=n, accuracy=float(ok_cal.mean()),
        ece_raw=e_raw, ece_cal=e_cal, brier_raw=brier_score(conf_raw, ok_raw), brier_cal=brier_score(conf_cal, ok_cal),
        mean_conf_raw=float(conf_raw.mean()), mean_conf_cal=float(conf_cal.mean()),
        reliability_raw=rel_raw, reliability_cal=rel_cal, tiers=thresholds(conf_cal, ok_cal, targets),
    )


def calibrate(questions: list[Question], records: dict, targets=(("auto", 0.95), ("review", 0.80)), model="") -> Calibration:
    return Calibration({q.id: calibrate_question(q, records[q.id], list(targets)) for q in questions if records.get(q.id)}, model=model)


def report(cal: Calibration) -> str:
    lines = [f"calibration report  model={cal.model or '?'}", ""]
    lines.append(f"{'question':16} {'n':>5} {'acc':>6} {'ECE raw':>8} {'ECE cal':>8} {'Brier raw':>9} {'Brier cal':>9} {'conf raw':>8} {'conf cal':>8} {'T':>6}")
    for q in cal.questions.values():
        lines.append(f"{q.id:16} {q.n:5d} {q.accuracy:6.3f} {q.ece_raw:8.3f} {q.ece_cal:8.3f} {q.brier_raw:9.3f} {q.brier_cal:9.3f} {q.mean_conf_raw:8.3f} {q.mean_conf_cal:8.3f} {q.T:6.2f}")
    lines.append("")
    lines.append("decision policy (calibrated confidence cutoffs, held-out):")
    for q in cal.questions.values():
        for t in q.tiers:
            if t.min_confidence is None:
                lines.append(f"  {q.id:16} {t.name:7} target {t.target_precision:.2f}: unreachable on this data")
            else:
                lines.append(f"  {q.id:16} {t.name:7} target {t.target_precision:.2f}: act when confidence >= {t.min_confidence:.3f}  -> precision {t.precision:.3f}, covers {t.coverage:.0%} of decisions")
        lines.append(f"  {q.id:16} human   everything below")
    return "\n".join(lines)
