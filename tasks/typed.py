"""A public typed-decision benchmark: four workflows (agent-trace observability, customer
service, invoice processing, security incidents), 300 training and 100 test cases each, five
typed questions per case, with soft labels from a teacher model. Loaded from the Hub as Pick,
Grade and Claim questions; the gold distributions become soft labels, so training sees the
same targets any other method on this benchmark sees, and evaluation scores the argmax."""
import json

from brier.questions import Pick, Grade, Claim

WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def _questions(qdefs):
    qs = []
    for qid, q in qdefs.items():
        t, ins, crit = q["type"], q["instructions"], q.get("criteria")
        if t == "choice":
            qs.append(Pick(qid, ins, {k: (v or "") for k, v in crit.items()}))
        elif t == "score":
            levels = tuple(str(i) for i in range(len(crit)))
            scale = "; ".join(f"{i}: {c}" for i, c in enumerate(crit))
            qs.append(Grade(qid, f"{ins} (levels: {scale})", levels))
        else:
            c = crit or {}
            qs.append(Claim(qid, ins, true=c.get("true") or "yes, the statement holds", false=c.get("false") or "no, the statement does not hold"))
    return qs


def _labels(gold, qs):
    out = {}
    for q in qs:
        g = gold[q.id]
        probs = g.get("probabilities") or {}
        if isinstance(q, Claim):
            out[q.id] = {"yes": float(probs.get("true", g.get("noul", 0.5))), "no": float(probs.get("false", 1 - g.get("noul", 0.5)))}
        else:
            out[q.id] = {str(k): float(v) for k, v in probs.items()}
    return out


def generate(n, seed=0, split="all", uncertain=0.0, workflows=WORKFLOWS):
    """`split` train -> the 1,200 training cases; eval -> the 400 test cases. `n` caps the
    total, spread evenly over the workflows, in file order (the benchmark is already fixed)."""
    from datasets import load_dataset
    part = "train" if split == "train" else "test"
    per = max(1, n // len(workflows))
    out = []
    for wf in workflows:
        ds = load_dataset("LocalLLaMA/typed-decisions", wf, split=part)
        for r in list(ds)[:per]:
            qdefs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
            gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
            state = json.loads(r["state"]) if isinstance(r["state"], str) and r["state"].lstrip().startswith("{") else r["state"]
            qs = _questions(qdefs)
            out.append({"state": state, "questions": qs, "labels": _labels(gold, qs), "workflow": wf,
                        "gold": {q.id: gold[q.id] for q in qs}})
    return out


def _one(wf):
    return lambda n, seed=0, split="all", uncertain=0.0: generate(n, seed, split, uncertain, workflows=(wf,))


TYPED = {"typed": generate, **{f"typed_{wf}": _one(wf) for wf in WORKFLOWS}}
