"""Judge a model on a labelled dataset with no calibration: raw accuracy, per-class recall,
confusion, majority baseline, and raw calibration error, per question.

    python3 eval_dataset.py --data ml_rows.json --model microsoft/Phi-4-mini-instruct [--adapter …] [--n 400]
"""
import argparse
import collections
import json
import sys
import time

from brier import ask
from brier.__main__ import load_scorer
from brier.metrics import ece
from brier.questions import Choice, Score, Claim

VARIANTS = {
    "default": (("core", "advanced", "frontier"), {"core": "core", "advanced": "advanced", "frontier": "frontier"}),
    "named": (("small", "mid-sized", "frontier"), {"core": "small", "advanced": "mid-sized", "frontier": "frontier"}),
    "difficulty": (("easy", "moderate", "hard"), {"core": "easy", "advanced": "moderate", "frontier": "hard"}),
}
VARIANT = "default"


def questions():
    labels, _ = VARIANTS[VARIANT]
    instr = {"default": "the smallest model tier that can fix this issue, judged from the report alone",
             "named": "the smallest model that can fix this issue, judged from the report alone",
             "difficulty": "how hard this issue is to fix, judged from the report alone"}[VARIANT]
    return [
        Score("tier", instr, labels),
        Claim("core_solves", "a small, cheap model could fix this issue"),
        Claim("advanced_solves", "a mid-sized model could fix this issue"),
    ]


def LABEL_OF(qid, r):
    if qid == "tier":
        return VARIANTS[VARIANT][1][r["cls"]]
    return "yes" if (r["core_res"] if qid == "core_solves" else r["adv_res"]) else "no"


def state_of(r, max_chars):
    t = r["text"]
    if len(t) > max_chars:
        t = t[:max_chars] + " …"
    return {"repo": r["repo"], "issue": t, "failing_tests": r["n_f2p"], "passing_tests": r["n_p2p"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="default", choices=list(VARIANTS))
    ap.add_argument("--dump", default=None, help="write per-row probabilities as JSONL here")
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    QUESTIONS = questions()
    rows = json.load(open(a.data))
    if a.n:
        import random
        random.Random(7).shuffle(rows)
        rows = rows[: a.n]
    sc = load_scorer(a.model, a.adapter)
    tag = (a.adapter.rstrip("/").split("/")[-1] if a.adapter else a.model.split("/")[-1])
    preds = {q.id: [] for q in QUESTIONS}
    t0 = time.perf_counter()
    lats = []
    dump = open(a.dump, "w") if a.dump else None
    for i, r in enumerate(rows):
        ans = ask(sc, state_of(r, a.max_chars), QUESTIONS)
        lats.append(ans.latency_ms)
        if dump:
            dump.write(json.dumps({"id": r["id"], "cls": r["cls"], "core_res": r["core_res"], "adv_res": r["adv_res"], "n_f2p": r["n_f2p"], "probs": {q.id: ans[q.id].probabilities for q in QUESTIONS}}) + "\n")
        for q in QUESTIONS:
            an = ans[q.id]
            top = max(an.probabilities, key=an.probabilities.get)
            preds[q.id].append((top, LABEL_OF(q.id, r), an.probabilities[top], an.probabilities))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    report = {"model": tag, "variant": VARIANT, "n": len(rows), "latency_p50_ms": sorted(lats)[len(lats) // 2], "questions": {}}
    for q in QUESTIONS:
        p = preds[q.id]
        labels = list(q.labels)
        acc = sum(t == y for t, y, c, _ in p) / len(p)
        maj = collections.Counter(y for _, y, _, _ in p).most_common(1)[0]
        conf = [[sum(1 for t, y, _, _ in p if y == yy and t == tt) for tt in labels] for yy in labels]
        recall = {yy: (conf[i][i] / max(1, sum(conf[i]))) for i, yy in enumerate(labels)}
        precision = {tt: (conf[j][j] / max(1, sum(conf[i][j] for i in range(len(labels))))) for j, tt in enumerate(labels)}
        f1 = {k: (2 * precision[k] * recall[k] / max(1e-9, precision[k] + recall[k])) for k in labels}
        e, rel = ece([c for _, _, c, _ in p], [float(t == y) for t, y, _, _ in p])
        pred_dist = collections.Counter(t for t, _, _, _ in p)
        report["questions"][q.id] = {"accuracy": acc, "majority_baseline": {"label": maj[0], "accuracy": maj[1] / len(p)}, "macro_f1": sum(f1.values()) / len(f1),
                                     "recall": recall, "precision": precision, "confusion_rows_true_cols_pred": {"labels": labels, "matrix": conf},
                                     "predicted_distribution": dict(pred_dist), "ece_raw": e, "mean_confidence": sum(c for _, _, c, _ in p) / len(p), "reliability": rel}
    print(json.dumps({k: v for k, v in report.items() if k != "questions"}))
    for qid, qr in report["questions"].items():
        print(f"\n== {qid}  acc {qr['accuracy']:.3f}  (majority '{qr['majority_baseline']['label']}' = {qr['majority_baseline']['accuracy']:.3f})  macro-F1 {qr['macro_f1']:.3f}  ECE raw {qr['ece_raw']:.3f}  mean conf {qr['mean_confidence']:.3f}")
        print("   recall   ", {k: round(v, 3) for k, v in qr["recall"].items()})
        print("   precision", {k: round(v, 3) for k, v in qr["precision"].items()})
        print("   predicted", qr["predicted_distribution"])
        print("   confusion (rows=true, cols=pred)", qr["confusion_rows_true_cols_pred"]["labels"], qr["confusion_rows_true_cols_pred"]["matrix"])
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
