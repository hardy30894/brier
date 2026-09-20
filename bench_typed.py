"""The public typed-decision benchmark's own metrics from our evaluation records: accuracy
(argmax against the gold label), soft accuracy (expected agreement with the gold distribution),
multi-class Brier against the gold distribution, ECE with 15 bins, and score MAE for ordinal
questions, overall and per workflow."""
import argparse
import glob
import json
import os

import numpy as np

from brier.scorer import softmax


def metrics(path):
    d = json.load(open(path))
    acc, soft, brier, conf, ok, mae, per_wf = [], [], [], [], [], [], {}
    labels, recs, tgts, wfs, gold, index = d["labels"], d["records"], d.get("targets", {}), d.get("workflow", []), d.get("gold", []), d.get("index", {})
    n_ex = len(wfs) if wfs else None
    for qid, rows in recs.items():
        tg = tgts.get(qid) or [None] * len(rows)
        idx = index.get(qid) or list(range(len(rows)))
        for (scores, accs), t, i in zip(rows, tg, idx):
            p = np.array(softmax(scores, 1.0))
            g = np.array(t) if t else None
            j = int(p.argmax())
            oq = d.get("question_id", {}).get(qid, qid)
            gl = gold[i][oq]["label"] if gold and gold[i] and oq in gold[i] else None   # the benchmark's own hard label
            if gl is not None:
                gl = {"true": "yes", "false": "no"}.get(str(gl).lower(), str(gl))
                c = 1.0 if labels[qid][j] == gl else 0.0
            else:
                c = 1.0 if j in accs else 0.0
            acc.append(c); ok.append(c); conf.append(float(p.max()))
            if g is not None:
                soft.append(float((p * g).sum())); brier.append(float(((p - g) ** 2).sum()))
                if all(l.isdigit() for l in labels[qid]):
                    lv = np.arange(len(p))
                    mae.append(abs(float((p * lv).sum()) - float((g * lv).sum())))
            if wfs:
                per_wf.setdefault(wfs[i], []).append(c)
    conf, ok = np.array(conf), np.array(ok)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return {"n": len(acc), "accuracy": np.mean(acc), "soft_accuracy": np.mean(soft) if soft else None, "brier": np.mean(brier) if brier else None,
            "ece15": ece, "score_mae": np.mean(mae) if mae else None, "per_workflow": {k: np.mean(v) for k, v in per_wf.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "rec_typed_*.json"))) if os.path.isdir(p) else [p]
    if a.md:
        print("| model | decisions | accuracy | soft acc | Brier | ECE (15 bins) | score MAE | agent trace | customer service | invoice | security |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        m = metrics(f)
        tag = os.path.basename(f)[len("rec_typed_"):-5]
        wf = m["per_workflow"]
        fmt = lambda x: "–" if x is None else f"{x:.3f}"
        if a.md:
            print(f"| {tag} | {m['n']} | **{m['accuracy']:.3f}** | {fmt(m['soft_accuracy'])} | {fmt(m['brier'])} | {m['ece15']:.3f} | {fmt(m['score_mae'])} | " + " | ".join(fmt(wf.get(w)) for w in ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")) + " |")
        else:
            print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "per_workflow"}, {k[:8]: round(v, 3) for k, v in wf.items()})
