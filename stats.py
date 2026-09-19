"""Uncertainty for every reported number. Bootstrap over examples gives a 95% interval for one
run; several training seeds give the spread between runs. Reads the rec_*.json files that
`python3 -m brier calibrate` writes next to its cal_*.json."""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from brier.metrics import ece
from brier.scorer import softmax


def per_example(rec):
    """(confidence, correct) per example at T=1, the raw numbers the tables report."""
    conf, ok = [], []
    for scores, acc in rec:
        p = softmax(scores, 1.0)
        i = int(np.argmax(p))
        conf.append(max(p))
        ok.append(1.0 if i in acc else 0.0)
    return np.array(conf), np.array(ok)


def bootstrap(conf, ok, k=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ok)
    accs, eces = [], []
    for _ in range(k):
        idx = rng.integers(0, n, n)
        accs.append(ok[idx].mean())
        eces.append(ece(conf[idx].tolist(), ok[idx].tolist())[0])
    return (float(ok.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)),
            float(ece(conf.tolist(), ok.tolist())[0]), float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5)))


def load(path):
    d = json.load(open(path))
    return d["task"], d["tag"], {q: per_example(r) for q, r in d["records"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="rec_*.json files or directories")
    ap.add_argument("--group", default=r"-seed\d+|-eval\d+", help="regex removed from tags to pool seeds")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "rec_*.json"))) if os.path.isdir(p) else [p]
    rows = defaultdict(list)   # (task, question, group tag) -> list of per-run stats
    for f in files:
        task, tag, recs = load(f)
        g = re.sub(a.group, "", tag)
        for q, (conf, ok) in recs.items():
            rows[(task, q, g)].append(bootstrap(conf, ok))
    sep = " | " if a.md else "  "
    head = ["task", "question", "run", "n runs", "acc", "95% CI (one run)", "± across runs", "ECE", "95% CI", "± across runs"]
    print(("| " + " | ".join(head) + " |\n|" + "---|" * len(head)) if a.md else "  ".join(f"{h:>14}" for h in head))
    for (task, q, g), st in sorted(rows.items()):
        st = np.array(st)
        acc, lo, hi = st[:, 0].mean(), st[:, 1].mean(), st[:, 2].mean()
        e, elo, ehi = st[:, 3].mean(), st[:, 4].mean(), st[:, 5].mean()
        sd_acc = st[:, 0].std(ddof=1) if len(st) > 1 else float("nan")
        sd_ece = st[:, 3].std(ddof=1) if len(st) > 1 else float("nan")
        cells = [task, q, g, str(len(st)), f"{acc:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{sd_acc:.3f}", f"{e:.3f}", f"[{elo:.3f}, {ehi:.3f}]", f"{sd_ece:.3f}"]
        print(("| " + " | ".join(cells) + " |") if a.md else "  ".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
