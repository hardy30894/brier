"""Shared-prefix cached scoring versus full rows: agreement of scores and latency, per backbone."""
import argparse
import sys
import time

from brier import Scorer, ask
from tasks import examples

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
exs = examples("tickets", a.n, seed=9)
print(f"{'model':36} {'p50 full':>9} {'p50 cached':>11} {'speedup':>8} {'max |Δprob|':>12} {'answers agree':>14}")
for m in a.models.split(","):
    sc = Scorer(m)
    lat = {}
    scores = {}
    for mode in (False, True):
        sc.shared_prefix = mode
        ask(sc, exs[0]["state"], exs[0]["questions"])
        ts, out = [], []
        for e in exs:
            t0 = time.perf_counter()
            ans = ask(sc, e["state"], e["questions"])
            ts.append((time.perf_counter() - t0) * 1000)
            out.append({q.id: (ans[q.id].probabilities, ans[q.id].value) for q in e["questions"]})
        ts.sort()
        lat[mode] = ts[len(ts) // 2]
        scores[mode] = out
    dmax, agree, tot = 0.0, 0, 0
    for f, c in zip(scores[False], scores[True]):
        for qid in f:
            for k in f[qid][0]:
                dmax = max(dmax, abs(f[qid][0][k] - c[qid][0][k]))
            fv, cv = f[qid][1], c[qid][1]
            agree += (abs(fv - cv) < 0.02) if isinstance(fv, float) else (fv == cv)
            tot += 1
    print(f"{m:36} {lat[False]:9.1f} {lat[True]:11.1f} {lat[False] / lat[True]:8.1f}x {dmax:12.3f} {agree / tot:14.3f}")
    del sc
