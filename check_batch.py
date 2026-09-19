"""Batched pointer scores must equal single-request scores, and the Batcher must return the
same answers as ask(). Run with a pointer adapter: python3 check_batch.py --adapter adapters/<dir>"""
import argparse
import threading

from brier.__main__ import load_scorer
from brier.questions import Pick, Grade, Claim, ask
from brier.pointer import pointer_scores, pointer_scores_batch, Batcher, warm

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--capture", action="store_true")
ap.add_argument("--fp32", action="store_true", help="run the model in fp32, where batched and single must agree to ~1e-4")
a = ap.parse_args()
sc = load_scorer(a.model, a.adapter)
if a.fp32:
    sc.model = sc.model.float(); sc.pointer = sc.pointer.float()
tol = 1e-3 if a.fp32 else 0.5
sc.capture = a.capture
qs = [Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
      Claim("needs_human", "the customer is asking for a human agent"),
      Grade("priority", "how fast this must be handled", ("low", "medium", "high", "urgent"))]
states = ["Webhooks are firing twice for every event since 3am UTC. Need this today.",
          "Please cancel my subscription and refund last month, I never used it.",
          "Can I talk to a real person? The bot keeps looping. Also add dark mode please, it would be nice for late nights, and the export is broken."]
reqs = [(s, qs[:2]) for s in states] + [(states[0], [qs[1]])]
sc.capture = False
single = [pointer_scores(sc, sc.pointer, s, q) for s, q in reqs]
sc.capture = a.capture
if a.capture:
    print("captured", warm(sc, max_batch=8), "graphs")
batched = pointer_scores_batch(sc, sc.pointer, reqs)
worst = max((x - y).abs().max().item() for ss, bs in zip(single, batched) for x, y in zip(ss, bs))
print(f"batched{' captured' if a.capture else ''} vs single eager{' fp32' if a.fp32 else ' bf16'}: max |score diff| = {worst:.2e}")
b = Batcher(sc, max_batch=8, wait_ms=20)
res = [None] * len(states)
def go(i):
    res[i] = b.ask(states[i], qs)
ths = [threading.Thread(target=go, args=(i,)) for i in range(len(states))]
[t.start() for t in ths]; [t.join() for t in ths]
sc.capture = False
ref = [ask(sc, s, qs) for s in states]
w2 = max(abs(r[q.id].probabilities[l] - f[q.id].probabilities[l]) for r, f in zip(res, ref) for q in qs for l in q.labels)
print(f"batcher vs ask: max |prob diff| = {w2:.2e}  batches={b.batches} served={b.served}")
assert worst < tol and w2 < tol / 10, "batched path disagrees with the single path"
print("OK")
