"""Latency of the pointer head against the row scorer, by number of questions, same backbone."""
import argparse
import time

import torch

from brier import Scorer, ask
from brier.__main__ import load_scorer
from brier.pointer import PointerHead, ask_pointer
from brier.questions import Pick, Grade, Claim

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter", default=None, help="pointer adapter dir (pointer.pt); untrained head if omitted")
a = ap.parse_args()
sc = load_scorer(a.model, a.adapter) if a.adapter else Scorer(a.model)
head = sc.pointer or PointerHead(sc.model.config.hidden_size).to(sc.device)
sc.pointer = None
state = "Webhooks are firing twice for every event since around 3am UTC. I need this sorted today if at all possible. We're on the Team plan."
base_qs = [Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
           Grade("priority", "how fast this must be handled", ("low", "medium", "high", "urgent")),
           Claim("needs_human", "the customer is asking for a human agent")]
print(f"{'questions':>9} {'candidates':>10} {'rows p50 ms':>12} {'pointer p50 ms':>15} {'speedup':>8}")
for nq in (3, 10, 30):
    qs = []
    for i in range(nq):
        q = base_qs[i % 3]
        qs.append(type(q)(f"{q.id}_{i}", q.instructions, q.options if isinstance(q, Pick) else q.levels) if not isinstance(q, Claim) else Claim(f"{q.id}_{i}", q.instructions))
    nc = sum(len(q.labels) for q in qs)
    def timeit(fn, k=7):
        fn(); ts = []
        for _ in range(k):
            torch.cuda.synchronize() if sc.device == "cuda" else None
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize() if sc.device == "cuda" else None; ts.append((time.perf_counter() - t0) * 1000)
        return sorted(ts)[k // 2]
    r = timeit(lambda: ask(sc, state, qs)); p = timeit(lambda: ask_pointer(sc, head, state, qs))
    print(f"{nq:>9} {nc:>10} {r:>12.1f} {p:>15.1f} {r / p:>7.1f}x")
