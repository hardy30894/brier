"""Does compiling the transformer body with CUDA graphs remove the per-layer launch overhead?
Pads the prompt to a fixed bucket so shapes are static."""
import time

import torch

from brier import Scorer
from brier.pointer import decoder, encode
from brier.questions import Pick, Claim

sc = Scorer("Qwen/Qwen3-0.6B")
dec = decoder(sc.model)
state = "Webhooks are firing twice for every event since around 3am UTC. I need this sorted today if at all possible. We're on the Team plan."
qs = [Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")), Claim("needs_human", "the customer is asking for a human agent"), Claim("refund", "the customer wants money back")]
ids, qpos, opos = encode(sc, state, qs)
pad = sc.tok.pad_token_id


def bucket(ids, b=256):
    n = len(ids)
    L = ((n + b - 1) // b) * b
    x = torch.tensor([ids + [pad] * (L - n)], device=sc.device)
    m = torch.tensor([[1] * n + [0] * (L - n)], device=sc.device)
    return x, m


def bench(name, fn, k=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    print(f"{name:44} {sorted(ts)[k // 2]:7.2f} ms")


x, m = bucket(ids)
with torch.inference_mode():
    bench(f"eager, {len(ids)} tokens", lambda: dec(input_ids=torch.tensor([ids], device=sc.device)).last_hidden_state)
    bench(f"eager, padded to {x.shape[1]}", lambda: dec(input_ids=x, attention_mask=m).last_hidden_state)
    cdec = torch.compile(dec, mode="reduce-overhead", dynamic=False)
    bench(f"compiled + cuda graphs, padded to {x.shape[1]}", lambda: cdec(input_ids=x, attention_mask=m).last_hidden_state)
    cdec2 = torch.compile(dec, mode="max-autotune-no-cudagraphs", dynamic=False)
    bench(f"compiled, no graphs, padded to {x.shape[1]}", lambda: cdec2(input_ids=x, attention_mask=m).last_hidden_state)
