"""Where the pointer path's time goes, stage by stage, per backbone."""
import argparse
import time

import torch

from brier import Scorer
from brier.pointer import PointerHead, render, char_to_token, decoder
from brier.questions import Pick, Grade, Claim

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
a = ap.parse_args()
sc = Scorer(a.model)
head = PointerHead(sc.model.config.hidden_size).to(sc.device)
state = "Webhooks are firing twice for every event since around 3am UTC. I need this sorted today if at all possible. We're on the Team plan."
qs = [Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
      Claim("needs_human", "the customer is asking for a human agent"), Claim("refund", "the customer wants money back")]


def stage(name, fn, k=20):
    fn()
    if sc.device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); r = fn()
        if sc.device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"{name:34} {sorted(ts)[k // 2]:7.2f} ms")
    return r


lay = stage("render prompt text", lambda: render(state, qs))
full = stage("chat template", lambda: sc.chat_prefix(lay.text))
enc = stage("tokenize with offsets", lambda: sc.tok(full, return_offsets_mapping=True, add_special_tokens=False))
enc_plain = stage("tokenize without offsets", lambda: sc.tok(full, add_special_tokens=False))
base = full.find(lay.text)
offs = enc["offset_mapping"]
stage("char -> token positions", lambda: ([char_to_token(offs, base + c) for c in lay.query_char], [[char_to_token(offs, base + c) for c in row] for row in lay.option_chars]))
x = torch.tensor([enc["input_ids"]], device=sc.device)
dec = decoder(sc.model)
with torch.no_grad():
    h = stage(f"forward, {x.shape[1]} tokens, body only", lambda: dec(input_ids=x).last_hidden_state[0])
    stage("forward with LM head (row scorer)", lambda: sc.model(input_ids=x).logits)
    qpos = [char_to_token(offs, base + c) for c in lay.query_char]; opos = [[char_to_token(offs, base + c) for c in row] for row in lay.option_chars]
    stage("head scoring, 3 questions", lambda: [head(h[qp], h[torch.tensor(op, device=sc.device)]) for qp, op in zip(qpos, opos)])
    stage("scores .tolist()", lambda: [head(h[qp], h[torch.tensor(op, device=sc.device)]).tolist() for qp, op in zip(qpos, opos)])
from brier.pointer import ask_pointer
stage("end to end ask_pointer", lambda: ask_pointer(sc, head, state, qs))
