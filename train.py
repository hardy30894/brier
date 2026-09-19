"""Milestone 2: fine-tune the backbone with LoRA so that the per-question softmax over candidates is
trained directly with cross-entropy, a proper scoring rule. Same scoring path as inference."""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from brier import Scorer
from brier.head import TypedHead
from brier.calibrate import acceptable, target
from brier.questions import build_prompt, groups_for
from tasks import examples

CANDIDATE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"]


def lora_targets(model):
    """Projection names present in this architecture (Qwen/Llama use q,k,v; Phi fuses qkv and gate_up)."""
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    return [t for t in CANDIDATE_TARGETS if t in names]


def batch_loss(scorer, exs):
    """One forward pass over the candidate rows of several examples; mean per-question NLL."""
    seqs, spans, layout = [], [], []
    for ex in exs:
        prompt = build_prompt(scorer, ex["state"], ex["questions"])
        groups = groups_for(ex["questions"])
        s, sp = scorer.build_rows(prompt, groups)
        seqs += s
        spans += sp
        layout.append((ex, groups))
    input_ids, attn = scorer.batch(seqs)
    cand = scorer.scores_from_rows(input_ids, attn, spans)
    loss, n, k = 0.0, 0, 0
    for ex, groups in layout:
        for q, (_, labels) in zip(ex["questions"], groups):
            s = cand[k : k + len(labels)]
            k += len(labels)
            if q.id not in ex["labels"]:
                continue
            logp = torch.log_softmax(s, dim=0)
            t = target(q, ex["labels"][q.id])
            if t is not None:
                loss = loss - (torch.tensor(t, device=logp.device) * logp).sum()
            else:
                acc = sorted(acceptable(q, ex["labels"][q.id]))
                loss = loss - torch.logsumexp(logp[acc], dim=0)
            n += 1
    return loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tasks", default="tickets,gridworld,toolcall,passages")
    ap.add_argument("--split", default="all", help="template split: all | train (eval templates held out)")
    ap.add_argument("--uncertain", type=float, default=0.0, help="fraction of examples generated as genuinely ambiguous with soft labels")
    ap.add_argument("--n", type=int, default=600, help="training examples per task")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="optimizer step every N micro-batches")
    ap.add_argument("--batch", type=int, default=1, help="examples per forward pass")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--head", action="store_true", help="train a typed head over hidden states jointly with LoRA")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=lora_targets(base), task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    scorer = Scorer(a.model, model=model, tok=tok)
    scorer.model = scorer.model.to(scorer.device)
    if a.head:
        scorer.head = TypedHead(base.config.hidden_size).to(scorer.device)

    data = []
    for t in a.tasks.split(","):
        data += examples(t, a.n, seed=a.seed, split=a.split, uncertain=a.uncertain)
    random.shuffle(data)
    print(f"{len(data)} training examples from {a.tasks}", file=sys.stderr)

    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}]
    if a.head:
        groups.append({"params": list(scorer.head.parameters()), "lr": a.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / a.steps))
    model.train()
    t0, running, i = time.perf_counter(), 0.0, 0
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            exs = [data[(i + j) % len(data)] for j in range(a.batch)]
            i += a.batch
            loss = batch_loss(scorer, exs) / a.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{a.steps} loss {running / 50:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
            running = 0.0
    name = f"{a.model.split('/')[-1]}-brier-r{a.rank}-s{a.steps}" + (f"-u{a.uncertain:g}" if a.uncertain else "") + (f"-{a.split}" if a.split != "all" else "") + ("-head" if a.head else "")
    path = os.path.join(a.out, name)
    model.save_pretrained(path)
    if a.head:
        torch.save(scorer.head.state_dict(), os.path.join(path, "head.pt"))
    json.dump(vars(a), open(os.path.join(path, "train_args.json"), "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
