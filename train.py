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
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from brier import Scorer
from brier.head import TypedHead
from brier.pointer import PointerHead, pointer_scores
from brier.calibrate import acceptable, target
from brier.questions import build_prompt, groups_for
from tasks import examples

CANDIDATE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"]


def lora_targets(model):
    """Projection names present in this architecture (Qwen/Llama use q,k,v; Phi fuses qkv and gate_up)."""
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    return [t for t in CANDIDATE_TARGETS if t in names]


def pointer_loss(scorer, head, exs, teacher=None):
    """One row per example: every question's options scored by the pointer head. Cross-entropy
    against the label (or soft target), plus KL to a teacher's distributions when given."""
    loss, n = 0.0, 0
    for ex in exs:
        scores = pointer_scores(scorer, head, ex["state"], ex["questions"], grad=True)
        tdists = None
        if teacher is not None:
            with torch.no_grad():
                ta = teacher(ex["state"], ex["questions"])
            tdists = {q.id: [ta[q.id].probabilities[l] for l in q.labels] for q in ex["questions"]}
        for q, s in zip(ex["questions"], scores):
            logp = torch.log_softmax(s, dim=0)
            if q.id in ex["labels"]:
                t = target(q, ex["labels"][q.id])
                if t is not None:
                    loss = loss - (torch.tensor(t, device=logp.device) * logp).sum()
                else:
                    acc = sorted(acceptable(q, ex["labels"][q.id]))
                    loss = loss - torch.logsumexp(logp[acc], dim=0)
                n += 1
            if tdists is not None:
                tp = torch.tensor(tdists[q.id], device=logp.device).clamp_min(1e-6)
                loss = loss + (tp * (tp.log() - logp)).sum()
                n += 1
    return loss / max(n, 1)


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


def ddp_setup():
    """torchrun sets these; a plain `python3 train.py` runs on one device."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, world, f"cuda:{os.environ['LOCAL_RANK']}"
    return 0, 1, None


def main():
    rank, world, device = ddp_setup()
    log = (lambda *a: print(*a, file=sys.stderr)) if rank == 0 else (lambda *a: None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tasks", default="tickets,gridworld,toolcall,passages")
    ap.add_argument("--split", default="all", help="template split: all | train (eval templates held out)")
    ap.add_argument("--uncertain", type=float, default=0.0, help="fraction of examples generated as genuinely ambiguous with soft labels")
    ap.add_argument("--examples", "--n", dest="n", type=int, default=600, help="training examples per task")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="optimizer step every N micro-batches")
    ap.add_argument("--batch", type=int, default=1, help="examples per forward pass")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--head", action="store_true", help="train a typed head over hidden states jointly with LoRA")
    ap.add_argument("--pointer", action="store_true", help="train the pointer head: one row per example, options scored from the prompt's hidden states")
    ap.add_argument("--distill", default=None, help="adapter dir of a row-scorer teacher; adds KL to its distributions (pointer only)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--ordinal", default="quadratic", choices=("quadratic", "cumulative"), help="pointer head's Grade term")
    ap.add_argument("--grade-numbers", action="store_true", help="pointer prompt writes Grade levels with their rank")
    ap.add_argument("--grade-boost", type=int, default=1, help="repeat examples that carry a Grade question this many times")
    ap.add_argument("--joint", dest="joint", action="store_true", default=True, help="with --pointer: also train the row-scorer objective, so the adapter serves both paths (default)")
    ap.add_argument("--pointer-only", dest="joint", action="store_false", help="train the pointer head alone; the row scorer under this adapter is not usable")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if world > 1:
        log(f"distributed: {world} processes, each step sees {world} × {a.accum} × {a.batch} examples")
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=lora_targets(base), task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    if rank == 0:
        model.print_trainable_parameters()
    scorer = Scorer(a.model, model=model, tok=tok, device=device)
    scorer.model = scorer.model.to(scorer.device)
    peft_model = scorer.model
    if world > 1:
        scorer.model = torch.nn.parallel.DistributedDataParallel(peft_model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)
    if a.head:
        scorer.head = TypedHead(base.config.hidden_size).to(scorer.device)
    pointer, teacher = None, None
    if a.pointer:
        pointer = PointerHead(base.config.hidden_size, ordinal=a.ordinal).to(scorer.device)
        scorer.grade_numbers = a.grade_numbers
        if a.distill:
            from brier.__main__ import load_scorer
            from brier.questions import ask as _ask
            tsc = load_scorer(a.model, a.distill)
            teacher = lambda st, qs: _ask(tsc, st, qs)

    data = []
    for t in a.tasks.split(","):
        data += examples(t, a.n, seed=a.seed, split=a.split, uncertain=a.uncertain)
    if a.grade_boost > 1:
        from brier.questions import Grade as _Grade
        data += [ex for ex in data if any(isinstance(q, _Grade) for q in ex["questions"])] * (a.grade_boost - 1)
    random.shuffle(data)
    log(f"{len(data)} training examples from {a.tasks}")
    if world > 1:
        data = data[rank::world]

    groups = [{"params": [p for p in scorer.model.parameters() if p.requires_grad], "lr": a.lr}]
    if a.head:
        groups.append({"params": list(scorer.head.parameters()), "lr": a.head_lr})
    if pointer is not None:
        groups.append({"params": list(pointer.parameters()), "lr": a.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / a.steps))
    scorer.model.train()
    t0, running, i = time.perf_counter(), 0.0, 0
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            exs = [data[(i + j) % len(data)] for j in range(a.batch)]
            i += a.batch
            if pointer is not None:
                loss = pointer_loss(scorer, pointer, exs, teacher)
                if a.joint:
                    loss = loss + batch_loss(scorer, exs)   # keeps the row scorer, which shares these weights, trained too
            else:
                loss = batch_loss(scorer, exs)
            loss = loss / a.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            log(f"step {step + 1}/{a.steps} loss {running / 50:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.perf_counter() - t0:.0f}s)")
            running = 0.0
    name = f"{a.model.split('/')[-1]}-brier-r{a.rank}-s{a.steps}" + (f"-u{a.uncertain:g}" if a.uncertain else "") + (f"-{a.split}" if a.split != "all" else "") + ("-head" if a.head else "") + ("-pointer" if a.pointer else "") + ("-distill" if a.distill else "") + ("-cum" if a.ordinal == "cumulative" else "") + ("-num" if a.grade_numbers else "") + (f"-gb{a.grade_boost}" if a.grade_boost > 1 else "") + ("-joint" if a.joint and a.pointer else "")
    path = os.path.join(a.out, name)
    if rank == 0:
        peft_model.save_pretrained(path)
        if a.head:
            torch.save(scorer.head.state_dict(), os.path.join(path, "head.pt"))
        if pointer is not None:
            torch.save(pointer.state_dict(), os.path.join(path, "pointer.pt"))
            json.dump({"ordinal": a.ordinal, "grade_numbers": a.grade_numbers}, open(os.path.join(path, "pointer.json"), "w"))
        json.dump({**vars(a), "world_size": world, "train_seconds": round(time.perf_counter() - t0)}, open(os.path.join(path, "train_args.json"), "w"), indent=1)
        print(f"saved {path}  ({world} GPU{'s' if world > 1 else ''}, {time.perf_counter() - t0:.0f}s)")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
