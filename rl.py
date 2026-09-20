"""Outcome-reward training on gridworld: no labels. The model states a distribution over moves,
one move is sampled and executed, and the environment reports whether it brought the player
closer to the exit. The loss has two parts: a policy-gradient term on the outcome, which
drives accuracy, and the Brier score of the stated probability against the outcome, which
drives honesty and is the reinforcement form of the calibrated-decisions objective."""
import argparse
import contextlib
import json
import os
import random
import sys
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from brier import Scorer
from brier.questions import build_prompt, groups_for
from tasks import gridworld
from tasks.gridworld import parse_state, outcome, step, render_state

CANDIDATE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"]


def lora_targets(model):
    """Projection names present in this architecture (Qwen/Llama use q,k,v; Phi fuses qkv and gate_up)."""
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    return [t for t in CANDIDATE_TARGETS if t in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--init", default=None, help="start from a supervised adapter instead of the base model")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8, help="states per step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the Brier calibration term")
    ap.add_argument("--episode", type=int, default=1, help="steps per episode; >1 uses the delayed episode outcome (reached the exit) as the reward for every step")
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--task", default="gridworld", help="gridworld | tickets | toolcall | typed: text tasks use the environments in tasks/outcomes.py")
    ap.add_argument("--uncertain", type=float, default=0.3, help="text tasks: fraction of ambiguous states, whose outcomes are drawn from the soft posterior")
    ap.add_argument("--pointer", action="store_true", help="act through the pointer head (one row per state); required for encoder backbones")
    ap.add_argument("--full", action="store_true", help="with --pointer: fine-tune every backbone weight instead of a LoRA adapter (encoders)")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    from brier.scorer import is_encoder
    encoder = is_encoder(a.model)
    if encoder:
        a.pointer = True
    tok = AutoTokenizer.from_pretrained(a.model)
    if encoder:
        from transformers import AutoModel
        base = AutoModel.from_pretrained(a.model, dtype=torch.float32 if a.full else torch.bfloat16, **({"reference_compile": False} if "modernbert" in a.model.lower() else {}))
    else:
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except (ValueError, TypeError):
        pass
    head = None
    if a.full:
        model = base
        if a.init and os.path.exists(os.path.join(a.init, "backbone.pt")):
            model.load_state_dict(torch.load(os.path.join(a.init, "backbone.pt"), map_location="cpu"))
            print(f"warm start from {a.init}")
        for prm in model.parameters():
            prm.requires_grad_(True)
    else:
        base.enable_input_require_grads()
        if a.init:
            model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=lora_targets(base), task_type="FEATURE_EXTRACTION" if encoder else "CAUSAL_LM"))
        model.print_trainable_parameters()
    scorer = Scorer(a.model, model=model, tok=tok)
    scorer.model = scorer.model.to(scorer.device)
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}]
    if a.pointer:
        from brier.pointer import PointerHead
        head = PointerHead(base.config.hidden_size).to(scorer.device)
        if a.init and os.path.exists(os.path.join(a.init, "pointer.pt")):
            head.load_state_dict(torch.load(os.path.join(a.init, "pointer.pt"), map_location=scorer.device), strict=False)
        groups.append({"params": list(head.parameters()), "lr": a.head_lr})
    opt = torch.optim.AdamW(groups)
    q = gridworld.QUESTIONS[0]
    baseline, t0 = 0.5, time.perf_counter()
    stats = {"reward": 0.0, "brier": 0.0, "conf": 0.0, "n": 0}
    model.train()
    if a.task != "gridworld":
        train_text(a, scorer, model, opt, stats, t0, head)
        save(a, model, head)
        return
    for it in range(a.steps):
        exs = gridworld.generate_q(a.batch, seed=a.seed * 100000 + it)
        opt.zero_grad(set_to_none=True)
        loss, count = 0.0, 0
        for ex in exs:
            grid, player, goal, monster = parse_state(ex)
            chosen, dead = [], False
            for t in range(a.episode):
                state = ex["state"] if t == 0 else render_state(grid, player, goal, monster)
                seqs, spans = scorer.build_rows(build_prompt(scorer, state, [q]), groups_for([q]))
                input_ids, attn = scorer.batch(seqs)
                logp = torch.log_softmax(scorer.scores_from_rows(input_ids, attn, spans), dim=0)
                i = torch.multinomial(logp.exp().detach(), 1).item()
                y_step = outcome(grid, player, goal, monster, q.labels[i])
                chosen.append((logp[i], y_step))
                player, status = step(grid, player, monster, q.labels[i])
                if status == "exit":
                    break
                if status == "dead":
                    dead = True
                    break
            reached = 1.0 if (status == "exit" and not dead) else 0.0
            for lp, y_step in chosen:
                y = reached if a.episode > 1 else y_step
                p = lp.exp()
                loss = loss - (y - baseline) * lp + a.lam * (p - y) ** 2
                baseline = 0.98 * baseline + 0.02 * y
                stats["reward"] += y
                stats["brier"] += ((p - y) ** 2).item()
                stats["conf"] += p.item()
                stats["n"] += 1
                count += 1
        (loss / max(count, 1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (it + 1) % 25 == 0:
            n = max(1, stats["n"])
            print(f"step {it + 1}/{a.steps} reward {stats['reward'] / n:.3f} brier {stats['brier'] / n:.3f} conf {stats['conf'] / n:.3f} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
            stats = {k: 0.0 for k in stats}
    save(a, model)


def save(a, model, head=None):
    name = f"{a.model.split('/')[-1]}-brier-rl-{a.task}-s{a.steps}" + (f"-ep{a.episode}" if a.episode > 1 else "") + ("-from-sup" if a.init else "") + ("-pointer" if a.pointer else "") + ("-full" if a.full else "")
    path = os.path.join(a.out, name)
    os.makedirs(path, exist_ok=True)
    if a.full:
        torch.save(model.state_dict(), os.path.join(path, "backbone.pt"))
    else:
        model.save_pretrained(path)
    if head is not None:
        torch.save(head.state_dict(), os.path.join(path, "pointer.pt"))
        json.dump({"ordinal": "quadratic", "grade_numbers": False}, open(os.path.join(path, "pointer.json"), "w"))
    json.dump(vars(a), open(os.path.join(path, "train_args.json"), "w"), indent=1)
    print(f"saved {path}")


def train_text(a, scorer, model, opt, stats, t0, head=None):
    """Bandit feedback on text decisions: one answer sampled per question, the environment
    returns a reward for that answer only, and the loss is the policy-gradient term against a
    per-question running baseline plus the Brier score of the stated probability. With a
    pointer head the whole state is one row; otherwise the row scorer's candidate rows."""
    from tasks import examples
    from tasks.outcomes import ENVS
    from brier.pointer import pointer_scores
    env = ENVS[a.task]
    rng = random.Random(a.seed)
    baselines = {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.full and torch.cuda.is_available()) else contextlib.nullcontext()
    for it in range(a.steps):
        exs = examples(a.task, a.batch, seed=a.seed * 100000 + it, split="train", uncertain=a.uncertain)
        opt.zero_grad(set_to_none=True)
        loss, count = 0.0, 0
        for ex in exs:
            qs = ex["questions"]
            if head is not None:
                with autocast:
                    per_q = pointer_scores(scorer, head, ex["state"], qs, grad=True)
            else:
                seqs, spans = scorer.build_rows(build_prompt(scorer, ex["state"], qs), groups_for(qs))
                input_ids, attn = scorer.batch(seqs)
                cand = scorer.scores_from_rows(input_ids, attn, spans)
            answers, picks, k = {}, [], 0
            for qi, q in enumerate(qs):
                if head is not None:
                    s_q = per_q[qi]
                else:
                    s_q = cand[k : k + len(q.labels)]
                    k += len(q.labels)
                logp = torch.log_softmax(s_q.float(), dim=0)
                i = torch.multinomial(logp.exp().detach(), 1).item()
                answers[q.id] = q.labels[i]
                picks.append((q.id, logp[i]))
            rewards = env(ex, answers, rng)
            for qid, lp in picks:
                if qid not in rewards:
                    continue
                y = rewards[qid]
                b = baselines.get(qid, 0.5)
                p = lp.exp()
                loss = loss - (y - b) * lp + a.lam * (p - y) ** 2
                baselines[qid] = 0.98 * b + 0.02 * y
                stats["reward"] += y
                stats["brier"] += ((p - y) ** 2).item()
                stats["conf"] += p.item()
                stats["n"] += 1
                count += 1
        (loss / max(count, 1)).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head is not None else []), 1.0)
        opt.step()
        if (it + 1) % 25 == 0:
            n = max(1, stats["n"])
            print(f"step {it + 1}/{a.steps} reward {stats['reward'] / n:.3f} brier {stats['brier'] / n:.3f} conf {stats['conf'] / n:.3f} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
            for key in stats:
                stats[key] = 0.0


if __name__ == "__main__":
    main()
