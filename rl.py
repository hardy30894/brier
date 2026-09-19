"""Outcome-reward training on gridworld: no labels. The model states a distribution over moves,
one move is sampled and executed, and the environment reports whether it brought the player
closer to the exit. The loss has two parts: a policy-gradient term on the outcome, which
drives accuracy, and the Brier score of the stated probability against the outcome, which
drives honesty and is the reinforcement form of the calibrated-decisions objective."""
import argparse
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
from tasks.gridworld import parse_state, outcome

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--init", default=None, help="start from a supervised adapter instead of the base model")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8, help="states per step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the Brier calibration term")
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--out", default="adapters")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    if a.init:
        model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    scorer = Scorer(a.model, model=model, tok=tok)
    scorer.model = scorer.model.to(scorer.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    q = gridworld.QUESTIONS[0]
    baseline, t0 = 0.5, time.perf_counter()
    stats = {"reward": 0.0, "brier": 0.0, "conf": 0.0}
    model.train()
    for step in range(a.steps):
        exs = gridworld.generate_q(a.batch, seed=a.seed * 100000 + step)
        opt.zero_grad(set_to_none=True)
        seqs, spans = [], []
        for ex in exs:
            s, sp = scorer.build_rows(build_prompt(scorer, ex["state"], [q]), groups_for([q]))
            seqs += s
            spans += sp
        input_ids, attn = scorer.batch(seqs)
        cand = scorer.candidate_logprobs(scorer.model(input_ids=input_ids, attention_mask=attn).logits, input_ids, spans)
        loss, k = 0.0, 0
        for ex in exs:
            logp = torch.log_softmax(cand[k : k + 4], dim=0)
            k += 4
            p = logp.exp()
            i = torch.multinomial(p.detach(), 1).item()
            grid, player, goal, monster = parse_state(ex)
            y = outcome(grid, player, goal, monster, q.labels[i])
            policy = -(y - baseline) * logp[i]
            brier = (p[i] - y) ** 2
            loss = loss + policy + a.lam * brier
            baseline = 0.98 * baseline + 0.02 * y
            stats["reward"] += y
            stats["brier"] += brier.item()
            stats["conf"] += p[i].item()
        (loss / a.batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            n = 25 * a.batch
            print(f"step {step + 1}/{a.steps} reward {stats['reward'] / n:.3f} brier {stats['brier'] / n:.3f} conf {stats['conf'] / n:.3f} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
            stats = {k: 0.0 for k in stats}
    name = f"{a.model.split('/')[-1]}-brier-rl-s{a.steps}" + ("-from-sup" if a.init else "")
    path = os.path.join(a.out, name)
    model.save_pretrained(path)
    json.dump(vars(a), open(os.path.join(path, "train_args.json"), "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
