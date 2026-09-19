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
        model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=lora_targets(base), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    scorer = Scorer(a.model, model=model, tok=tok)
    scorer.model = scorer.model.to(scorer.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    q = gridworld.QUESTIONS[0]
    baseline, t0 = 0.5, time.perf_counter()
    stats = {"reward": 0.0, "brier": 0.0, "conf": 0.0, "n": 0}
    model.train()
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
    name = f"{a.model.split('/')[-1]}-brier-rl-s{a.steps}" + (f"-ep{a.episode}" if a.episode > 1 else "") + ("-from-sup" if a.init else "")
    path = os.path.join(a.out, name)
    model.save_pretrained(path)
    json.dump(vars(a), open(os.path.join(path, "train_args.json"), "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
