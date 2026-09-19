"""python3 -m brier calibrate --task tickets --model Qwen/Qwen3-4B --n 300 [--adapter path]"""
import argparse
import os
import sys
import time

from . import Scorer
from .calibrate import collect, calibrate, report, Calibration
from .report import reliability_svg


def load_scorer(model_id, adapter=None):
    if not adapter:
        return Scorer(model_id)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .scorer import load_causal_lm, pick_device
    tok = AutoTokenizer.from_pretrained(model_id)
    base = load_causal_lm(model_id, pick_device())
    model = PeftModel.from_pretrained(base, adapter)
    sc = Scorer(model_id, model=model, tok=tok)
    sc.model = sc.model.to(sc.device)
    head_path = os.path.join(adapter, "head.pt")
    if os.path.exists(head_path):
        from .head import TypedHead
        sc.head = TypedHead(base.config.hidden_size).to(sc.device)
        sc.head.load_state_dict(torch.load(head_path, map_location=sc.device))
        sc.head.eval()
    return sc


def cmd_calibrate(a):
    from tasks import examples
    scorer = load_scorer(a.model, a.adapter)
    tag = a.tag or (a.adapter.rstrip("/").split("/")[-1] if a.adapter else a.model.split("/")[-1])
    if a.split != "all":
        tag += f"@{a.split}"
    os.makedirs(a.out, exist_ok=True)
    for task in a.task.split(","):
        exs = examples(task, a.n, seed=a.seed, split=a.split)
        questions = exs[0]["questions"]
        fixed = all(e["questions"] is questions or [q.id for q in e["questions"]] == [q.id for q in questions] and all(x.labels == y.labels for x, y in zip(e["questions"], questions)) for e in exs)
        if fixed:
            records = collect(scorer, questions, [(e["state"], e["labels"]) for e in exs])
        else:
            records = {q.id: [] for q in questions}
            t0 = time.perf_counter()
            for i, e in enumerate(exs):
                r = collect(scorer, e["questions"], [(e["state"], e["labels"])], log_every=0)
                for k, v in r.items():
                    records[k] += v
                if (i + 1) % 50 == 0:
                    print(f"  collected {i + 1}/{len(exs)} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
        cal = calibrate(questions, records, model=tag)
        print(f"\n== {task} ==")
        print(report(cal))
        cal.save(f"{a.out}/cal_{task}_{tag}.json")
        with open(f"{a.out}/rel_{task}_{tag}.svg", "w") as f:
            f.write(reliability_svg(cal, f"{task} · {tag}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--task", default="tickets,gridworld,toolcall,passages,frustration")
    c.add_argument("--model", default="Qwen/Qwen3-4B")
    c.add_argument("--adapter", default=None)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--split", default="all", help="all | eval (templates held out from training)")
    c.add_argument("--tag", default=None, help="override the results tag")
    c.add_argument("--out", default="results")
    a = ap.parse_args()
    cmd_calibrate(a)
