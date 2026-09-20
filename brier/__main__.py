"""python3 -m brier calibrate --task tickets --model Qwen/Qwen3-4B --n 300 [--adapter path]"""
import argparse
import json
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
    if os.path.exists(os.path.join(adapter, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, adapter)
    elif os.path.exists(os.path.join(adapter, "backbone.pt")):
        base.load_state_dict(torch.load(os.path.join(adapter, "backbone.pt"), map_location="cpu"))
        model = base
    else:
        model = base
    sc = Scorer(model_id, model=model, tok=tok)
    sc.model = sc.model.to(sc.device)
    ptr_path = os.path.join(adapter, "pointer.pt")
    if os.path.exists(ptr_path):
        from .pointer import PointerHead
        meta_path = os.path.join(adapter, "pointer.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        sc.grade_numbers = meta.get("grade_numbers", False)
        sc.pointer = PointerHead(base.config.hidden_size, ordinal=meta.get("ordinal", "quadratic")).to(sc.device)
        sc.pointer.load_state_dict(torch.load(ptr_path, map_location=sc.device), strict=False)
        sc.pointer.eval()
        if sc.encoder:
            sc.pointer_grade = True   # an encoder has no vocabulary head, so every question goes through the pointer
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
    scorer.pointer_grade = a.pointer_grade or scorer.pointer_grade
    tag = a.tag or (a.adapter.rstrip("/").split("/")[-1] if a.adapter else a.model.split("/")[-1])
    if a.split != "all":
        tag += f"@{a.split}"
    if a.pointer_grade:
        tag += "+ptrgrade"
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
        with open(f"{a.out}/rec_{task}_{tag}.json", "w") as f:
            json.dump({"task": task, "tag": tag, "seed": a.seed, "labels": {q.id: list(q.labels) for q in questions},
                       "records": {k: [[s, sorted(acc)] for s, acc in v] for k, v in records.items()}}, f)
        with open(f"{a.out}/rel_{task}_{tag}.svg", "w") as f:
            f.write(reliability_svg(cal, f"{task} · {tag}"))


def cmd_fetch(a):
    """Download a released adapter (or calibration file) into adapters/ with the GitHub CLI."""
    import subprocess
    os.makedirs("adapters", exist_ok=True)
    asset = a.name if a.name.endswith(".json") else f"{a.name}.tar.gz"
    subprocess.run(["gh", "release", "download", a.tag, "--repo", "hardy30894/brier", "--pattern", asset, "--dir", "adapters", "--clobber"], check=True)
    if asset.endswith(".tar.gz"):
        subprocess.run(["tar", "-xzf", os.path.join("adapters", asset), "-C", "adapters"], check=True)
        os.remove(os.path.join("adapters", asset))
        print(f"adapters/ now holds {a.name}; pass its directory as --adapter")
    else:
        print(f"adapters/{asset}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("name", help="qwen3-0.6b-mixed | phi-4-mini-mixed | qwen3-0.6b-pointer | phi-4-mini-pointer | qwen3-0.6b-gridworld-outcome | phi-4-mini-tickets-calibration.json")
    f.add_argument("--tag", default="adapters-v1")
    c = sub.add_parser("calibrate")
    c.add_argument("--task", default="tickets,gridworld,toolcall,passages,frustration")
    c.add_argument("--model", default="Qwen/Qwen3-4B")
    c.add_argument("--adapter", default=None)
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--split", default="all", help="all | eval (templates held out from training)")
    c.add_argument("--tag", default=None, help="override the results tag")
    c.add_argument("--pointer-grade", action="store_true", help="route Grade questions through the pointer head too")
    c.add_argument("--out", default="results")
    a = ap.parse_args()
    {"calibrate": cmd_calibrate, "fetch": cmd_fetch}[a.cmd](a)
