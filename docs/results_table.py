"""Prints the milestone 2 comparison table from results/cal_*.json (untrained vs trained)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TASKS = ("tickets", "gridworld", "toolcall", "passages", "frustration")


def load(task, tag):
    p = R / f"cal_{task}_{tag}.json"
    return json.load(open(p))["questions"] if p.exists() else {}


def table(model, adapter):
    rows = []
    for task in TASKS:
        base, trained = load(task, model), load(task, adapter)
        for qid in base:
            b, t = base[qid], trained.get(qid)
            if not t:
                continue
            gate = "pass" if t["ece_raw"] < b["ece_cal"] else "no"
            rows.append(f"| {task}{' (unseen)' if task == 'frustration' else ''} | {qid} | {b['accuracy']:.3f} → **{t['accuracy']:.3f}** | {b['ece_raw']:.3f} → **{t['ece_raw']:.3f}** | {b['ece_cal']:.3f} → {t['ece_cal']:.3f} | {b['T']:.1f} → {t['T']:.2f} | {gate} |")
    hdr = "| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |\n|---|---|---|---|---|---|---|\n"
    return hdr + "\n".join(rows)


if __name__ == "__main__":
    for model, adapter in (("Qwen3-0.6B", "Qwen3-0.6B-brier-r16-s1500"), ("Qwen3-1.7B", "Qwen3-1.7B-brier-r16-s1500"), ("Qwen3-4B", "Qwen3-4B-brier-r16-s1200")):
        print(f"\n### {model}\n")
        print(table(model, adapter))
