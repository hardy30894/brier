"""Round-two tables from results/cal_*@eval.json: backbones untrained, untrained vs trained per
model, and the outcome-reward runs on gridworld."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TASKS = ("tickets", "gridworld", "toolcall", "passages", "frustration", "duplicates", "banking")
UNSEEN = {"frustration", "duplicates", "banking"}


def load(task, tag):
    p = R / f"cal_{task}_{tag}@eval.json"
    return json.load(open(p))["questions"] if p.exists() else {}


def backbones(tags):
    hdr = "| task | question | " + " | ".join(tags) + " |\n|---|---|" + "---|" * len(tags) + "\n"
    rows = []
    for task in TASKS:
        qs = {}
        for t in tags:
            for qid, q in load(task, t).items():
                qs.setdefault(qid, {})[t] = q
        for qid, per in qs.items():
            cells = [f"{per[t]['accuracy']:.2f} / {per[t]['ece_raw']:.2f} → {per[t]['ece_cal']:.2f}" if t in per else "" for t in tags]
            rows.append(f"| {task} | {qid} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows)


def trained(model, adapter):
    hdr = "| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |\n|---|---|---|---|---|---|---|\n"
    rows = []
    for task in TASKS:
        b, t = load(task, model), load(task, adapter)
        for qid in b:
            if qid not in t:
                continue
            x, y = b[qid], t[qid]
            gate = "pass" if y["ece_raw"] < x["ece_cal"] else "no"
            tag = f"{task}{' (unseen)' if task in UNSEEN else ''}"
            rows.append(f"| {tag} | {qid} | {x['accuracy']:.3f} → **{y['accuracy']:.3f}** | {x['ece_raw']:.3f} → **{y['ece_raw']:.3f}** | {x['ece_cal']:.3f} → {y['ece_cal']:.3f} | {x['T']:.1f} → {y['T']:.2f} | {gate} |")
    return hdr + "\n".join(rows)


if __name__ == "__main__":
    print("### Backbones, untrained (accuracy / ECE raw → ECE after temperature)\n")
    print(backbones(["Qwen3-0.6B", "Qwen3-1.7B", "SmolLM2-1.7B-Instruct", "Phi-4-mini-instruct", "Qwen3-4B"]))
    for m, a in (("Qwen3-0.6B", "Qwen3-0.6B-brier-r16-s1500-u0.3-train"), ("Qwen3-1.7B", "Qwen3-1.7B-brier-r16-s1500-u0.3-train"), ("Qwen3-4B", "Qwen3-4B-brier-r16-s1200-u0.3-train")):
        print(f"\n### {m}: untrained → trained on held-out templates, 30% ambiguous training examples\n")
        print(trained(m, a))
    print("\n### Qwen3-0.6B on gridworld: supervised versus outcome reward\n")
    print("| run | move accuracy | move ECE raw | move ECE after temp | frustration accuracy | frustration ECE raw |\n|---|---|---|---|---|---|")
    for name, tag in (("untrained", "Qwen3-0.6B"), ("supervised (labels)", "Qwen3-0.6B-brier-r16-s1500-u0.3-train"), ("outcome reward from base", "Qwen3-0.6B-brier-rl-s600"), ("outcome reward from supervised", "Qwen3-0.6B-brier-rl-s400-from-sup")):
        g, f = load("gridworld", tag), load("frustration", tag)
        if g:
            m = g["move"]
            fr = f.get("frustration", {})
            print(f"| {name} | {m['accuracy']:.3f} | {m['ece_raw']:.3f} | {m['ece_cal']:.3f} | {fr.get('accuracy', float('nan')):.3f} | {fr.get('ece_raw', float('nan')):.3f} |")
