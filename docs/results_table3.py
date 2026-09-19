"""Round-three tables: contrastive data + typed head + Phi backbone + multi-step outcome reward."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TASKS = ("tickets", "gridworld", "toolcall", "passages", "matching", "frustration", "duplicates", "banking")
UNSEEN = {"frustration", "duplicates", "banking"}


def load(task, tag):
    p = R / f"cal_{task}_{tag}@eval.json"
    return json.load(open(p))["questions"] if p.exists() else {}


def compare(cols):
    """cols: list of (title, tag). One row per (task, question); cell = acc / ECE raw → ECE cal."""
    hdr = "| task | question | " + " | ".join(t for t, _ in cols) + " |\n|---|---|" + "---|" * len(cols) + "\n"
    rows = []
    for task in TASKS:
        qs = {}
        for title, tag in cols:
            for qid, q in load(task, tag).items():
                qs.setdefault(qid, {})[title] = q
        for qid, per in qs.items():
            cells = [f"{per[t]['accuracy']:.2f} / {per[t]['ece_raw']:.2f} → {per[t]['ece_cal']:.2f}" if t in per else "" for t, _ in cols]
            rows.append(f"| {task}{' (unseen)' if task in UNSEEN else ''} | {qid} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows)


if __name__ == "__main__":
    print("### Qwen3-0.6B: round two → round three (contrastive data) → round three + typed head\n")
    print(compare([("untrained", "Qwen3-0.6B"), ("round 2", "Qwen3-0.6B-brier-r16-s1500-u0.3-train"), ("round 3", "Qwen3-0.6B-r3"), ("round 3 + head", "Qwen3-0.6B-r3head")]))
    print("\n### Qwen3-4B: round two → round three → round three + typed head\n")
    print(compare([("untrained", "Qwen3-4B"), ("round 2", "Qwen3-4B-brier-r16-s1200-u0.3-train"), ("round 3", "Qwen3-4B-r3"), ("round 3 + head", "Qwen3-4B-r3head")]))
    print("\n### Phi-4-mini: untrained → trained (round three recipe)\n")
    print(compare([("untrained", "Phi-4-mini-instruct"), ("trained", "Phi-4-mini-r3")]))
    print("\n### Qwen3-0.6B on gridworld: single-step versus six-step outcome episodes\n")
    print("| run | move accuracy | move ECE raw | move ECE after temp | frustration accuracy | frustration ECE raw |\n|---|---|---|---|---|---|")
    for name, tag in (("untrained", "Qwen3-0.6B"), ("supervised, round 2", "Qwen3-0.6B-brier-r16-s1500-u0.3-train"), ("outcome reward, 1 step", "Qwen3-0.6B-brier-rl-s600"), ("outcome reward, 6-step episodes", "Qwen3-0.6B-rl-ep6")):
        g, f = load("gridworld", tag), load("frustration", tag)
        if g:
            m, fr = g["move"], f.get("frustration", {})
            print(f"| {name} | {m['accuracy']:.3f} | {m['ece_raw']:.3f} | {m['ece_cal']:.3f} | {fr.get('accuracy', float('nan')):.3f} | {fr.get('ece_raw', float('nan')):.3f} |")
