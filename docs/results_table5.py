"""The pointer head against the row scorer, per backbone: accuracy / raw ECE, same held-out examples."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TASKS = ("tickets", "gridworld", "toolcall", "passages", "matching", "ag_news", "dbpedia", "sst5", "yelp", "snli", "qqp", "civil", "msmarco", "massive", "banking", "mnli", "emotion", "imdb", "frustration", "duplicates")
HELD = {"mnli", "emotion", "imdb", "frustration", "duplicates"}
KIND = {"category": "pick", "priority": "grade", "needs_human": "claim", "move": "pick", "danger": "claim", "tool": "pick", "answers": "claim", "matches": "claim", "topic": "pick", "kind": "pick", "sentiment": "grade", "stars": "grade", "relation": "pick", "entailed": "claim", "duplicate": "claim", "toxic": "claim", "intent": "pick", "emotion": "pick", "positive": "claim", "frustration": "grade", "same": "claim"}


def load(task, tag):
    p = R / f"cal_{task}_{tag}@eval.json"
    return json.load(open(p))["questions"] if p.exists() else {}


def table(cols):
    hdr = "| task | question | kind | " + " | ".join(t for t, _ in cols) + " |\n|---|---|---|" + "---|" * len(cols) + "\n"
    rows = []
    for task in TASKS:
        qs = {}
        for t, tag in cols:
            for qid, q in load(task, tag).items():
                qs.setdefault(qid, {})[t] = q
        for qid, per in qs.items():
            cells = [f"{per[t]['accuracy']:.2f} / {per[t]['ece_raw']:.2f}" if t in per else "" for t, _ in cols]
            rows.append(f"| {task}{' (held out)' if task in HELD else ''} | {qid} | {KIND.get(qid, '')} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows)


if __name__ == "__main__":
    print("### Qwen3-0.6B\n")
    print(table([("row scorer, mixed", "Qwen3-0.6B-mix"), ("pointer, 8k", "Qwen3-0.6B-ptr"), ("pointer, 22k", "Qwen3-0.6B-ptr-long"), ("pointer, ordinal + distilled", "Qwen3-0.6B-ptr-ord")]))
    print("\n### Phi-4-mini\n")
    print(table([("row scorer, mixed", "Phi-4-mini-mix"), ("pointer, 6k, 1 GPU", "Phi-4-mini-ptr"), ("pointer, 16k, 4 GPUs", "Phi-4-mini-ptr4")]))
