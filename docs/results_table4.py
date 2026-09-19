"""Mixed-corpus round: untrained vs synthetic-only adapter vs mixed adapter, per question."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TRAIN = ("tickets", "gridworld", "toolcall", "passages", "matching", "ag_news", "dbpedia", "sst5", "yelp", "snli", "qqp", "civil", "msmarco", "massive", "banking")
HELD = ("mnli", "emotion", "imdb", "frustration", "duplicates")


def load(task, tag):
    p = R / f"cal_{task}_{tag}@eval.json"
    return json.load(open(p))["questions"] if p.exists() else {}


def table(model, tags):
    hdr = "| task | question | " + " | ".join(t for t, _ in tags) + " |\n|---|---|" + "---|" * len(tags) + "\n"
    rows = []
    for group, name in ((TRAIN, ""), (HELD, " (held out)")):
        for task in group:
            qs = {}
            for title, tag in tags:
                for qid, q in load(task, tag).items():
                    qs.setdefault(qid, {})[title] = q
            for qid, per in qs.items():
                cells = [f"{per[t]['accuracy']:.2f} / {per[t]['ece_raw']:.2f}" if t in per else "" for t, _ in tags]
                rows.append(f"| {task}{name} | {qid} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B"
    print(f"### {model}: accuracy / raw ECE, 300 held-out examples per task\n")
    print(table(model, [("untrained", f"{model}-base"), ("synthetic adapter", f"{model}-synthetic"), ("mixed adapter", f"{model}-mix"), ("mixed, 4 GPUs", f"{model}-mix4")]))
