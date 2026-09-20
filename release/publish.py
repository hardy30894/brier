"""Publish trained Brier models to the Hub, each with a model card generated from what it was
actually trained on and measured at.

    python3 release/publish.py --org brier-lab --dry-run \
        --model qwen3-0.6b-pointer-mixed --dir adapters/v2joint1/<adapter dir> \
        --backbone Qwen/Qwen3-0.6B --kind pointer --data mixed --recipe "train.py --pointer --grade-boost 3 ..." \
        --numbers results/v2/grade/summary.json

Dry runs write the cards to release/cards/<model>/README.md and upload nothing. A real run
creates (or reuses) <org>/<model> and uploads the directory plus the card. The token comes
from HF_TOKEN in the environment or from a `.env` line `HF_TOKEN=...` at the repo root."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = {
    "mixed": "the mixed corpus: five synthetic tasks (support tickets, gridworld, tool selection, passage "
             "relevance, record matching) and ten public datasets (AG News, DBpedia, SST-5, Yelp, SNLI, QQP, "
             "Civil Comments, MS MARCO, MASSIVE, Banking77), 600 to 1,500 examples each, with 30% of the "
             "synthetic examples built to be genuinely ambiguous and labelled with soft targets; the public "
             "draws are audited for label balance (two label-sorted sources are shuffled whole, two "
             "imbalanced ones drawn balanced)",
    "typed": "the training split of a public four-workflow typed-decision benchmark (agent-trace "
             "observability, customer service, invoice processing, security incidents): 1,200 cases, five "
             "typed questions each, with the benchmark's teacher distributions as soft labels",
}

KIND = {
    "row": "a LoRA adapter for the row scorer: every allowed answer is appended to the prompt as a "
           "candidate and scored by its tokens' log-probabilities in one batched pass",
    "pointer": "a LoRA adapter trained jointly for both paths plus a pointer head (`pointer.pt`): "
               "Pick and Claim questions are scored from the hidden states where the prompt lists each "
               "option, one row per request, and Grade questions may go through either path",
    "encoder": "a fully fine-tuned encoder (`backbone.pt`, the whole state dict) plus a pointer head "
               "(`pointer.pt`): a non-generative model with no vocabulary head; every question, Grade "
               "included, is scored from the hidden states where the prompt lists each option",
}

CARD = """---
license: apache-2.0
base_model: {backbone}
library_name: {library}
tags:
- brier
- typed-decisions
- calibration
- single-pass
{extra_tags}language:
- en
---

# {org}/{name}

{what}

Backbone: `{backbone}`. This repository holds {kind}.

## What it does

Given a state (text or JSON) and a set of typed questions, Brier answers every question in
one forward pass with a probability per allowed answer. Three kinds of question cover the
space: **Pick** one of several options, **Grade** on an ordered scale, **Claim** true or false.
An answer outside the allowed set is impossible by construction. The probabilities are
trained with a proper scoring rule and measured with the calibration kit in the repository.

## Training data

Trained on {data}.

Recipe, exactly as run:

```
{recipe}
```

## Held-out numbers

{numbers}

Every number in the Brier repository is reproducible from its `results/` directory; the
tables and intervals are in `docs/TRAINING.md`.

## Known limitations

- **Unseen question types.** Accuracy on tasks absent from training is lower than on
  trained ones{transfer_note}. The training corpus covers a few hundred question types,
  not the open set.
- **Ordered scales.** Grade questions on templated scales trail Pick and Claim questions,
  especially through the pointer head; scales whose levels carry descriptions do better.
- **English only.** The training data and the evaluation are English.
- **Research preview.** Single training run unless the numbers above say otherwise; no
  guarantees for production use without calibration on your own data.

## Use

```sh
pip install brier-decisions
```

```python
from huggingface_hub import snapshot_download
from brier.__main__ import load_scorer
from brier import Pick, Grade, Claim, ask

path = snapshot_download("{org}/{name}")
scorer = load_scorer("{backbone}", path)
answers = ask(scorer, "Webhooks are firing twice for every event since 3am. Need this today.", [
    Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
    Grade("priority", "how fast this must be handled", ("low", "medium", "high", "urgent")),
    Claim("needs_human", "the customer is asking for a human agent"),
])
print(answers["category"].probabilities, answers["needs_human"].value)
```

Calibrate on your own labelled examples and get an act / review / defer policy:

```sh
brier calibrate --model {backbone} --adapter <downloaded path> --task <your task>
```

## License

Apache 2.0.
"""


def numbers_table(path):
    if not path:
        return "See `docs/TRAINING.md` in the repository."
    d = json.load(open(path))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    if isinstance(rows, dict):
        rows = [{"task": k, **v} for k, v in rows.items()]
    cols = [c for c in ("task", "question", "accuracy", "ece", "brier", "note") if any(c in r for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="brier-lab")
    ap.add_argument("--model", required=True, help="repository name under the org, e.g. qwen3-0.6b-pointer-mixed")
    ap.add_argument("--dir", required=True, help="adapter directory (adapter_config.json + adapter_model.*, pointer.pt, pointer.json) or encoder directory (backbone.pt, pointer.pt, pointer.json)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--kind", choices=list(KIND), required=True)
    ap.add_argument("--data", choices=list(DATA), required=True)
    ap.add_argument("--recipe", required=True, help="the exact training command")
    ap.add_argument("--numbers", default=None, help="JSON: a list of rows or {rows: [...]} with task/question/accuracy/ece/brier/note")
    ap.add_argument("--transfer-note", default="", help="e.g. ' (frustration 0.36 against 0.53 for the language-model scorer)'")
    ap.add_argument("--dry-run", action="store_true", help="write the card to release/cards/<model>/ and upload nothing")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    src = Path(a.dir)
    if not src.is_dir():
        sys.exit(f"not a directory: {src}")
    expected = {"row": ["adapter_config.json"], "pointer": ["adapter_config.json", "pointer.pt"], "encoder": ["backbone.pt", "pointer.pt"]}[a.kind]
    missing = [f for f in expected if not (src / f).exists()]
    if missing:
        sys.exit(f"{src} is missing {missing} for kind={a.kind}")

    what = {
        "row": "A typed decision model: an open language model used as a scorer, not a generator.",
        "pointer": "A typed decision model: an open language model used as a scorer, with a pointer head that answers a whole request from one row.",
        "encoder": "A non-generative typed decision model: an encoder-only transformer with a pointer head, no vocabulary head, one forward pass per request.",
    }[a.kind]
    card = CARD.format(org=a.org, name=a.model, backbone=a.backbone, library="transformers" if a.kind == "encoder" else "peft",
                       extra_tags=("- encoder\n- non-generative\n" if a.kind == "encoder" else "- lora\n"),
                       what=what, kind=KIND[a.kind], data=DATA[a.data], recipe=a.recipe.strip(),
                       numbers=numbers_table(a.numbers), transfer_note=a.transfer_note)

    if a.dry_run:
        out = ROOT / "release" / "cards" / a.model
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card)
        files = sorted(p.name for p in src.iterdir())
        print(f"dry run: card written to {out / 'README.md'}; would upload {files} to {a.org}/{a.model}")
        return

    tok = token()
    if not tok:
        sys.exit("no HF_TOKEN in the environment or .env")
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    repo = f"{a.org}/{a.model}"
    api.create_repo(repo, repo_type="model", private=a.private, exist_ok=True)
    stage = ROOT / "release" / "stage" / a.model
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(src, stage, ignore=shutil.ignore_patterns("__pycache__", "*.log", "checkpoint-*"))
    (stage / "README.md").write_text(card)
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model", commit_message=f"{a.model}: weights and model card")
    print(f"published https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
