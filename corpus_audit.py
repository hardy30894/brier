"""Audit the decision corpus before anything trains on it: every spec in tasks/hub.py is drawn
for train and evaluation, and its label histogram, majority rate and state length are
reported. A multi-class draw with a majority above 0.8, or a single class, is flagged; a spec
that fails to load is reported, not guessed at. Writes docs/CORPUS.md."""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks.hub import SPECS, load  # noqa: E402
from brier.calibrate import acceptable  # noqa: E402


def histogram(exs):
    c = Counter()
    for e in exs:
        q = e["questions"][0]
        lab = e["labels"][q.id]
        c[str(lab)] += 1
    return c


def audit_one(spec, n, seed):
    row = {"name": spec.name, "family": spec.family, "kind": spec.kind, "held_out": spec.held_out, "path": spec.path, "config": spec.config,
           "per_row": spec.options is not None, "note": spec.note}
    for split in ("train", "eval"):
        t0 = time.perf_counter()
        try:
            exs = load(spec, n, seed=seed, split=split)
            c = histogram(exs)
            top = max(c.values()) / max(1, sum(c.values()))
            n_opt = len(exs[0]["questions"][0].labels) if exs else 0
            for e in exs[:5]:
                acceptable(e["questions"][0], e["labels"][e["questions"][0].id])   # the label must map onto the answer set
            row[split] = {"n": len(exs), "classes": len(c), "options": n_opt, "majority": round(top, 3), "hist": dict(c.most_common(8)),
                          "state_chars": round(sum(len(json.dumps(e["state"])) for e in exs) / max(1, len(exs))), "seconds": round(time.perf_counter() - t0, 1)}
            flags = []
            if len(c) == 1:
                flags.append("SINGLE-CLASS")
            elif spec.kind != "claim" and n_opt > 2 and top > 0.8:
                flags.append("SKEWED")
            elif spec.kind == "claim" and top > 0.8:
                flags.append("SKEWED")
            row[split]["flags"] = flags
        except Exception as e:
            row[split] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default=None, help="comma-separated spec names")
    ap.add_argument("--out", default="docs/CORPUS.md")
    ap.add_argument("--json", default="results/corpus_audit.json")
    a = ap.parse_args()
    specs = [s for s in SPECS if not a.only or s.name in a.only.split(",")]
    rows = []
    for s in specs:
        r = audit_one(s, a.n, a.seed)
        rows.append(r)
        tr, ev = r.get("train", {}), r.get("eval", {})
        status = "ERROR " + (tr.get("error") or ev.get("error")) if "error" in tr or "error" in ev else " ".join(tr.get("flags", []) + [f"eval:{f}" for f in ev.get("flags", [])]) or "ok"
        print(f"{s.name:22} {s.kind:5} train n={tr.get('n', 0):3} cls={tr.get('classes', 0):2} maj={tr.get('majority', 0):.2f}  eval n={ev.get('n', 0):3} maj={ev.get('majority', 0):.2f}  {status}", flush=True)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(rows, open(a.json, "w"), indent=1)
    if not a.only:
        write_doc(rows, a.out, a.n)


def write_doc(rows, out, n):
    ok = [r for r in rows if "error" not in r["train"] and "error" not in r["eval"]]
    bad = [r for r in rows if r not in ok]
    flagged = [r for r in ok if r["train"].get("flags") or r["eval"].get("flags")]
    fams = Counter(r["family"] for r in ok)
    held = [r for r in ok if r["held_out"]]
    lines = ["# The decision corpus", "",
             "Stage two of the [roadmap](ROADMAP.md): public labelled datasets converted into typed",
             "questions by one spec table (`tasks/hub.py`). Each spec names the dataset, its train and",
             "evaluation splits, how a row becomes a state, one question (Pick, Grade or Claim) and how the",
             "row's label maps onto that question's answers. There is no loader per dataset: one generic",
             "`load(spec, …)` builds every one of them, and each is registered as `hub_<name>`.", "",
             f"**{len(ok)} datasets load and pass** ({', '.join(f'{k} {v}' for k, v in sorted(fams.items()))}); "
             f"{len(held)} are held out of training.", "",
             "## How the corpus is drawn", "",
             "Every loader takes `(n, seed=0, split=\"all\", uncertain=0.0)` like every other task. `split=\"train\"`",
             "draws the train split; anything else draws the evaluation split. Within that:", "",
             "- **Seeded shuffle.** Sources are streamed through a seeded shuffle, so the same seed gives the",
             "  same draw. A streaming shuffle only mixes within its buffer, so sources stored in label order",
             "  (star ratings, intent sets) are loaded whole and shuffled globally instead.",
             "- **Balanced classes.** Claim tasks with natural imbalance (toxicity, spam, passage relevance,",
             "  acceptability) draw equal numbers of each class, so accuracy is never just the majority rate.",
             "- **Disjoint halves.** A source with only one usable split is streamed once and cut in two, so",
             "  the training half and the evaluation half share no row. Both draws use the same seed: a",
             "  different seed per split would reshuffle and put training rows back into evaluation. Where a",
             "  source repeats itself verbatim (`sms_spam` and `app_reviews` both do) the same text can still",
             "  appear on both sides; the rows are disjoint, the strings are the source's own duplicates.",
             "- **Options per row.** Multiple-choice sources become Pick questions whose options are that row's",
             "  own candidate texts, so the answer set changes with every example rather than being fixed.",
             "- **Long states are cut** to a fixed character budget with a marker, so no one source dominates",
             "  the sequence length.", "",
             "`hub_mixed` draws evenly across every trainable spec and shuffles the result: a corpus that is",
             "one question kind per example but many kinds across the draw. Held-out specs never appear in it.", "",
             "## Why the audit exists", "",
             "A dataset that loads is not a dataset that teaches. A silently broken draw looks exactly like a",
             "working one at the call site: the rows arrive, the labels are strings, training runs, and the",
             "resulting accuracy is meaningless because every row carried the same label, or because the",
             "answer set the question offered never matched the label the source gave. Both failures are",
             "invisible until the numbers are read, and by then the run has been paid for.", "",
             f"`python3 corpus_audit.py --n {n}` draws every spec for both splits before anything trains on it",
             "and reports the label histogram, the number of distinct classes seen, the majority-class rate and",
             "the mean state length. It flags a **single-class** draw and a **skewed** one (majority above 0.8),",
             "checks that the first rows' labels are answers the question actually allows, and reports a spec",
             "that fails to load rather than guessing at its schema. Three label-ordered sources were caught",
             "this way and are now loaded whole and shuffled.", "",
             "This table is that run's output" + (f"; {len(flagged)} carry a flag." if flagged else "; nothing is flagged."), "",
             "## Every dataset", "",
             "| dataset | family | kind | options | classes seen train / eval | majority train / eval | state chars | held out | flags | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: (r["held_out"], r["family"], r["name"])):
        tr, ev = r["train"], r["eval"]
        flags = " ".join(tr.get("flags", []) + [f"eval:{f}" for f in ev.get("flags", [])])
        opts = f"{tr['options']} per row" if r.get("per_row") else tr["options"]
        lines.append(f"| `hub_{r['name']}` | {r['family']} | {r['kind']} | {opts} | {tr['classes']} / {ev['classes']} | {tr['majority']:.2f} / {ev['majority']:.2f} | {tr['state_chars']} | {'**yes**' if r['held_out'] else ''} | {flags} | {r.get('note', '')} |")
    lines += ["", "`per row` means the options are that row's own candidate texts rather than a fixed set, which is",
              "why those rows show a classes count near `n`: almost every example has a different answer string.", ""]
    if bad:
        lines += ["## Not loaded", "", "Dropped rather than guessed at:", "", "| dataset | error |", "|---|---|"]
        for r in bad:
            lines.append(f"| `{r['path']}`{' / ' + r['config'] if r['config'] else ''} | {(r['train'].get('error') or r['eval'].get('error'))} |")
        lines.append("")
    lines += ["## Held out", "",
              "Never drawn by `hub_mixed`. They span all three question kinds, so transfer can be measured to",
              "a question type the model was not trained on rather than only to unseen wording:", ""]
    lines += [f"- `hub_{r['name']}` — {r['family']}, {r['kind']}" for r in held]
    lines += ["", "## Running it", "",
              f"```\npython3 corpus_audit.py --n {n}                     # audit every spec, rewrite this file",
              "python3 corpus_audit.py --n 50 --only piqa,boolq    # one or two specs, no rewrite",
              "```", "",
              "Train on the whole corpus, or on any single spec:", "",
              f"```\npython3 train.py --tasks hub_mixed --examples {len([r for r in ok if not r['held_out']]) * 100} --model <backbone>",
              "python3 train.py --tasks hub_piqa,hub_liar --examples 2000 --model <backbone>",
              "```", ""]
    open(out, "w").write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
