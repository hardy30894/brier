"""Audit the decision corpus before anything trains on it: every spec in tasks/hub.py is drawn
for train and evaluation, and its label histogram, majority rate and state length are
reported. A multi-class draw with a majority above 0.8, or a single class, is flagged; a spec
that fails to load is reported, not guessed at. Writes docs/CORPUS.md."""
import argparse
import json
import os
import sys
import time
import traceback
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
    row = {"name": spec.name, "family": spec.family, "kind": spec.kind, "held_out": spec.held_out, "path": spec.path, "config": spec.config}
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
    fams = Counter(r["family"] for r in ok)
    lines = ["# The decision corpus", "",
             "Stage two of the [roadmap](ROADMAP.md): public labelled datasets from the Hub converted into",
             "typed questions by one spec table (`tasks/hub.py`). Each spec names the dataset, its train and",
             "evaluation splits, how a row becomes a state, one question (Pick, Grade or Claim) and how the",
             "row's label maps onto that question's answers. Multiple-choice sources become Pick questions whose",
             "options are the candidate texts of that row, so the answer set changes with every example.",
             "Sources with one split are cut into disjoint halves by seed. Claim tasks with natural imbalance",
             "draw balanced classes, so accuracy is never the majority rate; label-sorted sources are loaded",
             "whole and shuffled. Held-out specs never enter training: they measure transfer to question types",
             "the model has not seen.", "",
             f"`python3 corpus_audit.py --n {n}` produced this table; it draws every spec for train and",
             "evaluation and flags a single class or a majority above 0.8. Train on the corpus with",
             "`python3 train.py --tasks hub_mixed --examples <total> …` (an even draw across every trainable spec) or on",
             "any single spec with `--tasks hub_<name>`.", "",
             f"**{len(ok)} datasets load and pass** ({', '.join(f'{k} {v}' for k, v in sorted(fams.items()))}); "
             f"{sum(1 for r in ok if r['held_out'])} are held out.", "",
             "| dataset | family | kind | options | classes seen | majority train / eval | state chars | held out | flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: (r["held_out"], r["family"], r["name"])):
        tr, ev = r["train"], r["eval"]
        flags = " ".join(tr.get("flags", []) + [f"eval:{f}" for f in ev.get("flags", [])])
        opts = tr["options"] if tr["options"] else "per row"
        lines.append(f"| `hub_{r['name']}` | {r['family']} | {r['kind']} | {opts} | {tr['classes']} / {ev['classes']} | {tr['majority']:.2f} / {ev['majority']:.2f} | {tr['state_chars']} | {'**yes**' if r['held_out'] else ''} | {flags} |")
    if bad:
        lines += ["", "## Not loaded", "", "| dataset | error |", "|---|---|"]
        for r in bad:
            lines.append(f"| `{r['path']}`{' / ' + r['config'] if r['config'] else ''} | {(r['train'].get('error') or r['eval'].get('error'))} |")
    lines += ["", "## Held out", "", "Never drawn by `hub_mixed`; used only to evaluate transfer:", ""]
    lines += [f"- `hub_{r['name']}` ({r['family']}, {r['kind']})" for r in ok if r["held_out"]]
    lines.append("")
    open(out, "w").write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
