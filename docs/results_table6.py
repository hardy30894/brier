"""Round six tables: baselines, seeds, the 8B backbone, and the Grade variants. Reads cal_*.json
(accuracy, raw ECE) and rec_*.json (per-example records, for intervals) under results/."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stats import per_example, bootstrap  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")


DIRS = ("v2/seeds", "v2/8b", "v2/baselines", "v2/grade", "seeds", "8b", "grade", "baselines", "")


def find(kind, task, tag, dirname=None):
    """With a directory given, only that directory; otherwise the first hit in DIRS order."""
    for d in ([dirname] if dirname else list(DIRS)):
        p = os.path.join(R, d, f"{kind}_{task}_{tag}.json")
        if os.path.exists(p):
            return p
    return None


def cal(dirname, task, tag):
    p = find("cal", task, tag, dirname)
    if not p:
        return {}
    d = json.load(open(p))
    return {q: (v["accuracy"], v["ece_raw"], v["n"]) for q, v in d["questions"].items()}


def rec(dirname, task, tag):
    p = find("rec", task, tag, dirname)
    if not p:
        return {}
    d = json.load(open(p))
    return {q: bootstrap(*per_example(r)) for q, r in d["records"].items()}, d.get("latency_ms")


def fmt(x, ci=None):
    return "–" if x is None else (f"{x:.3f}" if ci is None else f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")


def seeds(dirname, task, prefix, q, dirs=None):
    """mean ± sd across the runs tagged <prefix><digits>@eval (row-seed1, row-seed2, ...), taken
    from the first results directory that has any, so old and new corpora are never pooled."""
    pat = re.compile(rf"^cal_{re.escape(task)}_{re.escape(prefix)}\d*@eval\.json$")
    for d in ([dirname] if dirname else []) + list(dirs or DIRS):
        paths = sorted(p for p in glob.glob(os.path.join(R, d, f"cal_{task}_{prefix}*.json")) if pat.match(os.path.basename(p)))
        vals = []
        for p in paths:
            dd = json.load(open(p))
            if q in dd["questions"]:
                vals.append((dd["questions"][q]["accuracy"], dd["questions"][q]["ece_raw"]))
        if vals:
            v = np.array(vals)
            return v[:, 0].mean(), v[:, 0].std(ddof=1) if len(v) > 1 else 0.0, v[:, 1].mean(), v[:, 1].std(ddof=1) if len(v) > 1 else 0.0, len(v)
    return None


TASKS = ["tickets", "frustration", "duplicates", "banking", "msmarco", "mnli", "imdb", "emotion", "snli", "sst5", "yelp", "dbpedia", "civil"]


def first(kind_fn, *cands):
    for c in cands:
        r = kind_fn(c)
        if r:
            return r
    return None


def baselines_table(model_tag, brier_untrained_tag, brier_trained_prefix, dirs=("baselines", "seeds")):
    u_tags = brier_untrained_tag if isinstance(brier_untrained_tag, tuple) else (brier_untrained_tag,)
    t_tags = brier_trained_prefix if isinstance(brier_trained_prefix, tuple) else (brier_trained_prefix,)
    print(f"\n### Baselines · {model_tag}\n")
    print("| task · question | constrained decoding | classifier head | Brier, untrained | Brier, trained (mean of seeds) |")
    print("|---|---|---|---|---|")
    for t in TASKS:
        c = cal("", t, f"constrained-{model_tag}")
        k = cal("", t, f"classifier-{model_tag}")
        u = first(lambda tag: cal("", t, tag), *u_tags) or {}
        for q in sorted(set(c) | set(k) | set(u)):
            s = first(lambda tag: seeds("", t, tag, q), *t_tags)
            cell = lambda d: f"{d[q][0]:.3f} · {d[q][1]:.3f}" if q in d else "–"
            tr = ("–" if not s else f"**{s[0]:.3f}** ± {s[1]:.3f} · {s[2]:.3f} ± {s[3]:.3f}" if s[4] > 1 else f"**{s[0]:.3f}** · {s[2]:.3f}")
            print(f"| {t} · {q} | {cell(c)} | {cell(k)} | {cell(u)} | {tr} |")
    lat = []
    for t in TASKS:
        r = rec("", t, f"constrained-{model_tag}")
        if r and r[1]:
            lat.append(r[1])
    if lat:
        print(f"\nconstrained decoding latency, p50 per decision, median over tasks: {np.median(lat):.0f} ms")


def seeds_table():
    print("\n### Seeds · Qwen3-0.6B, mixed corpus, three training seeds\n")
    print("| task · question | untrained (3 eval seeds) | row scorer, 3 training seeds | pointer head, 3 training seeds |")
    print("|---|---|---|---|")
    for t in TASKS:
        qs = set()
        for p in glob.glob(os.path.join(R, "v2", "seeds", f"cal_{t}_*.json")):
            qs |= set(json.load(open(p))["questions"])
        for q in sorted(qs):
            u, r, pt = seeds("", t, "untrained-eval", q, ("v2/seeds",)), seeds("", t, "row-seed", q, ("v2/seeds",)), seeds("", t, "ptr-seed", q, ("v2/seeds",))
            f = lambda s: f"{s[0]:.3f} ± {s[1]:.3f} · ECE {s[2]:.3f} ± {s[3]:.3f} (n={s[4]})" if s else "–"
            print(f"| {t} · {q} | {f(u)} | {f(r)} | {f(pt)} |")


def grade_table():
    cases = (("tickets", "priority"), ("sst5", "sentiment"), ("yelp", "stars"), ("frustration", "frustration"))

    def rows(title, tags):
        print(f"\n### {title}\n")
        print("| variant | tickets · priority | sst5 · sentiment | yelp · stars | frustration (never trained on) |")
        print("|---|---|---|---|---|")
        for name, d, tag in tags:
            cells = []
            for t, q in cases:
                c = cal(d, t, tag)
                cells.append(f"{c[q][0]:.3f} · {c[q][1]:.3f}" if q in c else "–")
            print(f"| {name} | " + " | ".join(cells) + " |")

    rows("Grade on the pointer head · clean corpus, Qwen3-0.6B, seed 1 (accuracy · raw ECE)", [
        ("row scorer", "v2/seeds", "row-seed1@eval"),
        ("pointer head, quadratic term", "v2/seeds", "ptr-seed1@eval+ptrgrade"),
        ("pointer head, quadratic term, 3× Grade data", "v2/grade", "grade-gb3@eval+ptrgrade"),
        ("pointer head, cumulative link, 3× Grade data", "v2/grade", "grade-cum-gb3@eval+ptrgrade"),
        ("pointer head, joint objective", "v2/grade", "grade-joint@eval+ptrgrade"),
        ("pointer head, joint objective, 3× Grade data", "v2/grade", "grade-joint-gb3@eval+ptrgrade"),
        ("row scorer under the joint adapter, 3× Grade data", "v2/grade", "grade-joint-gb3@eval")])
    rows("Ablation · head form and rendering, earlier corpus, seed 1", [
        ("row scorer", "seeds", "row-seed1@eval"),
        ("pointer, quadratic (control)", "seeds", "ptr-seed1@eval+ptrgrade"),
        ("cumulative link", "grade", "grade-cum@eval+ptrgrade"),
        ("numbered levels", "grade", "grade-num@eval+ptrgrade"),
        ("cumulative + numbered", "grade", "grade-cum-num@eval+ptrgrade"),
        ("cumulative + numbered + 3× Grade data", "grade", "grade-cum-num-gb3@eval+ptrgrade")])


def eight_b_table():
    print("\n### Qwen3-8B · untrained and after the mixed-corpus adapter\n")
    print("| task · question | 0.6B untrained | 0.6B trained | 8B untrained | 8B trained |")
    print("|---|---|---|---|---|")
    for t in TASKS:
        u8, t8 = cal("v2/8b", t, "8b-untrained@eval"), cal("v2/8b", t, "8b-mixed@eval")
        for q in sorted(set(u8) | set(t8)):
            u6, t6 = seeds("", t, "untrained-eval", q), seeds("", t, "row-seed", q)
            f = lambda s: f"{s[0]:.3f} · {s[2]:.3f}" if s else "–"
            g = lambda c: f"{c[q][0]:.3f} · {c[q][1]:.3f}" if q in c else "–"
            print(f"| {t} · {q} | {f(u6)} | {f(t6)} | {g(u8)} | {g(t8)} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "baselines"):
        baselines_table("Qwen3-0.6B", "untrained-eval7@eval", "row-seed")
        baselines_table("Phi-4-mini-instruct", ("phi-untrained@eval", "Phi-4-mini-base@eval"), ("phi-mixed", "Phi-4-mini-mix@eval"))
        baselines_table("Qwen3-8B", "8b-untrained@eval", "8b-mixed")
    if which in ("all", "seeds"):
        seeds_table()
    if which in ("all", "grade"):
        grade_table()
    if which in ("all", "8b"):
        eight_b_table()
