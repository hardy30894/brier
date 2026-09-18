import argparse
import json
import os
import sys
import time

from brier import Scorer
from brier.metrics import summarize, fit_temperature, softmax_t
from tasks import tickets, gridworld

TASKS = {"tickets": tickets, "gridworld": gridworld}


def is_correct(choice, label):
    return choice in label if isinstance(label, tuple) else choice == label


def run(scorer, mod, n, seed, methods):
    data = mod.generate(n, seed=seed)
    schema = mod.SCHEMA
    report = {}
    for method in methods:
        fn = scorer.decide if method == "scored" else scorer.generate_json
        per_field = {f.name: {"conf": [], "ok": [], "records": []} for f in schema.fields}
        lat, type_errors = [], 0
        t0 = time.perf_counter()
        for i, ex in enumerate(data):
            d = fn(ex["context"], schema)
            lat.append(d.latency_ms)
            type_errors += d.type_error
            for f in schema.fields:
                fd = d.fields[f.name]
                lab = ex["labels"][f.name]
                ok = is_correct(fd.choice, lab)
                per_field[f.name]["conf"].append(fd.prob)
                per_field[f.name]["ok"].append(float(ok))
                if fd.logits:
                    idx = f.choices.index(lab if isinstance(lab, str) else (fd.choice if ok else lab[0]))
                    per_field[f.name]["records"].append(([fd.logits[c] for c in f.choices], idx))
            if (i + 1) % 25 == 0:
                print(f"  {method}: {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
        rep = {"latency_p50_ms": None, "fields": {}}
        allc, allok = [], []
        for name, v in per_field.items():
            s = summarize(v["conf"], v["ok"], lat, type_errors)
            if v["records"]:
                half = len(v["records"]) // 2
                T = fit_temperature(v["records"][:half])
                cal_conf, cal_ok = [], []
                for (logits, idx), ok in zip(v["records"][half:], v["ok"][half:]):
                    p = softmax_t(logits, T)
                    cal_conf.append(float(p.max()))
                    cal_ok.append(ok)
                s["temp_scaled"] = {"T": T, **{k: val for k, val in summarize(cal_conf, cal_ok, lat).items() if k in ("ece", "brier", "mean_conf", "accuracy", "n")}}
            rep["fields"][name] = s
            allc += v["conf"]
            allok += v["ok"]
        rep["overall"] = summarize(allc, allok, lat, type_errors)
        report[method] = rep
    return report


def fmt(report, task):
    lines = [f"\n== {task} =="]
    lines.append(f"{'method':8} {'field':12} {'acc':>6} {'ECE':>6} {'Brier':>6} {'conf':>6} {'p50ms':>7} {'p95ms':>7} {'typeErr':>7}")
    for method, rep in report.items():
        for name, s in list(rep["fields"].items()) + [("ALL", rep["overall"])]:
            lines.append(f"{method:8} {name:12} {s['accuracy']:6.3f} {s['ece']:6.3f} {s['brier']:6.3f} {s['mean_conf']:6.3f} {s['lat_p50_ms']:7.1f} {s['lat_p95_ms']:7.1f} {s['type_errors']:7d}")
            ts = s.get("temp_scaled")
            if ts:
                lines.append(f"{'':8} {'  +temp':12} {ts['accuracy']:6.3f} {ts['ece']:6.3f} {ts['brier']:6.3f} {ts['mean_conf']:6.3f} {'':7} {'':7} {'T=' + format(ts['T'], '.2f'):>7}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tickets,gridworld")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="scored,json")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--style", default="json")
    ap.add_argument("--debias", action="store_true")
    a = ap.parse_args()
    scorer = Scorer(a.model, style=a.style, debias=a.debias)
    print(f"device={scorer.device} model={a.model}", file=sys.stderr)
    out = {}
    for t in a.tasks.split(","):
        out[t] = run(scorer, TASKS[t], a.n, a.seed, a.methods.split(","))
        print(fmt(out[t], t))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{a.model.split('/')[-1]}_{a.seed}.json", "w") as f:
        json.dump(out, f, indent=1)
