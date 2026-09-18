import sys
from brier import Scorer
from tasks import tickets, gridworld

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
sc = Scorer(model, style="chat")
for mod, n in ((tickets, 4), (gridworld, 2)):
    for ex in mod.generate(n, seed=3):
        print("\n" + ex["context"][:200].replace("\n", " | "))
        for label, debias in (("plain", False), ("debias", True)):
            sc.debias = debias
            d = sc.decide(ex["context"], mod.SCHEMA)
            for name, fd in d.fields.items():
                top = sorted(fd.dist.items(), key=lambda kv: -kv[1])[:3]
                print(f"  {label:7} {name:12} label={ex['labels'][name]}  -> " + "  ".join(f"{c}:{p:.2f}" for c, p in top))
        g = sc.generate_json(ex["context"], mod.SCHEMA)
        print(f"  json    -> {{ {', '.join(f'{k}={v.choice}' for k, v in g.fields.items())} }} conf={list(g.fields.values())[0].prob:.2f} typeErr={g.type_error}")
