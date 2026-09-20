"""Pool calibration reports by base question id, weighted by n. When a task's option set varies
per example, records are keyed `id#k`; every such key belongs to the same question."""
import glob
import json
import os
import re
import sys
from collections import defaultdict


def read(pattern):
    out = defaultdict(lambda: defaultdict(list))   # (task, base qid) -> tag -> [(acc, ece, n)]
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)
        if not b.startswith("cal_"):
            continue
        d = json.load(open(f))
        tag_full = d.get("model") or b.split("_", 2)[2][:-5]       # the tag the report was written with
        task = b[4:-5]
        if task.endswith("_" + tag_full):
            task = task[: -len(tag_full) - 1]                      # dataset names may contain underscores
        tag = tag_full.split("@")[0]
        for q, v in d["questions"].items():
            out[(task, q.split("#")[0])][tag].append((v["accuracy"], v["ece_raw"], v["n"]))
    return out


def pooled(rows):
    n = sum(r[2] for r in rows)
    return sum(r[0] * r[2] for r in rows) / n, sum(r[1] * r[2] for r in rows) / n, n


if __name__ == "__main__":
    pattern = sys.argv[1]
    groups = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    data = read(pattern)
    for (task, q), tags in sorted(data.items()):
        cells = []
        for g in (groups or sorted(tags)):
            vals = [r for t, rs in tags.items() if t.startswith(g) for r in rs]
            per_run = [pooled([r for r in rs]) for t, rs in tags.items() if t.startswith(g)]
            if not vals:
                cells.append(f"{g}: –")
                continue
            a, e, n = pooled(vals)
            sd = (sum((p[0] - a) ** 2 for p in per_run) / (len(per_run) - 1)) ** 0.5 if len(per_run) > 1 else None
            cells.append(f"{g}: {a:.3f}" + (f" ± {sd:.3f}" if sd is not None else "") + f" · ECE {e:.3f} (n={n}, runs={len(per_run)})")
        print(f"{task:10} {q:12} " + "   ".join(cells))
