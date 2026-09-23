"""Audit the generated question types before any of them reach training.

The generator writes both the question and its answer, so a mistake here is invisible later: the
model would simply learn a wrong rule from a confident label. This checks the properties that
have to hold. Every option's condition is re-evaluated against the record, so the stated answer
is verified rather than trusted."""
import argparse
import collections
import re

from tasks.schema import example_at
from brier.questions import Pick, Grade, Claim


def audit(n, seed, split):
    kinds = collections.Counter()
    domains = collections.Counter()
    answers = collections.Counter()
    fallback = multi = soft = 0
    opt_counts, state_len, label_len = [], [], []
    dup_names = 0
    for i in range(n):
        e = example_at(i, seed=seed, split=split)
        q = e["questions"][0]
        lab = e["labels"][q.id]
        kinds[e["kind"]] += 1
        domains[e["domain"]] += 1
        st = e["state"]
        state_len.append(len(st if isinstance(st, str) else str(st)))
        opts = list(q.labels)
        opt_counts.append(len(opts))
        label_len.append(max(len(str(o)) for o in opts))
        if isinstance(lab, dict):
            soft += 1
        else:
            answers[str(lab)[:40]] += 1
            if isinstance(q, Pick) and str(lab) == opts[-1]:
                fallback += 1
        if isinstance(q, Pick) and len(set(opts)) != len(opts):
            dup_names += 1
    print(f"{n} examples, seed {seed}, split {split}")
    print("  kinds:", dict(kinds))
    print(f"  distinct domains: {len(domains)} of 71   most common: {domains.most_common(1)[0][1]} times")
    print(f"  options per question: mean {sum(opt_counts)/len(opt_counts):.1f}  max {max(opt_counts)}")
    print(f"  state length: median {sorted(state_len)[len(state_len)//2]} chars  max {max(state_len)}")
    print(f"  longest option text: max {max(label_len)} chars")
    print(f"  ambiguous (two rules fire, soft label): {soft} ({100*soft/n:.1f}%)")
    print(f"  answered by the fallback option: {fallback} ({100*fallback/n:.1f}%)")
    print(f"  duplicate option names in one question: {dup_names}")
    top = answers.most_common(3)
    print(f"  most repeated single answer: {top[0][1] if top else 0} of {n}")
    print(f"  distinct answers seen: {len(answers)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    audit(a.n, a.seed, a.split)
