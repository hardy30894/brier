"""Audit the generated question types before any of them reach training.

The generator writes both the question and its answer, so a mistake here is invisible later: the
model would learn a wrong rule from a confident label. Every check below re-derives the answer
from the rule the question states, rather than trusting what the generator recorded."""
import argparse
import collections

from brier.questions import Pick, Grade, Claim
from tasks.schema import example_at, LEVELS, domains_for
from tasks import vocab


def audit(n, seed, split, verbose=False):
    kinds = collections.Counter(); domains = collections.Counter(); widths = collections.Counter()
    styles = collections.Counter(); per_case = collections.Counter(); claims = collections.Counter()
    bad = nq = soft = fallback = 0
    overlap_hit = overlap_tot = 0
    state_len = []
    for i in range(n):
        e = example_at(i, seed=seed, split=split)
        checks, rec = e["_verify"]
        domains[e["domain"]] += 1
        per_case[len(e["questions"])] += 1
        st = e["state"]
        styles["structured" if isinstance(st, dict) else "text"] += 1
        state_len.append(len(str(st)))
        for q in e["questions"]:
            nq += 1
            lab = e["labels"][q.id]
            kinds[type(q).__name__] += 1
            if isinstance(lab, dict):
                soft += 1
            if isinstance(q, Pick):
                widths["2-6" if len(q.labels) < 7 else ("7-24" if len(q.labels) < 25 else "25+")] += 1
                if not isinstance(lab, dict):
                    if lab == list(q.labels)[-1]:
                        fallback += 1
                    text = str(st).lower()
                    overlap_tot += 1
                    sc = {k: sum(1 for x in str(v).lower().split("'") if len(x) > 4 and x in text)
                          for k, v in q.options.items()}
                    best = max(sc, key=sc.get)
                    if best == lab and list(sc.values()).count(sc[best]) == 1:
                        overlap_hit += 1
            if isinstance(q, Claim) and not isinstance(lab, dict):
                claims[lab] += 1
            chk = checks[q.id]
            if isinstance(q, Claim):
                truth = "yes" if chk(rec) else "no"
            elif isinstance(q, Grade):
                truth = chk(rec)
            else:
                h = chk(rec)
                truth = h[0] if len(h) == 1 else {x: 1.0 / len(h) for x in h}
            if truth != lab:
                bad += 1
                if verbose and bad < 4:
                    print("  MISMATCH", q.id, "recorded", lab, "rule says", truth)

    held = len(domains_for(vocab, "eval"))
    print(f"{n} cases, {nq} questions, seed {seed}, split {split}")
    print(f"  questions per case: {dict(sorted(per_case.items()))}")
    print(f"  kinds: {dict(kinds)}")
    print(f"  Pick widths: {dict(widths)}")
    print(f"  domains used: {len(domains)}   (corpus reserves {held} for evaluation only)")
    print(f"  state form: {dict(styles)}   median length {sorted(state_len)[len(state_len)//2]} chars")
    print(f"  claims: {dict(claims)}" + (f"   balance {min(claims.values())/sum(claims.values()):.2f}" if claims else ""))
    print(f"  genuine ties, soft-labelled: {100*soft/nq:.0f}%   fallback answer: {100*fallback/max(nq,1):.0f}%")
    print(f"  string-overlap shortcut wins: {100*overlap_hit/max(overlap_tot,1):.0f}%")
    print(f"  LABELS RE-DERIVED FROM THEIR RULES: {nq} checked, {bad} disagreements")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    raise SystemExit(1 if audit(a.n, a.seed, a.split, a.verbose) else 0)
