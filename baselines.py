"""The two opponents a single-pass scorer has to beat, evaluated with the same kit and the same
held-out examples as everything else.

constrained  Constrained decoding: the backbone generates each answer token by token, with the
             vocabulary masked at every step to the tokens that can still complete one of the
             allowed answers (a trie over their token sequences). The type guarantee is the same
             as ours; the cost is one forward pass per generated token per question, and the
             confidence is the product of the chosen tokens' probabilities renormalised over the
             tokens the mask allowed.

classifier   A fine-tuned classifier: the backbone's last hidden state at the end of the prompt
             (frozen, no adapter) with a logistic-regression head per question, trained on the
             same training examples an adapter sees and read out with the same softmax.

Both write rec_*.json files that stats.py and the calibration kit understand."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from brier.__main__ import load_scorer
from brier.calibrate import acceptable, calibrate, report
from brier.questions import Grade, Claim, build_prompt, groups_for
from tasks import examples


class Trie:
    def __init__(self, seqs):
        self.root = {}
        for i, seq in enumerate(seqs):
            node = self.root
            for t in seq:
                node = node.setdefault(t, {})
            node[None] = i

    def allowed(self, prefix):
        node = self.root
        for t in prefix:
            node = node.get(t)
            if node is None:
                return {}
        return node


@torch.no_grad()
def constrained_answer(scorer, prompt_ids, header_ids, cands):
    """Greedy decode one question's answer under the trie mask. Returns (index, confidence)."""
    tok, model, dev = scorer.tok, scorer.model, scorer.device
    seqs = [scorer._enc(c) for c in cands]
    trie = Trie(seqs)
    ids = torch.tensor([prompt_ids + header_ids], device=dev)
    out = model(input_ids=ids, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    prefix, conf = [], 1.0
    while True:
        node = trie.allowed(prefix)
        keys = [k for k in node if k is not None]
        if None in node and not keys:
            return node[None], conf
        allowed = torch.tensor(keys + ([tok.eos_token_id] if None in node else []), device=dev)
        p = torch.softmax(logits[allowed].float(), dim=0)
        j = int(p.argmax())
        if None in node and j == len(keys):
            return node[None], conf * float(p[j])
        conf *= float(p[j])
        prefix.append(keys[j])
        out = model(input_ids=torch.tensor([[keys[j]]], device=dev), past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]


def run_constrained(scorer, task, exs, out_dir, tag):
    questions = exs[0]["questions"]
    records = {q.id: [] for q in questions}
    lats = []
    t0 = time.perf_counter()
    for i, ex in enumerate(exs):
        prompt = build_prompt(scorer, ex["state"], ex["questions"])
        pids = scorer._enc(prompt)
        t1 = time.perf_counter()
        for q, (header, cands) in zip(ex["questions"], groups_for(ex["questions"])):
            idx, conf = constrained_answer(scorer, pids, scorer._enc(header), cands)
            if q.id in ex["labels"]:
                # scores: log-confidence on the chosen answer, the rest share what is left
                rest = np.log(max(1e-6, (1 - conf) / max(1, len(cands) - 1)))
                scores = [float(np.log(max(conf, 1e-6))) if k == idx else float(rest) for k in range(len(cands))]
                records[q.id].append((scores, acceptable(q, ex["labels"][q.id])))
        lats.append((time.perf_counter() - t1) * 1000)
        if (i + 1) % 50 == 0:
            print(f"  constrained {task}: {i + 1}/{len(exs)} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    finish(task, tag, questions, records, exs, out_dir, latency_ms=float(np.median(lats)))


def hidden_at_end(scorer, prompt):
    ids = torch.tensor([scorer._enc(prompt)], device=scorer.device)
    with torch.no_grad():
        from brier.pointer import decoder
        h = decoder(scorer.model)(input_ids=ids).last_hidden_state[0, -1]
    return h.float().cpu().numpy()


def run_classifier(scorer, task, train_exs, exs, out_dir, tag):
    from sklearn.linear_model import LogisticRegression
    questions = exs[0]["questions"]
    t0 = time.perf_counter()
    feats = lambda es: np.stack([hidden_at_end(scorer, build_prompt(scorer, e["state"], e["questions"])) for e in es])
    Xtr, Xte = feats(train_exs), feats(exs)
    print(f"  classifier {task}: features for {len(train_exs)}+{len(exs)} examples ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    records = {q.id: [] for q in questions}
    lats = []
    for q in questions:
        ytr = [sorted(acceptable(q, e["labels"][q.id]))[0] for e in train_exs if q.id in e["labels"]]
        Xq = np.stack([x for x, e in zip(Xtr, train_exs) if q.id in e["labels"]])
        if len(set(ytr)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xq, ytr)
        classes = list(clf.classes_)
        for x, e in zip(Xte, exs):
            if q.id not in e["labels"]:
                continue
            t1 = time.perf_counter()
            lp = clf.predict_log_proba(x[None])[0]
            lats.append((time.perf_counter() - t1) * 1000)
            scores = [float(lp[classes.index(k)]) if k in classes else -30.0 for k in range(len(q.labels))]
            records[q.id].append((scores, acceptable(q, e["labels"][q.id])))
    finish(task, tag, questions, records, exs, out_dir)


def finish(task, tag, questions, records, exs, out_dir, latency_ms=None):
    os.makedirs(out_dir, exist_ok=True)
    cal = calibrate(questions, records, model=tag)
    print(f"\n== {task} · {tag} ==" + (f"  latency p50 {latency_ms:.0f} ms/decision" if latency_ms else ""))
    print(report(cal))
    cal.save(f"{out_dir}/cal_{task}_{tag}.json")
    with open(f"{out_dir}/rec_{task}_{tag}.json", "w") as f:
        json.dump({"task": task, "tag": tag, "latency_ms": latency_ms, "labels": {q.id: list(q.labels) for q in questions},
                   "records": {k: [[s, sorted(acc)] for s, acc in v] for k, v in records.items()}}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--method", default="constrained,classifier")
    ap.add_argument("--task", default="tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--train-n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    scorer = load_scorer(a.model, a.adapter)
    base_tag = a.tag or a.model.split("/")[-1]
    for task in a.task.split(","):
        exs = examples(task, a.n, seed=a.seed, split="eval")
        if "constrained" in a.method:
            run_constrained(scorer, task, exs, a.out, f"constrained-{base_tag}")
        if "classifier" in a.method:
            train_exs = examples(task, a.train_n, seed=100, split="train", uncertain=0.0)
            run_classifier(scorer, task, train_exs, exs, a.out, f"classifier-{base_tag}")
