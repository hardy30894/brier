<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brier-dark.svg">
  <img alt="Brier" src="docs/brier.svg" width="340">
</picture>

**Decisions, not text. Typed, in one pass, with a probability you can trust.**

*Give any open model a schema. Get every field back at once, each with a calibrated probability, and never a malformed value.*

[![one forward pass](https://img.shields.io/badge/inference-one%20forward%20pass-3557C4?style=flat-square)](#how-it-works) [![type errors impossible](https://img.shields.io/badge/type%20errors-impossible%20by%20construction-0B8399?style=flat-square)](#how-it-works) [![calibrated](https://img.shields.io/badge/probabilities-measured%2C%20not%20claimed-8A2BE2?style=flat-square)](#benchmarks) [![status](https://img.shields.io/badge/status-milestone%201%20of%203-B7791F?style=flat-square)](#roadmap) [![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-6A6F7A?style=flat-square)](LICENSE)

</div>

Most of what software asks a language model to do is not writing. It is deciding: which
team gets this ticket, which tool to call next, is this record a duplicate, is this state
dangerous. Today that decision is made by generating a JSON string one token at a time,
parsing it, validating it, retrying when it comes back malformed, and trusting a
confidence number the model made up.

Brier turns an open causal model into a decision function. You hand it unstructured input
and a schema of typed fields. It scores every allowed value of every field in a single
batched forward pass and returns one value per field, each with a probability. The model
never writes a string, so it can never return a value outside the type. That guarantee is
structural, not learned.

Named after the Brier score, the scoring rule that defines whether a probability is honest.

New to all of this? [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks from what a language model computes to how Brier decides, how the probabilities are measured, and how the model is trained, assuming nothing.

**Contents**: [Architecture, from zero](docs/ARCHITECTURE.md) · [The primer](docs/PRIMER.md) · [How it works](#how-it-works) · [What it is not](#what-it-is-not) · [Benchmarks](#benchmarks) · [Run it](#run-it) · [On a GPU pod](#on-a-gpu-pod) · [Roadmap](#roadmap) · [Layout](#layout)

## How it works

<p align="center">
  <img src="docs/how-it-works.svg" width="900" alt="input and schema go into one batched forward pass; every allowed value of every field is scored at once; the result is a typed decision with a probability per field">
</p>

1. **Schema in.** Each field is an `Enum` over a fixed set of values or a `Bool`. The
   schema is rendered into the prompt alongside the input.
2. **One pass.** For every field, every allowed value is appended to the same prompt as a
   candidate continuation. All candidates for all fields go through the model in one
   batch. The score of a candidate is the log-probability the model assigns to it.
3. **Softmax per field.** Scores are normalised within each field, so each field carries a
   full distribution, not just an argmax.
4. **Typed out.** The top value per field is decoded back to its Python type. The
   distribution is kept, so downstream code can act on the probability, escalate below a
   threshold, or ask a human.

Because candidates are scored rather than generated, latency is one forward pass over a
batch of short sequences, and the output cannot be malformed. Fields are scored
independently in the same pass, which is the honest trade: no chain of thought, no
cross-field dependency. That is the right shape for the enormous middle of routing,
classification, extraction, and state-to-action work where a frontier model is currently
used as a slow, expensive classifier.

## What it is not

- **Not a chat model.** It cannot write prose, explain itself, or reason step by step. If a
  task needs chain of thought, generate text.
- **Not trained yet.** Milestone 1 uses an open model exactly as published. The
  probabilities come from the model's own log-likelihoods, which are known to be
  overconfident. Measuring that gap is the point of this milestone; closing it is the next.
- **Not a guarantee of correctness.** "Cannot produce a type error" means the value is
  always one of the allowed ones. It says nothing about whether it is the right one.
  Accuracy and calibration are measured, below, and reported per field.

## Benchmarks

Two synthetic tasks with computable ground truth, 200 examples each, on a single RTX 4090.
The comparison is against the same model generating a JSON object with a self-reported
confidence, which is how a decision is usually extracted from a language model today.

- **tickets**: support tickets routed to a `category` (5 values), a `priority` (4), and a
  `needs_human` flag. Signals are mixed and shuffled with noise sentences.
- **gridworld**: a 5x5 grid with a player, exit, monster, and walls. The `move` field is
  the first step of a shortest safe path (ties count as correct); `danger` is whether the
  monster is adjacent.

ECE is expected calibration error over ten confidence bins, lower is better. Brier is the
mean squared error between the reported confidence and the outcome. Latency is per
decision, all fields included.

| model | task | method | accuracy | ECE raw | ECE after temp | p50 latency | type errors |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | tickets | scored | 0.498 | 0.313 | 0.101 | 44 ms | 0 |
| Qwen3-0.6B | tickets | json | 0.368 | 0.532 | n/a | 1108 ms | 0 |
| Qwen3-0.6B | gridworld | scored | 0.603 | 0.204 | 0.070 | 41 ms | 0 |
| Qwen3-0.6B | gridworld | json | 0.625 | 0.275 | n/a | 849 ms | 0 |
| Qwen3-1.7B | tickets | scored | 0.495 | 0.438 | 0.214 | 51 ms | 0 |
| Qwen3-1.7B | tickets | json | 0.485 | 0.371 | n/a | 912 ms | 0 |
| Qwen3-1.7B | gridworld | scored | 0.540 | 0.252 | 0.112 | 40 ms | 0 |
| Qwen3-1.7B | gridworld | json | 0.647 | 0.253 | n/a | 692 ms | 0 |
| Qwen3-4B | tickets | scored | 0.585 | 0.369 | 0.158 | 111 ms | 0 |
| Qwen3-4B | tickets | json | 0.587 | 0.314 | n/a | 1268 ms | 0 |
| Qwen3-4B | gridworld | scored | 0.805 | 0.187 | 0.111 | 84 ms | 0 |
| Qwen3-4B | gridworld | json | 0.757 | 0.186 | n/a | 1031 ms | 0 |

Reading the table:

- **Speed.** The single-pass scorer is 10 to 20 times faster than JSON generation from the
  same model at every size, because there is no token loop.
- **Accuracy.** Equal to or better than JSON generation at 4B on both tasks, and mixed at
  the smaller sizes, where neither method is good. Intelligence comes from the backbone;
  the scorer does not add or remove it.
- **Type errors.** Zero for the scorer by construction. Also zero for JSON generation
  here, so at this size and with this simple a schema the guarantee is not yet the
  differentiator.
- **Calibration.** Raw log-likelihood confidence is badly overconfident: the 4B model
  reports 0.95 mean confidence on tickets against 0.59 accuracy. A single temperature per
  field, fit on half the data and scored on the other half, cuts calibration error by two
  to four times without any training. That gap is what milestone 2 is for.

Reproduce with `python3 bench.py --n 200 --model Qwen/Qwen3-4B`. Raw results, including
per-bin reliability tables, are in `results/`.

## Run it

Ask questions about a state. Every question is answered in the same forward pass.

```python
from brier import Scorer, Choice, Score, Noul, ask

scorer = Scorer("Qwen/Qwen3-4B")
answers = ask(scorer, "Webhooks are firing twice for every event since 3am. Need this today.", [
    Choice("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
    Score("priority", "how fast this must be handled", ("low", "medium", "high", "urgent")),
    Noul("needs_human", "the customer is asking for a human agent rather than an automated reply"),
])

answers["category"].value          # 'bug'
answers["category"].probabilities  # {'billing': 0.07, 'bug': 0.86, 'feature': 0.0, 'account': 0.05, 'other': 0.02}
answers["priority"].expected       # 2.3   (0 = low … 3 = urgent)
answers["needs_human"].value       # 0.06  (probability the statement is true)
answers.latency_ms                 # ~110 on a 4090
```

Then make the probabilities honest and turn them into a policy, on your own labelled
examples:

```sh
python3 -m brier calibrate --task tickets --model Qwen/Qwen3-4B --n 300
```

```
question             n    acc  ECE raw  ECE cal      T
category           300  0.680    0.295    0.054   6.46
  auto    target 0.95: act when confidence >= 0.910  -> precision 1.000, covers 4% of decisions
  review  target 0.80: act when confidence >= 0.721  -> precision 0.800, covers 48% of decisions
  human   everything below
```

[QUESTIONS.md](docs/QUESTIONS.md) is the API reference, [CALIBRATION.md](docs/CALIBRATION.md)
the kit, [TRAINING.md](docs/TRAINING.md) the training recipe and results.

Any Hugging Face causal LM works as the backbone. The scorer needs a GPU with enough memory
for the model plus a batch of short sequences.

## On a GPU pod

`pod.py` wraps RunPod with a hard time-to-live: every pod it creates is terminated by a
detached watchdog at its deadline, whatever happens to the session that started it.

```sh
echo 'RUNPOD_API_KEY=...' > .env         # gitignored, never synced to the pod
python3 pod.py up --ttl 60               # RTX 4090, forced termination after 60 minutes
python3 pod.py sync                      # ship the repo
python3 pod.py exec "bash setup.sh"      # deps, model, smoke bench
python3 pod.py exec "python3 bench.py --n 200 --model Qwen/Qwen3-4B"
python3 pod.py pull                      # bring results/ back
python3 pod.py down                      # terminate now; status should then show no pods
```

## Roadmap

- [x] **Milestone 1, zero training.** Single-pass scorer, JSON baseline, two tasks with
      ground truth, calibration measured. The number that decides whether training is
      worth it.
- [x] **The question API and the calibration kit.** Choice, Score, Noul in one pass;
      per-question temperature with cross-fitting, held-out ECE, reliability diagrams,
      and a precision-targeted auto/review/human policy.
- [ ] **Milestone 2, training for calibrated decisions.** LoRA on the backbone with
      per-question cross-entropy over the answer set, a proper scoring rule, on five
      tasks with computable ground truth. Gate: beat the temperature-scaled untrained
      calibration, including on a task never trained on. In progress; see TRAINING.md.
- [ ] **Milestone 3, live demo.** A structured-state environment where the model chooses
      an action every tick with a visible confidence bar.

## Layout

- `brier/questions.py` Choice, Score, Noul and `ask()`
- `brier/scorer.py` the single-pass scorer core and the JSON-generation baseline
- `brier/calibrate.py`, `brier/report.py` the calibration kit and its figures
- `brier/metrics.py` accuracy, ECE, Brier, latency, temperature scaling
- `train.py` LoRA training with the per-question cross-entropy objective
- `tasks/` five synthetic tasks with computable ground truth
- `bench.py` the milestone 1 scorer-versus-JSON benchmark
- `docs/` architecture, primer, API, calibration and training guides; `assets.py` draws every figure
- `pod.py` RunPod lifecycle with the time-to-live watchdog
- `setup.sh` fresh-GPU-box setup and smoke run

## License

Apache 2.0. See [LICENSE](LICENSE).
