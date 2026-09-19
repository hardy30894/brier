<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brier-dark.png">
  <img alt="Brier" src="docs/brier.png" width="150">
</picture>

<br>

**Calibrated typed decisions from any open language model, in one forward pass.**

[Architecture](docs/ARCHITECTURE.md) · [Primer](docs/PRIMER.md) · [API](docs/QUESTIONS.md) · [Calibration](docs/CALIBRATION.md) · [Training](docs/TRAINING.md) · [Milestones](docs/MILESTONES.md) · [Demo](docs/demo.html)

[![license](https://img.shields.io/badge/license-Apache%202.0-1B1F24?style=flat-square)](LICENSE) [![status](https://img.shields.io/badge/status-research%20preview-4F46E5?style=flat-square)](docs/TRAINING.md) [![adapters](https://img.shields.io/badge/adapters-v1-1B1F24?style=flat-square)](https://github.com/hardy30894/brier/releases/tag/adapters-v1)

</div>

## Abstract

Language models are asked to decide far more often than they are asked to write: route
this ticket, pick a tool, is this passage relevant, same customer or not. Each is a choice
from a short list, and each is answered today by generating a JSON string one token at a
time, parsing it, retrying when it breaks, and believing a confidence the model made up.

Brier does not write the answer. It scores every allowed answer at once, in one forward
pass of an open model, and returns a probability for each. There is no token loop, so a
decision costs a tenth of generation on the same weights, and no string, so an answer
outside the list is not unlikely but impossible. Because the probability is the product,
Brier measures whether it is true: it fits one temperature per question on held-out
decisions and reports the exact confidence above which the model may act alone at a
required precision, and how much of the work that covers.

Training then does what temperature cannot. A proper scoring rule over the answer set, on
decisions built with real ambiguity and deliberate near-misses, makes a model sharper and
better calibrated at once, on tasks it has never seen. The same rule learns from outcomes
alone, with no labels. A pointer head reads every option's score from the prompt's own
hidden states, so a request of any width is one row; replayed from a CUDA graph and batched
across clients, a 0.6B backbone answers in 12 ms and serves 170 decisions a second on one
card.

<p align="center"><img src="docs/fig-results.svg" width="900" alt="three panels: latency per decision for one pass versus generation; raw calibration error on never-trained tasks before and after training; reliability of the 4B model on the frustration task before and after"></p>

## Contributions

1. **A single-pass typed decision function** for any open-weight causal language model:
   `ask(scorer, state, questions)` returns a distribution per question with the type
   guarantee built in. Five backbones from three families run unchanged.
2. **A calibration kit** that turns those distributions into a decision policy:
   per-question temperature by cross-fitted maximum likelihood, expected calibration
   error and Brier score raw and corrected, and precision-targeted act / review / defer
   tiers with their coverage.
3. **A training recipe** that improves accuracy and calibration together: cross-entropy
   over the answer set on soft targets, contrastive negatives against learned shortcuts,
   an optional typed head, and an outcome-reward form that needs no labels.
4. **A pointer head** that scores every option of every question from one row of hidden
   states, with no vocabulary projection, plus a serving path (CUDA-graph capture,
   request batching) measured end to end over HTTP.
5. **Eight tasks, twelve public datasets, and every number reproducible** from `results/`
   with the scripts that produced it.

## Results

All numbers are held-out. Templated tasks split their templates into disjoint train and
evaluation halves; "never trained on" means the task or dataset was absent from training
entirely. Protocol, full tables and per-round lessons are in [TRAINING.md](docs/TRAINING.md).

### One pass against generation

| | generating JSON | one pass | |
|---|---|---|---|
| Latency, three questions, Qwen3-4B, RTX 4090 | 1268 ms | **111 ms** | 11× |
| Type errors, every model, every task | 0 (with retries) | **0** | by construction |
| Confidence | self-reported, made up | a measured probability | see below |

### Calibration and training

| task · backbone | untrained | trained | |
|---|---|---|---|
| Frustration (never trained on), Qwen3-4B | 0.44 acc · 0.54 ECE | **0.83 acc · 0.20 ECE** | raw, no correction |
| Duplicates (never trained on), Qwen3-4B | 0.73 acc · 0.26 ECE | **0.997 acc · 0.04 ECE** | raw |
| Banking77 (real, never trained on), Qwen3-0.6B | 0.85 acc · 0.10 ECE | **0.90 acc · 0.02 ECE** | raw |
| MS MARCO relevance, Qwen3-0.6B, mixed corpus | 0.12 acc · 0.83 ECE | **0.88 acc · 0.07 ECE** | real, raw |
| MNLI (never trained on), Qwen3-0.6B, mixed corpus | 0.38 acc | **0.77 acc** | real, held-out dataset |
| Civil Comments toxicity, Phi-4-mini, mixed corpus | 0.49 acc · 0.33 ECE | **0.94 acc · 0.02 ECE** | real, raw |
| Ticket category, Phi-4-mini | 0.71 acc | **0.96 acc** | held-out templates |
| Gridworld from outcomes only, Qwen3-0.6B | 0.35 acc | **0.81 acc** | no labels; 0.84 supervised |

Three findings shape the recipe. Clean labels alone teach overconfidence, and the fix is
training data with irreducible uncertainty and soft labels equal to the generator's own
posterior. A shortcut learned from cue-triggered claims transfers as a bias, and the fix is
a contrastive claim where the cue is present and the answer is no. A 0.6B backbone with a
25-minute adapter beats an untrained 4B on every non-templated task at a quarter of the
latency.

### Serving

| | row scorer | pointer head | |
|---|---|---|---|
| Ten questions, Phi-4-mini, RTX 4090 | 852 ms | **32 ms** | 27× |
| Thirty questions, Phi-4-mini | out of memory | **fits** | |
| Three questions over HTTP, Qwen3-0.6B, single client | 40 ms | **12 ms** | captured body |
| 32 concurrent clients, Qwen3-0.6B | 94 req/s | **171 req/s** | batched, `--batch 8` |
| 32 clients, one Grade per request (rides the row scorer) | 14 req/s | **70 req/s** | row scorer batched across requests |

The pointer head matches the row scorer on Pick and Claim questions across the real
datasets and trails on Grade, so `ask()` routes each kind to its better path.

## Method

<p align="center"><img src="docs/how-it-works.svg" width="900" alt="state and questions in, batched candidates through one forward pass, a distribution per question out"></p>

**Scoring.** The row scorer appends each allowed answer to the prompt as a candidate and
scores it by its tokens' log-probabilities under the backbone, all candidates in one
batched pass. The pointer head reads the hidden state where the prompt lists each option
and scores it against the hidden state at the end of its question, so a whole request is
one row. A softmax within each question yields its distribution.

**Calibration.** One temperature per question, fit by maximum likelihood with five-fold
cross-fitting, then the lowest confidence cutoff that reaches a target precision on
held-out decisions, with its coverage. Calibration is per question, per model, per data
distribution, and the kit says which set it was fit on.

**Training.** LoRA adapters trained with cross-entropy over the answer set, a strictly
proper scoring rule, on hard or soft targets. A typed head adds an input-conditional
temperature and a per-candidate residual over hidden states, zero-initialised to the
plain scorer. The outcome-reward form replaces labels with the environment's result and
adds the Brier score of the stated probability against it.

**Evaluation.** The gate for any training run is raw calibration error on tasks it never
saw, against the untrained model after temperature scaling. Three tasks and three public
datasets are held out for that purpose.

The [architecture document](docs/ARCHITECTURE.md) explains all of this from nothing; the
[primer](docs/PRIMER.md) derives it, from probability to the policy-gradient objective.

## Getting started

### Install

```sh
pip install torch transformers peft numpy
```

Any open-weight causal language model is a valid backbone. The results above use Qwen3
at three sizes, SmolLM2 and Phi-4-mini with no code changes.

### Ask

```python
from brier import Scorer, Pick, Grade, Claim, ask

scorer = Scorer("microsoft/Phi-4-mini-instruct")
answers = ask(scorer, "Webhooks are firing twice for every event since 3am. Need this today.", [
    Pick("category", "which team should handle this ticket", ("billing", "bug", "feature", "account", "other")),
    Grade("priority", "how fast this must be handled", ("low", "medium", "high", "urgent")),
    Claim("needs_human", "the customer is asking for a human agent rather than an automated reply"),
])

answers["category"].value          # 'bug'
answers["category"].probabilities  # {'billing': 0.07, 'bug': 0.86, 'feature': 0.0, 'account': 0.05, 'other': 0.02}
answers["priority"].expected       # 2.3   (0 = low … 3 = urgent)
answers["needs_human"].value       # 0.06  (probability the claim is true)
```

Three kinds of question cover the space: **Pick** one of several unordered options,
**Grade** on an ordered scale, **Claim** true or false. [QUESTIONS.md](docs/QUESTIONS.md)
is the reference.

### Calibrate

```sh
python3 -m brier calibrate --task tickets --model microsoft/Phi-4-mini-instruct --n 300
```

```
question             n    acc  ECE raw  ECE cal      T
category           300  0.680    0.295    0.054   6.46
  auto    target 0.95: act when confidence >= 0.910  -> precision 1.000, covers 4% of decisions
  review  target 0.80: act when confidence >= 0.721  -> precision 0.800, covers 48% of decisions
  human   everything below
```

The same command takes your own labelled examples. [CALIBRATION.md](docs/CALIBRATION.md)
explains every number.

### Serve

```sh
python3 -m brier.serve --model Qwen/Qwen3-0.6B            # then open http://localhost:8000
python3 -m brier.serve --mock                              # the page with no model
```

The page decides as you type, shows every question's distribution from one pass with its
latency, applies the act / review / defer tiers when a calibration file is loaded, and
plays gridworld live. On an 8 GB Apple-silicon laptop the 0.6B answers a three-question
ticket in about a second; on a GPU with a pointer adapter, `--capture` replays the body
from CUDA graphs and `--batch` lets concurrent requests share a pass:

```sh
python3 -m brier fetch qwen3-0.6b-pointer                 # released adapters, needs the GitHub CLI
python3 -m brier.serve --model Qwen/Qwen3-0.6B --adapter adapters/<the fetched dir> --capture --batch 8
python3 bench_serve.py --clients 32 --n 30                 # throughput as clients see it
```

Released adapters: `qwen3-0.6b-mixed`, `phi-4-mini-mixed`, `qwen3-0.6b-pointer`,
`phi-4-mini-pointer`, `qwen3-0.6b-gridworld-outcome`.

### Train

```sh
python3 train.py --model microsoft/Phi-4-mini-instruct --split train --uncertain 0.3 --steps 1200
python3 train.py --model Qwen/Qwen3-0.6B --pointer --examples 1500 --steps 5000 \
    --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking
python3 rl.py --model Qwen/Qwen3-0.6B --episode 6                                     # outcome reward, no labels
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --adapter adapters/<name> --split eval
```

`torchrun --nproc_per_node 4 train.py …` trains on several cards. GPU work in this
repository runs on rented pods managed by `pod.py`, which enforces a hard time-to-live on
every pod it creates. [TRAINING.md](docs/TRAINING.md) has each round's recipe.

## Repository

| | |
|---|---|
| `brier/questions.py` | Pick, Grade, Claim and `ask()` |
| `brier/scorer.py` | the row scorer, full-row and shared-prefix paths, and the JSON-generation baseline |
| `brier/pointer.py` | the pointer head, CUDA-graph capture, and the request batcher |
| `brier/calibrate.py` · `brier/report.py` | the calibration kit and its figures |
| `brier/head.py` | the typed head |
| `brier/serve.py` | the server behind the demo page |
| `train.py` · `rl.py` · `record.py` | supervised training, outcome-reward training, trajectory recording |
| `tasks/` | eight tasks with computable ground truth or human labels, and twelve public datasets |
| `results/` | every number in the documents, as produced |
| `docs/` | the documents, the demo page, and `assets.py`, which draws every figure from `results/` |

| document | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | what Brier does and why, assuming nothing |
| [PRIMER.md](docs/PRIMER.md) | the mathematics, from distributions to the policy-gradient objective |
| [QUESTIONS.md](docs/QUESTIONS.md) | the API |
| [CALIBRATION.md](docs/CALIBRATION.md) | the kit and how its numbers are computed |
| [TRAINING.md](docs/TRAINING.md) | recipes, protocol, every result table, and what each round taught |
| [MILESTONES.md](docs/MILESTONES.md) | each milestone's question, what it delivered, and what is next |

## Citation

```bibtex
@software{brier2026,
  title  = {Brier: calibrated typed decisions from open language models in one forward pass},
  author = {Dubey, Harsh},
  year   = {2026},
  url    = {https://github.com/hardy30894/brier}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
