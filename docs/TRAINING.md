# Training: milestone 2

Milestone 1 established that an open model's raw scores rank answers well and state
probabilities badly, and that one temperature per question repairs the global part of
that. Milestone 2 asks whether *training* the model on the decision objective does
better: higher accuracy, and calibration that beats the temperature-scaled baseline
without any post-hoc fix, including on a task the model never saw.

This page is the recipe and the measured result. The mathematics is in the
[primer](PRIMER.md), sections 13 to 16.

**Contents**: [The objective](#the-objective) · [What is trained](#what-is-trained) · [Data](#data) · [Running it](#running-it) · [Evaluation and the gate](#evaluation-and-the-gate) · [Round one results](#round-one-results) · [Round one lessons](#what-was-learned-in-round-one) · [Round two protocol](#round-two-a-harder-protocol) · [Round two results](#round-two-results) · [Round two lessons](#what-was-learned-in-round-two) · [Next](#next)

## The objective

For each training example (a state, its questions, their labels), and for each question:

1. Score every allowed answer with the backbone, in one batched pass. This is the same
   `build_rows`, `batch` and `candidate_logprobs` that `ask()` uses at inference.
2. Softmax over the scores to get a distribution over the allowed answers.
3. Loss = −log (probability of the correct answer). When several answers are acceptable
   (gridworld ties), −log of their summed probability.

Averaged over the questions in the example, then over four examples per optimizer step.

<p align="center"><img src="fig-gradient.svg" width="900" alt="the training loop"></p>

This is cross-entropy over the *answer set*, a strictly proper scoring rule. Two things
follow that a plain accuracy objective would not give. The correct answer's probability
goes up, and any systematic overconfidence goes down, because a model that says 0.99
while being right 60% of the time is paying −log(0.01) on 40% of examples and gradient
descent will not tolerate that. There is no separate calibration term: honesty is a
consequence of the rule being proper (primer §10, §15).

## What is trained

LoRA adapters of rank 16 (alpha 32, dropout 0.05) on the seven projection matrices of
every transformer block: `q, k, v, o` in attention and `gate, up, down` in the MLP. The
base weights are frozen. For Qwen3-0.6B that is about 10M trainable parameters; for 4B
about 33M. Adapters are saved to `adapters/<model>-brier-r16-s<steps>/` and loaded onto
the base model at evaluation with `--adapter`.

<p align="center"><img src="fig-lora.svg" width="900" alt="LoRA: a frozen matrix plus a low-rank trainable pair"></p>

Optimizer: AdamW, learning rate 1e-4, 50-step warm-up then linear decay to 5%,
gradient clipping at 1.0, gradient checkpointing on, bfloat16 throughout.

## Data

Four of the five built-in tasks, 600 examples each, generated with a seed disjoint from
the evaluation seed:

| task | question(s) | trained on |
|---|---|---|
| tickets | category (Choice), priority (Score), needs_human (Claim) | yes |
| gridworld | move (Choice, ties allowed), danger (Claim) | yes |
| toolcall | tool (Choice over a per-example subset of eight tools) | yes |
| passages | answers (Claim) | yes |
| frustration | frustration (Score, four levels) | **no**, held out entirely |

The generators know every label because they chose it, so there is no label noise to
learn. Frustration is excluded so that its result measures whether the model learned to
answer typed questions honestly in general, not just these templates.

## Running it

```sh
python3 train.py --model Qwen/Qwen3-0.6B --steps 1500          # ~25 min on an RTX 4090
python3 train.py --model Qwen/Qwen3-4B   --steps 1200          # ~75 min
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/Qwen3-0.6B-brier-r16-s1500 --n 300
```

The evaluation command is the calibration kit with an adapter, so trained and untrained
models are measured by exactly the same code on exactly the same held-out examples.

## Evaluation and the gate

For each model size, the kit is run untrained and trained on 300 examples per task
(seed 7, never seen in training). Three numbers per question decide the gate:

- **ECE raw**, the calibration of the model's own probabilities with no correction. The
  trained model's raw ECE must beat the untrained model's *temperature-scaled* ECE: the
  learned honesty must exceed what one fitted number achieves.
- **ECE after temperature**, cross-fitted, for both. Training should still leave the
  temperature close to 1, which means the model is honest without help.
- **Frustration**, the unseen task. Any improvement there is generalisation.

Accuracy is reported but is not the gate, because on in-distribution examples it can be
inflated by memorising templates.

## Round one results

Untrained → trained, 300 held-out examples per task. "ECE raw" is the model's own
probabilities; "ECE after temp" is cross-fitted temperature scaling on top. The gate
column asks whether the trained model's *raw* calibration beats the untrained model's
*temperature-scaled* calibration.

### Qwen3-0.6B

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.287 → **1.000** | 0.425 → **0.000** | 0.082 → 0.000 | 11.8 → 0.05 | pass |
| tickets | priority | 0.340 → **1.000** | 0.502 → **0.000** | 0.168 → 0.000 | 5.8 → 0.05 | pass |
| tickets | needs_human | 0.310 → **1.000** | 0.570 → **0.000** | 0.261 → 0.000 | 20.0 → 0.05 | pass |
| gridworld | move | 0.353 → **0.840** | 0.488 → **0.123** | 0.095 → 0.131 | 9.4 → 1.79 | no |
| gridworld | danger | 0.870 → **0.977** | 0.155 → **0.012** | 0.012 → 0.015 | 0.5 → 1.06 | no |
| toolcall | tool | 0.883 → **1.000** | 0.064 → **0.000** | 0.036 → 0.000 | 2.0 → 0.05 | pass |
| passages | answers | 0.490 → **0.987** | 0.332 → **0.013** | 0.273 → 0.003 | 3.0 → 2.09 | pass |
| frustration (unseen) | frustration | 0.383 → **0.520** | 0.600 → **0.451** | 0.087 → 0.166 | 11.8 → 5.99 | no |

### Qwen3-1.7B

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.483 → **1.000** | 0.488 → **0.000** | 0.065 → 0.000 | 10.1 → 0.05 | pass |
| tickets | priority | 0.573 → **1.000** | 0.241 → **0.000** | 0.076 → 0.000 | 4.1 → 0.05 | pass |
| tickets | needs_human | 0.363 → **1.000** | 0.631 → **0.000** | 0.322 → 0.000 | 20.0 → 0.05 | pass |
| gridworld | move | 0.437 → **0.853** | 0.457 → **0.112** | 0.100 → 0.134 | 19.3 → 1.73 | no |
| gridworld | danger | 0.130 → **1.000** | 0.855 → **0.000** | 0.436 → 0.000 | 20.0 → 0.05 | pass |
| toolcall | tool | 0.797 → **1.000** | 0.192 → **0.001** | 0.060 → 0.000 | 7.8 → 0.05 | pass |
| passages | answers | 0.477 → **0.997** | 0.521 → **0.003** | 0.284 → 0.002 | 20.0 → 1.28 | pass |
| frustration (unseen) | frustration | 0.563 → **0.767** | 0.372 → **0.216** | 0.188 → 0.091 | 7.2 → 3.96 | no |

### Qwen3-4B

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.680 → **1.000** | 0.295 → **0.000** | 0.054 → 0.000 | 6.5 → 0.05 | pass |
| tickets | priority | 0.493 → **1.000** | 0.422 → **0.000** | 0.092 → 0.000 | 7.8 → 0.05 | pass |
| tickets | needs_human | 0.643 → **1.000** | 0.319 → **0.000** | 0.168 → 0.000 | 10.1 → 0.05 | pass |
| gridworld | move | 0.733 → **0.873** | 0.201 → **0.106** | 0.100 → 0.123 | 5.6 → 1.23 | no |
| gridworld | danger | 0.870 → **1.000** | 0.128 → **0.001** | 0.048 → 0.000 | 5.0 → 0.05 | pass |
| toolcall | tool | 0.970 → **1.000** | 0.028 → **0.000** | 0.008 → 0.000 | 3.4 → 0.05 | pass |
| passages | answers | 0.933 → **0.997** | 0.058 → **0.004** | 0.009 → 0.002 | 4.3 → 1.28 | pass |
| frustration (unseen) | frustration | 0.493 → **0.693** | 0.449 → **0.237** | 0.073 → 0.116 | 9.1 → 3.04 | no |

<p align="center"><img src="../results/rel_frustration_Qwen3-4B.svg" width="440" alt="reliability of the untrained 4B model on frustration"> <img src="../results/rel_frustration_Qwen3-4B-brier-r16-s1200.svg" width="440" alt="reliability of the trained 4B model on frustration"></p>

*Frustration, the task never trained on: untrained Qwen3-4B on the left, trained on the
right. Raw in purple, temperature-scaled in blue.*

## What was learned in round one

**The objective works, and the in-distribution numbers are not the evidence.** On the
four trained tasks every model reaches near-perfect accuracy with a temperature of 0.05,
the bottom of the search grid. That is memorisation: the evaluation examples are new
draws, but from the same finite template pools the model trained on. Those rows show
that the loss does what section 15 of the primer says it does when the answer is
learnable; they say nothing about generalisation. They are reported because hiding them
would be worse, and marked as such.

**Gridworld is the honest in-distribution test.** Grids are random, so there are no
templates to memorise, and the answer is a shortest-path computation. The 0.6B model
goes from 0.37 to 0.84 on `move`, the 4B from 0.73 to 0.87, with raw calibration error
halving. This is a model learning to read a grid and pick a direction from a proper
scoring rule alone, with no chain of thought. At every size the trained raw
calibration on `move` lands within a few hundredths of the temperature-scaled untrained
number without quite beating it, so by the strict gate it is a narrow miss, but the
accuracy gain is large and real.

**The unseen task shows both the gain and the limit.** On frustration, which no model
trained on, accuracy rises by 14 to 20 points at every size and raw calibration error
falls by a quarter to a half, with the fitted temperature dropping by half or more
(from 9 to 3 for the 4B). The model
learned something general about answering typed questions: it is more often right and
much less overconfident. It is still overconfident. Its raw ECE does not beat what one
temperature does for the untrained model, so the strict gate is not met on the task
that matters most.

**Why: clean training data teaches confidence.** Every training label was exact and
every training task became solvable, so the model saw a world in which it was always
right and learned to say so. Nothing in the data taught it what a genuinely uncertain
case looks like. Gridworld ties were the only source of "several answers are
acceptable," and they are not the same as "the answer is unknowable from this state."
A proper scoring rule can only teach honesty about uncertainty that exists in the
training distribution. The next training set needs irreducible uncertainty on purpose:
ambiguous tickets with soft labels, passages that partially answer the question, states
where the generator itself is unsure, and labels distilled from a stronger model with
their disagreements kept rather than resolved.

**Size matters less than expected.** The 0.6B model trained reaches accuracies on
gridworld and frustration that the untrained 4B does not. For decision tasks with a
clear objective, a small adapted model beats a larger unadapted one, at a quarter of
the latency.

**The kit did its job.** Every claim above is a held-out number the calibration kit
produced, and the gate refused to pass where the numbers did not support it. That is
the point of having the measurement before the training.

## Round two: a harder protocol

The first round's in-distribution numbers were memorisation, and its unseen-task
calibration missed the gate because the training data contained no uncertainty. Round
two changes the protocol in four ways, all now in the code:

- **Template hold-out.** Every templated task splits its template pools in two.
  Training draws from one half (`--split train`), evaluation from the other
  (`--split eval`), so an evaluation example never shares a sentence template with
  anything the model trained on.
- **Uncertainty on purpose.** A fraction of training examples (`--uncertain 0.3`) is
  built to be genuinely ambiguous, with a soft label equal to the generator's own
  posterior: two categories' sentences mixed (0.5 each), a ticket with no priority
  sentence (the prior over priorities), a request that names two tools, a passage that
  names the topic without the value, adjacent frustration levels joined. The loss is
  cross-entropy against the soft target, still a proper scoring rule, and now the model
  is shown what "not knowable from this input" looks like.
- **A second unseen task**, `duplicates`: do two contact records describe the same
  person, with variants (initials, email formats, typos) as positives and near-misses
  (same company, same surname) as negatives. Never trained on, like frustration.
- **A real dataset**, `banking`: customer banking queries with human intent labels
  (PolyAI banking77), restricted to ten intents. Real text, real label noise, never
  trained on.

And one new training mode, `rl.py`: outcome-reward training on gridworld with no
labels. The model states a distribution over moves, one move is sampled and executed,
and the environment reports whether the player got closer to the exit. The loss is a
policy-gradient term on that outcome plus the Brier score of the stated probability
against it. A caveat that the environment makes plain: for a single step, "got closer"
is exactly the optimal-move label, so this is reinforcement learning in mechanism (the
signal is for the chosen action only, on the model's own decisions) and not yet in
information content. Multi-step episodes with delayed outcomes would change that.

## Round two results

Every number below is on the **eval template split**: no evaluation example shares a
sentence template with training. Frustration, duplicates and banking were never trained
on. Tables are generated by `docs/results_table2.py`.

### Backbones, untrained (accuracy / ECE raw → ECE after temperature)

| task | question | Qwen3-0.6B | Qwen3-1.7B | SmolLM2-1.7B-Instruct | Phi-4-mini-instruct | Qwen3-4B |
|---|---|---|---|---|---|---|
| tickets | category | 0.23 / 0.48 → 0.03 | 0.33 / 0.61 → 0.13 | 0.25 / 0.12 → 0.07 | 0.71 / 0.12 → 0.11 | 0.60 / 0.35 → 0.08 |
| tickets | priority | 0.40 / 0.38 → 0.13 | 0.63 / 0.20 → 0.06 | 0.20 / 0.34 → 0.07 | 0.61 / 0.16 → 0.07 | 0.35 / 0.58 → 0.09 |
| tickets | needs_human | 0.32 / 0.54 → 0.39 | 0.40 / 0.59 → 0.28 | 0.31 / 0.66 → 0.24 | 0.67 / 0.18 → 0.08 | 0.61 / 0.36 → 0.22 |
| gridworld | move | 0.35 / 0.49 → 0.09 | 0.44 / 0.46 → 0.10 | 0.36 / 0.05 → 0.09 | 0.63 / 0.08 → 0.09 | 0.73 / 0.20 → 0.10 |
| gridworld | danger | 0.87 / 0.15 → 0.01 | 0.13 / 0.85 → 0.44 | 0.13 / 0.71 → 0.39 | 0.87 / 0.08 → 0.05 | 0.87 / 0.13 → 0.05 |
| toolcall | tool | 0.94 / 0.03 → 0.02 | 0.86 / 0.13 → 0.05 | 0.61 / 0.10 → 0.04 | 0.96 / 0.02 → 0.02 | 0.99 / 0.01 → 0.01 |
| passages | answers | 0.50 / 0.35 → 0.32 | 0.50 / 0.50 → 0.33 | 0.50 / 0.42 → 0.03 | 0.88 / 0.12 → 0.09 | 0.88 / 0.12 → 0.09 |
| frustration | frustration | 0.38 / 0.62 → 0.23 | 0.60 / 0.38 → 0.10 | 0.27 / 0.30 → 0.11 | 0.52 / 0.34 → 0.15 | 0.44 / 0.54 → 0.31 |
| duplicates | same | 0.86 / 0.10 → 0.06 | 0.96 / 0.03 → 0.02 | 0.52 / 0.45 → 0.03 | 0.95 / 0.03 → 0.02 | 0.73 / 0.26 → 0.14 |
| banking | intent | 0.85 / 0.10 → 0.03 | 0.94 / 0.06 → 0.03 | 0.61 / 0.15 → 0.13 | 0.94 / 0.03 → 0.02 | 0.96 / 0.03 → 0.02 |

### Qwen3-0.6B: untrained → trained on held-out templates, 30% ambiguous training examples

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.227 → **0.703** | 0.477 → **0.195** | 0.032 → 0.074 | 13.7 → 2.33 | no |
| tickets | priority | 0.397 → **0.727** | 0.382 → **0.074** | 0.127 → 0.069 | 2.9 → 0.65 | pass |
| tickets | needs_human | 0.320 → **1.000** | 0.543 → **0.000** | 0.388 → 0.000 | 10.5 → 0.05 | pass |
| gridworld | move | 0.353 → **0.837** | 0.488 → **0.119** | 0.095 → 0.112 | 9.4 → 1.86 | no |
| gridworld | danger | 0.870 → **1.000** | 0.155 → **0.000** | 0.012 → 0.000 | 0.5 → 0.05 | pass |
| toolcall | tool | 0.943 → **1.000** | 0.029 → **0.003** | 0.022 → 0.000 | 0.9 → 0.05 | pass |
| passages | answers | 0.503 → **0.877** | 0.345 → **0.104** | 0.320 → 0.096 | 2.4 → 2.33 | pass |
| frustration (unseen) | frustration | 0.377 → **0.443** | 0.618 → **0.397** | 0.234 → 0.102 | 14.8 → 5.15 | no |
| duplicates (unseen) | same | 0.863 → **0.643** | 0.095 → **0.275** | 0.058 → 0.289 | 0.5 → 2.42 | no |
| banking (unseen) | intent | 0.847 → **0.900** | 0.095 → **0.025** | 0.025 → 0.035 | 2.2 → 1.33 | pass |

### Qwen3-1.7B: untrained → trained on held-out templates, 30% ambiguous training examples

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.333 → **0.820** | 0.612 → **0.097** | 0.129 → 0.045 | 11.8 → 2.01 | pass |
| tickets | priority | 0.627 → **0.680** | 0.205 → **0.127** | 0.060 → 0.104 | 2.8 → 1.23 | no |
| tickets | needs_human | 0.400 → **1.000** | 0.588 → **0.000** | 0.280 → 0.000 | 20.0 → 0.05 | pass |
| gridworld | move | 0.437 → **0.847** | 0.457 → **0.102** | 0.100 → 0.138 | 19.3 → 1.93 | no |
| gridworld | danger | 0.130 → **1.000** | 0.855 → **0.000** | 0.436 → 0.000 | 20.0 → 0.05 | pass |
| toolcall | tool | 0.860 → **1.000** | 0.129 → **0.003** | 0.047 → 0.000 | 7.0 → 0.05 | pass |
| passages | answers | 0.500 → **0.877** | 0.496 → **0.109** | 0.334 → 0.092 | 20.0 → 2.71 | pass |
| frustration (unseen) | frustration | 0.603 → **0.620** | 0.377 → **0.226** | 0.101 → 0.223 | 8.1 → 1.73 | no |
| duplicates (unseen) | same | 0.957 → **0.987** | 0.026 → **0.022** | 0.022 → 0.013 | 1.5 → 0.50 | no |
| banking (unseen) | intent | 0.943 → **0.947** | 0.056 → **0.018** | 0.028 → 0.018 | 5.6 → 1.49 | pass |

### Qwen3-4B: untrained → trained on held-out templates, 30% ambiguous training examples

| task | question | accuracy | ECE raw | ECE after temp | T | raw beats untrained+temp |
|---|---|---|---|---|---|---|
| tickets | category | 0.603 → **0.870** | 0.349 → **0.072** | 0.081 → 0.020 | 6.5 → 1.86 | pass |
| tickets | priority | 0.350 → **0.627** | 0.577 → **0.249** | 0.088 → 0.134 | 20.0 → 2.93 | no |
| tickets | needs_human | 0.607 → **1.000** | 0.363 → **0.000** | 0.217 → 0.000 | 10.9 → 0.05 | pass |
| gridworld | move | 0.733 → **0.873** | 0.201 → **0.107** | 0.100 → 0.116 | 5.6 → 1.33 | no |
| gridworld | danger | 0.870 → **1.000** | 0.128 → **0.000** | 0.048 → 0.000 | 5.0 → 0.05 | pass |
| toolcall | tool | 0.993 → **1.000** | 0.008 → **0.000** | 0.007 → 0.000 | 2.6 → 0.05 | pass |
| passages | answers | 0.877 → **0.877** | 0.123 → **0.093** | 0.089 → 0.074 | 6.5 → 2.52 | no |
| frustration (unseen) | frustration | 0.440 → **0.633** | 0.537 → **0.247** | 0.308 → 0.166 | 8.1 → 2.01 | pass |
| duplicates (unseen) | same | 0.733 → **0.997** | 0.263 → **0.037** | 0.136 → 0.005 | 13.2 → 0.15 | pass |
| banking (unseen) | intent | 0.963 → **0.950** | 0.031 → **0.033** | 0.020 → 0.024 | 4.0 → 1.66 | no |

### Qwen3-0.6B on gridworld: supervised versus outcome reward

| run | move accuracy | move ECE raw | move ECE after temp | frustration accuracy | frustration ECE raw |
|---|---|---|---|---|---|
| untrained | 0.353 | 0.488 | 0.095 | 0.377 | 0.618 |
| supervised (labels) | 0.837 | 0.119 | 0.112 | 0.443 | 0.397 |
| outcome reward from base | 0.813 | 0.183 | 0.093 | 0.247 | 0.730 |
| outcome reward from supervised | 0.820 | 0.123 | 0.153 | 0.577 | 0.357 |

## What was learned in round two

**Template hold-out removed the memorisation.** With templates held out, the trained
rows are no longer a wall of 1.000. Category goes to 0.70, priority to 0.73, passages
to 0.88 on the 0.6B. Three questions still reach 1.000 (needs_human, danger, tool)
because their answers are single cues the model can learn from either template half.
Those rows are now believable.

**Ambiguous training examples fixed the overconfidence, mostly.** In round one the
fitted temperatures after training were 0.05, the bottom of the grid: the model was
maximally sharp. With 30% of training examples carrying soft labels equal to the
generator's own posterior, the temperatures land between 0.7 and 2.3 on most questions,
and raw calibration error on the held-out tasks falls to within a few hundredths of the
temperature-scaled number. On the real banking data the strict gate passes at 0.6B:
accuracy 0.85 to 0.90, raw ECE 0.095 to 0.025, with no post-hoc fix. That is the result
round one was missing, and the change that produced it was data, not the objective.

**Real data behaves.** Banking77 is human-written text with human labels and their
noise. The untrained 0.6B already routes ten intents at 0.85; trained on synthetic
tasks only, it reaches 0.90 and stays honest. The automatic tier at 95% precision
covers 82% of queries before training. This is the shape of most production decision
workloads, and it is the strongest argument for the toolkit.

**Outcome reward works without labels.** Trained from the base model on nothing but
the environment's "did the player get closer" signal, the 0.6B reaches 0.81 on gridworld
against 0.84 supervised, with raw calibration error 0.18 and 0.09 after temperature.
Started from the supervised adapter it reaches 0.82 and, unexpectedly, lifts the
never-trained frustration task from 0.44 to 0.58; that transfer is one run on 300
examples and needs replication before it is claimed.

**At 4B the unseen gate passes.** Frustration goes from 0.44 to 0.63 with raw ECE 0.54
to 0.25, under the untrained temperature-scaled 0.31. Duplicates goes from 0.73 to 0.997
with raw ECE 0.26 to 0.04. Two of the three never-trained tasks clear the strict gate
with no post-hoc correction; the third, banking, was already at 0.96 accuracy and 0.03
ECE untrained and stays there. This is the milestone round one set: a model trained on
synthetic decisions is more accurate and more honest, raw, on tasks it never saw.

**At 1.7B the picture is in between.** Banking passes the gate (raw ECE 0.056 to 0.018
at 0.947 accuracy), duplicates stays high and honest at 0.99 with no negative transfer,
and frustration improves in raw calibration without clearing the bar.

**At 0.6B, transfer can go the wrong way.** The same duplicates task got worse after
training the small model: 0.86 to 0.64. The confusion matrix says why: it answers "same
person" 89% of the time against a true rate of 52%, false positives 28 to 74, misses
zero. Every Claim in the training set is cue-triggered, present cue means yes, so the
small model learned that shared surface details imply yes, and duplicates is built from
near-misses that share details on purpose. The 4B learned the finer rule from the same
data. Two lessons: capacity decides whether a model extracts the general rule or the
shortcut, and a training set needs contrastive claims, where overlap is misleading, if
small models are to be trusted on unseen claims.

**Frustration is the hard one for small models.** At 0.6B accuracy rises and raw ECE
falls but the gate is not met; at 4B it passes. A four-level ordinal judgement from one
sentence is ambiguous enough that a small model cannot separate its own uncertainty from
the task's, and temperature alone is very effective on a task where a model is uniformly
unsure, which makes the gate a high bar.

**Backbone independence is measured, and it changes the default.** SmolLM2-1.7B and
Phi-4-mini ran through the identical pipeline with no code changes. Untrained,
Phi-4-mini (3.8B) is the strongest small backbone in the table: it beats Qwen3-4B on
ticket category (0.71 against 0.60), needs_human (0.67 against 0.61) and frustration
(0.52 against 0.44), matches it on passages and banking, and its raw calibration errors
are the lowest of the five before any temperature. SmolLM2 is weak on these tasks and
badly biased on danger. Qwen3-1.7B is oddly worse than Qwen3-0.6B on danger (0.13, it
answers "yes" almost always) while better elsewhere, a reminder that untrained
behaviour on a binary claim is mostly prior. The working backbone should be re-chosen
by this table, not by habit; Phi-4-mini is the candidate to train next.

**Shared-prefix scoring is exact and pays from 1.7B up.** See
[ARCHITECTURE.md §2.8](ARCHITECTURE.md#28-sharing-the-prompt-across-candidates).

## Round three: contrastive data, a typed head, another backbone, longer episodes

Four changes, run in parallel on four pods:

- **Contrastive claims in training.** A new training task, `matching` (does an invoice
  line match an order line, where negatives share the product or the total on purpose),
  and contrastive negatives in tickets ("no need for a person, an automated reply is
  fine") so that `needs_human` cannot be learned from the word "person". This targets
  the round-two failure where the 0.6B learned that shared cues mean yes.
- **A typed head.** `brier/head.py`: two corrections over the LM-head scores, an
  input-conditional temperature from the state's hidden representation and a
  per-candidate residual from the candidate's, all parameters starting at zero so the
  untrained head is exactly the plain scorer (verified: max difference 0.0). Trained
  jointly with LoRA. This is the first piece of the model that reads hidden states
  rather than the vocabulary, and it gives the model a way to be more or less sure
  per input, which a single temperature cannot.
- **Phi-4-mini trained.** The strongest untrained backbone in round two, put through the
  same recipe. LoRA targets are now detected per architecture (Phi fuses the q, k, v
  projections into one matrix).
- **Six-step episodes.** `rl.py --episode 6`: the reward for every move in an episode is
  whether the episode reached the exit, a delayed outcome that single-step training
  never sees. And `record.py` writes trajectories with per-step distributions for the
  [demo page](demo.html).

ROUND3_TABLES

## What was learned in round three

**Contrastive data cured the negative transfer.** With the matching task and the
not-human phrases in training, and nothing else changed, the 0.6B's duplicates result
goes from 0.64 (round two, worse than untrained) to 0.97, with raw calibration error
0.27 to 0.03. The model no longer learns that shared cues mean yes, because the training
set now contains claims where shared cues mean no. Category also improves, 0.70 to 0.82.
This is the cleanest demonstration in the project that the model learns exactly the
decision rules its data contains, and that the fix for a shortcut is a counterexample.

**The typed head is a calibration tool, not an accuracy tool, at 0.6B.** Joint training
with the head lowers raw calibration error on the unseen frustration task from 0.44 to
0.25, which is the input-conditional temperature doing its job on a task where the
model should be unsure. But the per-candidate residual costs accuracy on the ordinal
and many-way questions: priority 0.76 to 0.47, banking 0.89 to 0.83. At this size the
residual term has more capacity to overfit the training tasks than the LoRA adapters
do. The natural next configuration is the temperature term alone, or the residual with
a much smaller learning rate. HEAD_4B_SENTENCE

**Phi-4-mini trained.** PHI_SENTENCE

**Six-step episodes.** RL_SENTENCE


## Next

- **Outcome rewards.** Gridworld is an environment, not just a dataset. The same
  adapter can be updated from the Brier score of the stated probability against the
  observed outcome of the chosen action, with no labels. Same fixed point (honesty),
  different data source.
- **Typed heads.** Replace the LM head's role in scoring with a small head reading the
  hidden states, initialised to reproduce the LM-head scores and then trained. Removes
  the tokenization and prompt-format dependence.
- **Throughput.** The runs below used one example per forward pass and under half of a
  4090 at 0.6B. `--batch N` now packs N examples' candidate rows into one pass; it was
  added after these runs and is not reflected in their timings.
- **Real data.** Every number here is on synthetic tasks with clean labels. The next
  training run should include distilled labels on real tickets or real tool-call logs,
  where the label noise will show up as calibrated uncertainty rather than as accuracy.
