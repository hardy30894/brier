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
| tickets | category (Pick), priority (Grade), needs_human (Claim) | yes |
| gridworld | move (Pick, ties allowed), danger (Claim) | yes |
| toolcall | tool (Pick over a per-example subset of eight tools) | yes |
| passages | answers (Claim) | yes |
| frustration | frustration (Grade, four levels) | **no**, held out entirely |

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

Cells are accuracy / ECE raw → ECE after temperature, on the eval template split, 300 examples per task. Tables are generated by `docs/results_table3.py`.

### Qwen3-0.6B: round two → round three (contrastive data) → round three + typed head

| task | question | untrained | round 2 | round 3 | round 3 + head |
|---|---|---|---|---|---|
| tickets | category | 0.23 / 0.48 → 0.03 | 0.70 / 0.20 → 0.07 | 0.82 / 0.11 → 0.09 | 0.86 / 0.18 → 0.09 |
| tickets | priority | 0.40 / 0.38 → 0.13 | 0.73 / 0.07 → 0.07 | 0.76 / 0.19 → 0.16 | 0.47 / 0.16 → 0.12 |
| tickets | needs_human | 0.32 / 0.54 → 0.39 | 1.00 / 0.00 → 0.00 | 0.99 / 0.01 → 0.00 | 0.98 / 0.02 → 0.01 |
| gridworld | move | 0.35 / 0.49 → 0.09 | 0.84 / 0.12 → 0.11 | 0.84 / 0.13 → 0.12 | 0.83 / 0.12 → 0.13 |
| gridworld | danger | 0.87 / 0.15 → 0.01 | 1.00 / 0.00 → 0.00 | 0.95 / 0.02 → 0.02 | 0.91 / 0.04 → 0.02 |
| toolcall | tool | 0.94 / 0.03 → 0.02 | 1.00 / 0.00 → 0.00 | 1.00 / 0.01 → 0.00 | 0.98 / 0.02 → 0.02 |
| passages | answers | 0.50 / 0.35 → 0.32 | 0.88 / 0.10 → 0.10 | 0.87 / 0.08 → 0.03 | 0.88 / 0.13 → 0.08 |
| matching | matches | 0.63 / 0.21 → 0.22 |  | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 |
| frustration (unseen) | frustration | 0.38 / 0.62 → 0.23 | 0.44 / 0.40 → 0.10 | 0.46 / 0.44 → 0.13 | 0.43 / 0.25 → 0.20 |
| duplicates (unseen) | same | 0.86 / 0.10 → 0.06 | 0.64 / 0.27 → 0.29 | 0.97 / 0.03 → 0.02 | 0.98 / 0.02 → 0.01 |
| banking (unseen) | intent | 0.85 / 0.10 → 0.03 | 0.90 / 0.02 → 0.03 | 0.89 / 0.06 → 0.03 | 0.83 / 0.07 → 0.05 |

### Qwen3-4B: round two → round three → round three + typed head

| task | question | untrained | round 2 | round 3 | round 3 + head |
|---|---|---|---|---|---|
| tickets | category | 0.60 / 0.35 → 0.08 | 0.87 / 0.07 → 0.02 | 0.85 / 0.12 → 0.09 | 0.87 / 0.10 → 0.03 |
| tickets | priority | 0.35 / 0.58 → 0.09 | 0.63 / 0.25 → 0.13 | 0.59 / 0.28 → 0.20 | 0.65 / 0.33 → 0.07 |
| tickets | needs_human | 0.61 / 0.36 → 0.22 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 |
| gridworld | move | 0.73 / 0.20 → 0.10 | 0.87 / 0.11 → 0.12 | 0.87 / 0.11 → 0.12 | 0.84 / 0.07 → 0.08 |
| gridworld | danger | 0.87 / 0.13 → 0.05 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 |
| toolcall | tool | 0.99 / 0.01 → 0.01 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 |
| passages | answers | 0.88 / 0.12 → 0.09 | 0.88 / 0.09 → 0.07 | 0.89 / 0.06 → 0.05 | 0.93 / 0.01 → 0.00 |
| matching | matches | 0.76 / 0.23 → 0.11 |  | 1.00 / 0.00 → 0.00 | 1.00 / 0.00 → 0.00 |
| frustration (unseen) | frustration | 0.44 / 0.54 → 0.31 | 0.63 / 0.25 → 0.17 | 0.83 / 0.20 → 0.20 | 0.65 / 0.24 → 0.16 |
| duplicates (unseen) | same | 0.73 / 0.26 → 0.14 | 1.00 / 0.04 → 0.01 | 0.97 / 0.02 → 0.01 | 0.98 / 0.02 → 0.02 |
| banking (unseen) | intent | 0.96 / 0.03 → 0.02 | 0.95 / 0.03 → 0.02 | 0.96 / 0.03 → 0.03 | 0.96 / 0.02 → 0.03 |

### Phi-4-mini: untrained → trained (round three recipe)

| task | question | untrained | trained |
|---|---|---|---|
| tickets | category | 0.71 / 0.12 → 0.11 | 0.96 / 0.04 → 0.02 |
| tickets | priority | 0.61 / 0.16 → 0.07 | 0.79 / 0.20 → 0.21 |
| tickets | needs_human | 0.67 / 0.18 → 0.08 | 1.00 / 0.00 → 0.00 |
| gridworld | move | 0.63 / 0.08 → 0.09 | 0.85 / 0.10 → 0.10 |
| gridworld | danger | 0.87 / 0.08 → 0.05 | 0.99 / 0.01 → 0.00 |
| toolcall | tool | 0.96 / 0.02 → 0.02 | 1.00 / 0.00 → 0.00 |
| passages | answers | 0.88 / 0.12 → 0.09 | 0.88 / 0.12 → 0.10 |
| matching | matches | 0.82 / 0.10 → 0.09 | 1.00 / 0.00 → 0.00 |
| frustration (unseen) | frustration | 0.52 / 0.34 → 0.15 | 0.58 / 0.23 → 0.26 |
| duplicates (unseen) | same | 0.95 / 0.03 → 0.02 | 0.99 / 0.02 → 0.01 |
| banking (unseen) | intent | 0.94 / 0.03 → 0.02 | 0.92 / 0.05 → 0.04 |

### Qwen3-0.6B on gridworld: single-step versus six-step outcome episodes

| run | move accuracy | move ECE raw | move ECE after temp | frustration accuracy | frustration ECE raw |
|---|---|---|---|---|---|
| untrained | 0.353 | 0.488 | 0.095 | 0.377 | 0.618 |
| supervised, round 2 | 0.837 | 0.119 | 0.112 | 0.443 | 0.397 |
| outcome reward, 1 step | 0.813 | 0.183 | 0.093 | 0.247 | 0.730 |
| outcome reward, 6-step episodes | 0.813 | 0.280 | 0.174 | 0.330 | 0.506 |

## What was learned in round three

**Contrastive data cured the negative transfer.** With the matching task and the
not-human phrases in training, and nothing else changed, the 0.6B's duplicates result
goes from 0.64 (round two, worse than untrained) to 0.97, with raw calibration error
0.27 to 0.03. The model no longer learns that shared cues mean yes, because the training
set now contains claims where shared cues mean no. Category also improves, 0.70 to 0.82.
This is the cleanest demonstration in the project that the model learns exactly the
decision rules its data contains, and that the fix for a shortcut is a counterexample.

**Contrastive data also lifted the hardest unseen task at 4B.** With the matching task
and the not-human phrases added and nothing else changed, the 4B's frustration accuracy
goes from 0.63 in round two to 0.83, with raw calibration error 0.25 to 0.20. Two
training tasks whose negatives share surface features with their positives taught a
general habit of reading past the cue, and a four-level ordinal judgement the model
never trained on is where that habit paid most.

**The typed head is a calibration tool, and it has a cost.** Compared with the same
data and no head, the head lowers raw calibration error on the trained tasks (passages
0.06 to 0.01 at 4B, 0.13 to 0.08 at 0.6B) and helps the ordinal priority question at 4B
(0.59 to 0.65). It also hurts the unseen ordinal task: frustration at 4B is 0.83 without
the head and 0.65 with it, and at 0.6B the residual term costs priority 0.76 to 0.47 and
banking 0.89 to 0.83. The input-conditional temperature is doing what it was designed
to do; the per-candidate residual has enough capacity to fit the training tasks'
answer sets and carries that fit to tasks it should not. The head therefore stays
optional: on by evaluation, not by default, and the next configuration to test is the
temperature term alone.

**Phi-4-mini trained is the strongest model in the project.** The best untrained
backbone, put through the round-three recipe, reaches 0.96 on ticket category (the
0.6B reaches 0.82 and the 4B 0.87 in round two), 0.79 on priority, 0.85 on gridworld
moves, and on the never-trained tasks 0.99 on duplicates, 0.58 on frustration with raw
calibration error down from 0.34 to 0.23, and 0.92 on banking. Nothing in the pipeline
changed for it except the automatic detection of its fused projection matrices for
LoRA. It should be the working backbone from here.

**Six-step episodes teach the same policy from a weaker signal.** Rewarding every move
in an episode with whether the episode reached the exit, the 0.6B reaches 0.813 on
gridworld, identical to single-step outcome reward and three points under supervised,
from a reward that arrives up to six moves late and is shared by good and bad moves
alike. Its raw calibration is worse (0.28 against 0.18 single-step), which is what
credit assignment noise looks like: the model cannot tell which of its six stated
probabilities the outcome was about. Discounting or a value estimate is the standard
fix and the next item on the outcome-reward list. Training took 45 minutes on one 4090
against 11 for single-step, because each step plays out six sequential forward passes.


## Round four: the generic corpus

The question this round asks is the one the project rests on: does the objective learn
real decisions in general, or only the synthetic ones it was built on? The answer needs
real labelled data from many domains, whole domains held out, and one adapter trained
on the mixture.

**The corpus.** Twelve public datasets, streamed from the Hub, each mapped onto one or
two questions in our three kinds (`tasks/public.py`):

| kind | datasets |
|---|---|
| Pick | AG News topics, DBpedia entity types, SNLI and MNLI relation, Emotion, MASSIVE intents (ten), banking77 intents (ten) |
| Grade | SST-5 sentiment, Yelp star rating |
| Claim | SNLI and MNLI entailment, Quora duplicate questions, Civil Comments toxicity, MS MARCO passage relevance, IMDB recommendation |

Civil Comments carries the fraction of annotators who flagged a comment, which becomes a
soft label in training: the first real source of irreducible uncertainty in the data.

**The protocol.** Training mixes the five synthetic tasks with ten of the public sets,
600 examples each, 30% ambiguous synthetic examples as before. Three whole public
datasets (MNLI, Emotion, IMDB) and the two synthetic unseen tasks (frustration,
duplicates) are never trained on. Every task is evaluated on 300 examples from its own
evaluation split, raw, and the adapter is compared with the untrained backbone and the
synthetic-only adapter from round three on exactly the same examples.

**The runs.** One adapter per backbone at 0.6B and Phi-4-mini on single GPUs, and the
Phi adapter again on four GPUs with the distributed trainer, same examples, to measure
the speedup: 8,000 examples in 16 minutes on four L40S against about 39 minutes for the
same job on one 4090.

Cells are accuracy / raw ECE on 300 examples from each dataset's own evaluation split; no calibration file applied. Generated by `docs/results_table4.py`.

### Qwen3-0.6B: accuracy / raw ECE, 300 held-out examples per task

| task | question | untrained | synthetic adapter | mixed adapter | mixed, 4 GPUs |
|---|---|---|---|---|---|
| tickets | category | 0.23 / 0.47 | 0.82 / 0.11 | 0.86 / 0.06 |  |
| tickets | priority | 0.39 / 0.42 | 0.60 / 0.14 | 0.90 / 0.20 |  |
| tickets | needs_human | 0.31 / 0.55 | 0.99 / 0.01 | 0.97 / 0.03 |  |
| gridworld | move | 0.35 / 0.49 | 0.83 / 0.14 | 0.84 / 0.06 |  |
| gridworld | danger | 0.87 / 0.15 | 1.00 / 0.00 | 0.95 / 0.04 |  |
| toolcall | tool | 0.94 / 0.03 | 0.99 / 0.01 | 1.00 / 0.00 |  |
| passages | answers | 0.50 / 0.35 | 0.87 / 0.09 | 0.95 / 0.06 |  |
| matching | matches | 0.63 / 0.21 | 1.00 / 0.00 | 1.00 / 0.00 |  |
| ag_news | topic | 0.75 / 0.21 | 0.65 / 0.25 | 0.87 / 0.09 |  |
| dbpedia | kind | 0.74 / 0.17 | 0.90 / 0.03 | 0.97 / 0.03 |  |
| sst5 | sentiment | 0.34 / 0.23 | 0.35 / 0.42 | 0.51 / 0.11 |  |
| yelp | stars | 0.27 / 0.32 | 0.43 / 0.14 | 0.58 / 0.12 |  |
| snli | relation | 0.33 / 0.58 | 0.52 / 0.34 | 0.84 / 0.10 |  |
| snli | entailed | 0.38 / 0.47 | 0.89 / 0.07 | 0.92 / 0.06 |  |
| qqp | duplicate | 0.40 / 0.46 | 0.72 / 0.20 | 0.86 / 0.07 |  |
| civil | toxic | 0.82 / 0.09 | 0.90 / 0.06 | 0.94 / 0.01 |  |
| msmarco | answers | 0.12 / 0.83 | 0.38 / 0.37 | 0.88 / 0.07 |  |
| massive | intent | 0.71 / 0.18 | 0.70 / 0.16 | 0.81 / 0.13 |  |
| banking | intent | 0.85 / 0.10 | 0.90 / 0.02 | 0.98 / 0.02 |  |
| mnli (held out) | relation | 0.38 / 0.56 | 0.64 / 0.21 | 0.77 / 0.11 |  |
| mnli (held out) | entailed | 0.44 / 0.38 | 0.85 / 0.05 | 0.88 / 0.05 |  |
| emotion (held out) | emotion | 0.47 / 0.40 | 0.53 / 0.30 | 0.55 / 0.20 |  |
| imdb (held out) | positive | 0.88 / 0.04 | 0.87 / 0.06 | 0.97 / 0.02 |  |
| frustration (held out) | frustration | 0.38 / 0.62 | 0.43 / 0.46 | 0.46 / 0.36 |  |
| duplicates (held out) | same | 0.86 / 0.10 | 1.00 / 0.03 | 0.99 / 0.04 |  |

### Phi-4-mini: accuracy / raw ECE, 300 held-out examples per task

| task | question | untrained | synthetic adapter | mixed adapter | mixed, 4 GPUs |
|---|---|---|---|---|---|
| tickets | category | 0.71 / 0.14 | 0.96 / 0.03 | 0.91 / 0.02 | 0.87 / 0.06 |
| tickets | priority | 0.60 / 0.17 | 0.81 / 0.17 | 0.79 / 0.04 | 0.79 / 0.06 |
| tickets | needs_human | 0.72 / 0.14 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| gridworld | move | 0.63 / 0.08 | 0.85 / 0.13 | 0.84 / 0.12 | 0.83 / 0.18 |
| gridworld | danger | 0.87 / 0.08 | 0.99 / 0.01 | 0.98 / 0.02 | 0.92 / 0.03 |
| toolcall | tool | 0.96 / 0.02 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| passages | answers | 0.88 / 0.12 | 0.88 / 0.12 | 0.88 / 0.10 | 0.88 / 0.10 |
| matching | matches | 0.82 / 0.10 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.01 |
| ag_news | topic | 0.78 / 0.18 | 0.79 / 0.16 | 0.91 / 0.05 | 0.90 / 0.03 |
| dbpedia | kind | 0.87 / 0.09 | 0.92 / 0.04 | 1.00 / 0.00 | 0.99 / 0.01 |
| sst5 | sentiment | 0.44 / 0.36 | 0.45 / 0.22 | 0.55 / 0.13 | 0.52 / 0.06 |
| yelp | stars | 0.62 / 0.23 | 0.62 / 0.14 | 0.65 / 0.15 | 0.64 / 0.09 |
| snli | relation | 0.87 / 0.07 | 0.84 / 0.07 | 0.88 / 0.08 | 0.90 / 0.05 |
| snli | entailed | 0.94 / 0.02 | 0.93 / 0.02 | 0.96 / 0.02 | 0.96 / 0.01 |
| qqp | duplicate | 0.60 / 0.30 | 0.86 / 0.05 | 0.86 / 0.07 | 0.87 / 0.03 |
| civil | toxic | 0.49 / 0.33 | 0.84 / 0.06 | 0.94 / 0.03 | 0.94 / 0.02 |
| msmarco | answers | 0.26 / 0.69 | 0.30 / 0.59 | 0.88 / 0.04 | 0.87 / 0.04 |
| massive | intent | 0.79 / 0.11 | 0.80 / 0.12 | 0.81 / 0.13 | 0.82 / 0.11 |
| banking | intent | 0.94 / 0.03 | 0.92 / 0.05 | 0.97 / 0.02 | 0.98 / 0.02 |
| mnli (held out) | relation | 0.78 / 0.12 | 0.81 / 0.11 | 0.84 / 0.08 | 0.80 / 0.06 |
| mnli (held out) | entailed | 0.89 / 0.05 | 0.90 / 0.04 | 0.91 / 0.04 | 0.89 / 0.05 |
| emotion (held out) | emotion | 0.60 / 0.27 | 0.63 / 0.24 | 0.60 / 0.20 | 0.59 / 0.23 |
| imdb (held out) | positive | 0.98 / 0.01 | 0.96 / 0.02 | 1.00 / 0.01 | 1.00 / 0.01 |
| frustration (held out) | frustration | 0.52 / 0.34 | 0.62 / 0.24 | 0.67 / 0.21 | 0.63 / 0.22 |
| duplicates (held out) | same | 0.95 / 0.03 | 0.98 / 0.03 | 0.95 / 0.03 | 0.95 / 0.06 |

## What was learned in round four

**The objective learns real decisions, and generally.** With ten real datasets in the
mix, the 0.6B goes from 0.12 to 0.88 on MS MARCO passage relevance, 0.33 to 0.84 on SNLI,
0.40 to 0.86 on Quora duplicates, 0.27 to 0.58 on Yelp stars, and 0.85 to 0.98 on
banking77. Phi-4-mini reaches 0.87 on MS MARCO from 0.26, 0.94 on toxicity from 0.49,
0.99 on DBpedia, 0.98 on banking. These are held-out examples from the datasets' own
evaluation splits, raw probabilities, no calibration file.

**It transfers to datasets it never saw.** MNLI, Emotion and IMDB were held out
entirely. The 0.6B goes from 0.38 to 0.77 on MNLI relation and from 0.88 to 0.97 on
IMDB; Phi holds its already-high MNLI and IMDB numbers and improves their calibration.
The synthetic held-out tasks move too: frustration 0.38 to 0.46 at 0.6B, 0.52 to 0.63 at
Phi. This is what "generic" has to mean: a domain the adapter has never trained on gets
better because the adapter learned how to answer typed questions, not a domain.

**Calibration improves on real data without a calibration file.** Raw calibration error
on the mixed adapters is 0.10 or under on almost every task at both sizes, against 0.2
to 0.8 untrained. Civil Comments, whose annotator fractions became soft labels, lands at
0.01 to 0.02. The training data now contains real uncertainty, and the proper scoring
rule turns it into honest probabilities; this is the round-two lesson holding on real
labels.

**A 0.6B with the mixed adapter rivals an untrained 3.8B.** On MS MARCO, SNLI, Quora,
banking, DBpedia and toxicity the 0.6B mixed adapter matches or beats untrained
Phi-4-mini, at a quarter of the latency. The value is in the training, not the size.

**Breadth has a small price on the narrowest task.** Phi's ticket category, 0.96 with the
synthetic-only adapter, is 0.87 with the mix; the 0.6B's is unchanged. Everything else
improved. A per-domain adapter on top of the generic one is the obvious way to buy that
back where it matters.

**Four GPUs, one job, same cost.** The distributed trainer put the Phi mixed run through
four L40S in 16 minutes for the same 8,000 examples the single-GPU run takes about 39
minutes to see, with the adapter reaching the same quality. Peer-to-peer transfers on
that host hang; the pipeline disables them. Job-level parallelism across pods stays the
larger lever for experiments; data parallelism is the lever for any single run that
matters.

**Same examples, fewer updates, slightly less model.** The four-GPU Phi adapter saw the
same 8,000 examples as the single-GPU one but in 500 optimizer steps of 16 instead of
2,000 steps of 4, and it lands a few points lower where the task is hardest: ticket
category 0.87 against 0.91, frustration 0.63 against 0.67, MNLI relation 0.80 against
0.84. Fewer, larger updates are not free. The standard remedy is to keep the step count
and let the larger batch see more data, or to scale the learning rate with the batch;
either way the four cards still finish in a fraction of the time. The next distributed
run should use 1,000 steps, twice the data, and still finish in about half the
single-GPU time.


## Round five: the pointer head

The row scorer pays one row per allowed answer and a full vocabulary projection at every
position. The pointer head (`brier/pointer.py`) pays one row per request: the prompt
already lists every option, so the head reads the hidden state where each option is
written, compares it with the hidden state at the end of its question line, and that
bilinear comparison is the option's score. The transformer body runs once with no
vocabulary head. Any number of questions and options is one forward pass; the head can
only score options that were listed, so the type guarantee is unchanged. For Grade
questions an ordinal term lets the query predict a position on the scale and scores
options by their distance from it, since the option words carry no order of their own.

**Latency, trained heads, RTX 4090, same backbone, same questions**

| backbone | questions | candidates | row scorer | pointer | speedup |
|---|---|---|---|---|---|
| Phi-4-mini | 3 | 11 | 145 ms | 48 ms | 3× |
| Phi-4-mini | 10 | 38 | 1,230 ms | 50 ms | 25× |
| Phi-4-mini | 30 | 110 | out of memory | fits | |
| Qwen3-0.6B | 3 | 11 | 40 ms | 34 ms | 1.2× |
| Qwen3-0.6B | 10 | 38 | 228 ms | 36 ms | 6× |
| Qwen3-0.6B, captured | 3 | 11 | 40 ms | **13 ms** | 3× |
| Qwen3-0.6B, captured | 10 | 38 | 228 ms | **17 ms** | 14× |
| Phi-4-mini, captured | 3 | 11 | 105 ms | **18 ms** | 6× |
| Phi-4-mini, captured | 10 | 38 | 852 ms | **32 ms** | 27× |

The pointer's latency is flat in the number of questions; the row scorer's grows with
every candidate and at thirty questions no longer fits on the card. Profiling the single
row showed that everything outside the forward pass, rendering, tokenising with offsets,
position lookup, the head, costs under two milliseconds; the forward pass itself was 30
ms for 136 tokens on the 0.6B, which is per-layer kernel launch overhead at batch size
one, not compute. Replaying the transformer body from a CUDA graph removes the launches.
Three ways of doing that were measured on the 0.6B (`graph_probe.py`, one row of 128
tokens):

| body | setup per shape | per pass | output vs eager |
|---|---|---|---|
| eager | 0 | 53 ms | exact |
| Inductor, static shapes | 62 s | 10.1 ms | differs (fused kernels) |
| Inductor, dynamic shapes | 177 s, then fails on batches | 13.0 ms | differs |
| CUDA-graph capture, no compiler | **0.3 s** | **8.0 ms** | **exact to the bit** |

Plain capture wins on every column: it costs three forward passes per input shape instead
of a minute of compilation, replays faster, and returns the eager numbers unchanged.
`scorer.capture = True` turns it on; inputs are padded to fixed length and batch buckets
so the number of graphs stays small, and the server captures them all at start-up.

**Batching concurrent requests.** Several requests share one pass by right-padding their
prompts to a common length. Under causal attention a padded token can only influence
positions after itself, so the real positions are unchanged: in fp32 the batched scores
match single-request scores to 2e-5 on both backbones (`check_batch.py --fp32`; in bf16
the batched matmul picks different kernels and scores move by up to 0.2, which is the
same noise the row scorer shows between its own paths). The server queues requests, waits
two milliseconds for company, and runs every waiting Pick and Claim set in one pass.
Measured over HTTP on one RTX 4090 with three questions per request:

| backbone | batch | 1 client | 8 clients | 32 clients |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | 12 ms, 82 req/s | 85 ms, 94 req/s | 340 ms, 94 req/s |
| Qwen3-0.6B | 8 | 14 ms, 71 req/s | 52 ms, **143 req/s** | 185 ms, **171 req/s** |
| Qwen3-0.6B | 16 | 14 ms, 70 req/s | 57 ms, 141 req/s | 191 ms, 125 req/s |
| Phi-4-mini | 1 | 23 ms, 44 req/s | 171 ms, 47 req/s | 687 ms, 47 req/s |
| Phi-4-mini | 8 | 25 ms, 40 req/s | 184 ms, 42 req/s | 647 ms, 49 req/s |

(latency is the client's p50.) Batching pays where the card is idle between launches:
on the 0.6B it lifts throughput 1.8× at 32 clients and cuts queueing latency in half.
Phi-4-mini fills the card with a single request, so a batch of eight costs eight times as
long and throughput does not move; a batch of sixteen is worse for both because the
bucket padding grows faster than the work. The default is a batch of eight; for Phi-sized
backbones on one card, `--batch 1` is the honest setting. With ten questions per request
the 0.6B goes from 54 to 80 req/s under eight clients, because a longer prompt fills the
card better per launch saved.

**Accuracy.** Five runs on the mixed corpus, evaluated on the same held-out examples as
round four. Cells are accuracy / raw ECE.

Generated by `docs/results_table5.py`.

### Qwen3-0.6B

| task | question | kind | row scorer, mixed | pointer, 8k | pointer, 22k | pointer, ordinal + distilled |
|---|---|---|---|---|---|---|
| tickets | category | pick | 0.86 / 0.06 | 0.83 / 0.11 | 0.82 / 0.13 | 0.86 / 0.05 |
| tickets | priority | grade | 0.90 / 0.20 | 0.59 / 0.21 | 0.62 / 0.12 | 0.59 / 0.06 |
| tickets | needs_human | claim | 0.97 / 0.03 | 0.96 / 0.02 | 0.93 / 0.04 | 0.97 / 0.03 |
| gridworld | move | pick | 0.84 / 0.06 | 0.81 / 0.09 | 0.81 / 0.19 | 0.84 / 0.13 |
| gridworld | danger | claim | 0.95 / 0.04 | 0.87 / 0.06 | 1.00 / 0.01 | 0.98 / 0.06 |
| toolcall | tool | pick | 1.00 / 0.00 | 0.97 / 0.02 | 1.00 / 0.01 | 1.00 / 0.02 |
| passages | answers | claim | 0.95 / 0.06 | 0.95 / 0.03 | 0.90 / 0.11 | 0.98 / 0.06 |
| matching | matches | claim | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| ag_news | topic | pick | 0.87 / 0.09 | 0.83 / 0.08 | 0.85 / 0.04 | 0.87 / 0.03 |
| dbpedia | kind | pick | 0.97 / 0.03 | 0.97 / 0.02 | 0.97 / 0.03 | 0.97 / 0.03 |
| sst5 | sentiment | grade | 0.51 / 0.11 | 0.45 / 0.07 | 0.46 / 0.12 | 0.47 / 0.05 |
| yelp | stars | grade | 0.58 / 0.12 | 0.56 / 0.06 | 0.59 / 0.07 | 0.58 / 0.06 |
| snli | relation | pick | 0.84 / 0.10 | 0.80 / 0.07 | 0.83 / 0.02 | 0.82 / 0.04 |
| snli | entailed | claim | 0.92 / 0.06 | 0.94 / 0.05 | 0.93 / 0.04 | 0.93 / 0.02 |
| qqp | duplicate | claim | 0.86 / 0.07 | 0.80 / 0.07 | 0.87 / 0.06 | 0.84 / 0.05 |
| civil | toxic | claim | 0.94 / 0.01 | 0.94 / 0.03 | 0.94 / 0.03 | 0.94 / 0.05 |
| msmarco | answers | claim | 0.88 / 0.07 | 0.88 / 0.08 | 0.88 / 0.10 | 0.88 / 0.03 |
| massive | intent | pick | 0.81 / 0.13 | 0.79 / 0.17 | 0.83 / 0.07 | 0.82 / 0.11 |
| banking | intent | pick | 0.98 / 0.02 | 0.96 / 0.03 | 0.97 / 0.02 | 0.98 / 0.01 |
| mnli (held out) | relation | pick | 0.77 / 0.11 | 0.72 / 0.09 | 0.72 / 0.04 | 0.72 / 0.08 |
| mnli (held out) | entailed | claim | 0.88 / 0.05 | 0.87 / 0.07 | 0.87 / 0.06 | 0.87 / 0.05 |
| emotion (held out) | emotion | pick | 0.55 / 0.20 | 0.49 / 0.18 | 0.49 / 0.07 | 0.49 / 0.07 |
| imdb (held out) | positive | claim | 0.97 / 0.02 | 0.99 / 0.08 | 1.00 / 0.04 | 0.97 / 0.08 |
| frustration (held out) | frustration | grade | 0.46 / 0.36 | 0.44 / 0.34 | 0.60 / 0.17 | 0.50 / 0.28 |
| duplicates (held out) | same | claim | 0.99 / 0.04 | 0.93 / 0.03 | 0.92 / 0.05 | 0.97 / 0.04 |

### Phi-4-mini

| task | question | kind | row scorer, mixed | pointer, 6k, 1 GPU | pointer, 16k, 4 GPUs |
|---|---|---|---|---|---|
| tickets | category | pick | 0.91 / 0.02 | 0.88 / 0.03 | 0.86 / 0.05 |
| tickets | priority | grade | 0.79 / 0.04 | 0.51 / 0.27 | 0.57 / 0.08 |
| tickets | needs_human | claim | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.04 |
| gridworld | move | pick | 0.84 / 0.12 | 0.85 / 0.13 | 0.79 / 0.15 |
| gridworld | danger | claim | 0.98 / 0.02 | 0.93 / 0.02 | 0.91 / 0.07 |
| toolcall | tool | pick | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.03 |
| passages | answers | claim | 0.88 / 0.10 | 0.90 / 0.08 | 0.88 / 0.08 |
| matching | matches | claim | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.03 |
| ag_news | topic | pick | 0.91 / 0.05 | 0.85 / 0.07 | 0.87 / 0.04 |
| dbpedia | kind | pick | 1.00 / 0.00 | 1.00 / 0.01 | 0.98 / 0.02 |
| sst5 | sentiment | grade | 0.55 / 0.13 | 0.50 / 0.16 | 0.41 / 0.09 |
| yelp | stars | grade | 0.65 / 0.15 | 0.53 / 0.21 | 0.55 / 0.06 |
| snli | relation | pick | 0.88 / 0.08 | 0.85 / 0.07 | 0.88 / 0.04 |
| snli | entailed | claim | 0.96 / 0.02 | 0.94 / 0.04 | 0.96 / 0.02 |
| qqp | duplicate | claim | 0.86 / 0.07 | 0.87 / 0.09 | 0.86 / 0.04 |
| civil | toxic | claim | 0.94 / 0.03 | 0.93 / 0.03 | 0.93 / 0.08 |
| msmarco | answers | claim | 0.88 / 0.04 | 0.88 / 0.10 | 0.88 / 0.09 |
| massive | intent | pick | 0.81 / 0.13 | 0.87 / 0.07 | 0.85 / 0.05 |
| banking | intent | pick | 0.97 / 0.02 | 0.98 / 0.02 | 0.95 / 0.08 |
| mnli (held out) | relation | pick | 0.84 / 0.08 | 0.81 / 0.07 | 0.80 / 0.03 |
| mnli (held out) | entailed | claim | 0.91 / 0.04 | 0.90 / 0.06 | 0.87 / 0.03 |
| emotion (held out) | emotion | pick | 0.60 / 0.20 | 0.57 / 0.21 | 0.57 / 0.10 |
| imdb (held out) | positive | claim | 1.00 / 0.01 | 1.00 / 0.02 | 0.99 / 0.09 |
| frustration (held out) | frustration | grade | 0.67 / 0.21 | 0.50 / 0.34 | 0.50 / 0.38 |
| duplicates (held out) | same | claim | 0.95 / 0.03 | 1.00 / 0.02 | 0.97 / 0.09 |

## What was learned in round five

**One row is enough for Pick and Claim.** On every real Pick and Claim question the
pointer head matches the row scorer trained on the same data: MS MARCO 0.88 against
0.88, Quora duplicates 0.87 against 0.86, toxicity 0.94 against 0.94, SNLI entailment
0.93 to 0.96, banking 0.96 to 0.98, at both sizes, with raw calibration error as low or
lower. It gets there from scratch in ten to forty minutes, and it does it at a latency
that does not move when the request grows.

**Grade is the head's weak point, and the reason is structural.** On ordered scales
the head trails the row scorer by ten to thirty points from labels alone (priority 0.59
against 0.90 at 0.6B, 0.57 against 0.79 at Phi). The row scorer inherits from language
that "medium" lies between "low" and "high"; a head comparing option tokens to a query
has to learn it. Three fixes were measured. More data helps most: the 22,000-example run
lifts the never-trained frustration scale to 0.60, above the row scorer's 0.46, and
nudges the others. The ordinal term with distillation improves calibration on every
Grade row (priority ECE 0.21 to 0.06) but not accuracy. So the order prior can be
taught, and the cheapest teacher is examples, not architecture.

**The practical answer is a hybrid.** Route Pick and Claim questions through the
pointer head and Grade questions through the row scorer. A request with three Picks and
one Grade is then one row plus four, not one plus fifteen; latency stays near flat and
Grade keeps its quality. Both paths share the backbone and the adapters, so this is a
dispatch rule, not a second model. It is the next change to `ask()`.

**Fixed overhead now dominates the small model.** The 0.6B's single row costs 62 ms of
which the forward pass is a fraction; tokenisation with offsets, position lookup and
Python account for the rest. That is the next latency target, and it is engineering,
not modelling.

**On running five experiments at once.** The round took one afternoon because every
question had its own pod. Two things went wrong and are now guarded in `pod.py`: a
transient API failure during a pull, chained to a terminate, lost one run, so
terminating now refuses without a pull marker; and syncing the growing adapters folder
truncated a transfer and left a pod without its task package, so sync excludes adapters.


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
