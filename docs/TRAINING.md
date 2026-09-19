# Training: milestone 2

Milestone 1 established that an open model's raw scores rank answers well and state
probabilities badly, and that one temperature per question repairs the global part of
that. Milestone 2 asks whether *training* the model on the decision objective does
better: higher accuracy, and calibration that beats the temperature-scaled baseline
without any post-hoc fix, including on a task the model never saw.

This page is the recipe and the measured result. The mathematics is in the
[primer](PRIMER.md), sections 13 to 16.

**Contents**: [The objective](#the-objective) · [What is trained](#what-is-trained) · [Data](#data) · [Running it](#running-it) · [Evaluation and the gate](#evaluation-and-the-gate) · [Results](#results) · [What was learned](#what-was-learned) · [Next](#next)

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
| tickets | category (Choice), priority (Score), needs_human (Noul) | yes |
| gridworld | move (Choice, ties allowed), danger (Noul) | yes |
| toolcall | tool (Choice over a per-example subset of eight tools) | yes |
| passages | answers (Noul) | yes |
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

## Results

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

## What was learned

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
