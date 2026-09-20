# Roadmap

Brier's aim is a new class of model: a non-generative model that takes any state and any
set of typed questions, including question types it has never seen, and answers every one
of them in one forward pass with a probability per allowed answer that is calibrated by
training, and that learns from what happened after a decision rather than only from
labels. Everything measured so far is on the way to that model, not the model itself.
This document says what remains, in order, with what each stage needs and costs.

## Where the evidence stands

Rounds one to six ([TRAINING.md](TRAINING.md)) established the single-pass typed decision
function, the calibration kit, the training recipe, the pointer head, and the serving
path, with baselines, seeds and intervals, on open decoder language models used as
scorers. Round seven, in progress, tests the two things that separate that from the
target: an encoder-only backbone, which is non-generative in every sense, and training
from outcomes alone on text decisions.

## Stage 1 · identity and learning signal (running)

**Question.** Does the method survive without a generative language model underneath,
and can consequences replace labels on text?

**Runs.** ModernBERT-large fully fine-tuned through the pointer head, three seeds, with
base, LoRA and frozen-backbone ablations, on the same corpus and tables as the decoder
backbones. Outcome-reward training on ticket routing and tool selection against
supervised training on the same states, three seeds each, where the environment returns
a bounced route, a missed SLA, a failed call, and never a label.

**Decides.** Whether the released model can be an encoder, and whether the released
training recipe can be outcome-first. Either answer is publishable; both are measured.

## Stage 2 · the decision corpus

**Question.** What does a model need to have seen to answer questions it has not seen?

**Data.** Training consumes decisions, not text: a state, a set of typed questions, and
an answer or an outcome. Internet text has none of these; it is what the backbone was
pretrained on and teaches reading, not deciding. The sources, in the order they are used,
each passing the audit that round six made mandatory (label histograms per split, balance
for Claim tasks, no label-sorted streaming):

1. **Public datasets converted into typed questions**: classification, entailment,
   ranking, preference, retrieval and QA sets, several hundred of them, the way
   instruction-tuned models were built. Free. Covers every question type that exists as a
   dataset, on the order of ten million decisions. The first training runs use only this,
   and the measurement that matters is transfer to question types held out entirely.
2. **Generated decisions, only if that transfer plateaus**: a strong language model
   writes novel question sets and options on realistic states with verified answers,
   deliberate ambiguity as soft labels, and contrastive near-misses. This is distillation
   of a large model's judgement into a small non-generative one, and it exists to cover
   the question types no dataset has. Added at one million decisions for a few hundred
   dollars of API cost before anything larger is considered.
3. **Outcome logs.** None exist yet without users; simulated environments stand in, as in
   stage 1.

**Also.** A public benchmark drawn from held-out datasets and held-out question types,
with calibration and coverage as first-class metrics, seeds and intervals, and the
opponents from round six plus zero-shot generation from strong open models.

## Stage 3 · the backbone at scale

**Question.** How good does the non-generative model get with a large backbone?

**Backbone.** The largest strong encoder is about 400M parameters. Above that, a 4B to
8B open decoder is converted into a bidirectional encoder (the adaptation exists in the
literature) and then trained on the stage-2 corpus with the joint, outcome and
calibration objectives.

**Compute.** Eight H100s for about a day per run, roughly 700 dollars of pod time, several
runs per iteration. The tooling from rounds four to six (`launch.sh`, `watch.py`,
`pod.py` with hard time-to-lives) already runs fleets of pods and is the harness.

## Stage 4 · release and the novel work

**Release.** Weights and adapters on the Hub with a model card, the serving path with
capture and batching, the calibration kit, the benchmark, and every number reproducible
from `results/`.

**Method.** The parts no one else has published: outcome learning at scale for typed
decisions, the calibration head, conformal prediction sets with a coverage guarantee per
question, and the pointer head over options as the unit of decision.

## Principles

- Measure before claiming; seeds and intervals on every number; withdraw what does not
  survive.
- Audit every dataset's label histogram before believing an accuracy.
- Never borrow a vendor's vocabulary; differentiate on substance.
- Every pod has a hard time-to-live; nothing runs unattended past it.
