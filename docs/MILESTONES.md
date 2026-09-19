# Milestones

The project's plan, what each stage set out to prove, what it measured, and what it
changed. Dates are when the numbers were produced. Every result named here is in
[TRAINING.md](TRAINING.md) or [CALIBRATION.md](CALIBRATION.md) with its table.

<p align="center"><img src="fig-milestones.svg" width="900" alt="timeline of milestones and rounds, each with its question and the number that answered it"></p>

**Contents**: [Milestone 1](#milestone-1-zero-training) · [Question API and kit](#question-api-and-calibration-kit) · [Milestone 2](#milestone-2-training-for-calibrated-decisions) · [Round three](#round-three) · [Milestone 3](#milestone-3-the-demo) · [Next](#next) · [Principles](#principles-that-held)

## Milestone 1, zero training

*2026-09-18. Done.*

**Question.** Is it worth training anything? Take an open model exactly as released,
score every allowed answer in one forward pass, and measure against the same weights
generating JSON.

**Delivered.** The single-pass scorer, the JSON baseline with a self-reported confidence,
two synthetic tasks with computable ground truth (tickets, gridworld), and the metrics
(accuracy, ECE, Brier, latency, held-out temperature scaling). Qwen3 at 0.6B, 1.7B and
4B on an RTX 4090.

**Found.** One pass is 10 to 20× faster than generation from the same weights at equal or
better accuracy. Type errors are zero for both methods on this simple a schema, so the
structural guarantee was not yet the differentiator. Raw confidence is badly inflated
(0.95 mean confidence at 0.59 accuracy on the 4B) and one temperature per question cuts
the calibration error two to four times. Two format lessons: the scoring header must be
something the model would have written (a JSON key works, a bare `field:` line does
not), and Score levels must be listed as bare names, because inline descriptions bleed
into the answer space.

**Decision.** Training is worth it, and calibration is the gap.

## Question API and calibration kit

*2026-09-18. Done.*

**Question.** Vendors of decision models ship probabilities and tell users to pick their
own thresholds. Can the thresholds be derived from the user's own data with a stated
guarantee?

**Delivered.** Choice, Score and Claim questions answered in one pass with full
distributions, a stated definition of confidence, and expected level for scores. The
kit: per-question temperature fit by maximum likelihood with five-fold cross-fitting,
ECE and Brier score on held-out folds, reliability diagrams, and the lowest confidence
cutoff that reaches a target precision for `auto`, `review` and `human` tiers, with the
coverage each gets.

**Found.** On the 4B untrained, tool selection and passage relevance are near-solved and
become honest after one temperature, with an auto tier covering essentially everything
at 95% precision. Tickets sit in the regime where calibration cannot manufacture
precision and most decisions go to a person. Contextual debiasing with a content-free
input is fragile and is off by default.

**Decision.** The kit is the evaluation harness for every model from here, trained or
not.

## Milestone 2, training for calibrated decisions

*Round one 2026-09-18, round two 2026-09-19. Done.*

**Question.** Does training the backbone on the decision objective beat the one-parameter
fix, including on tasks it never saw?

**Delivered.** LoRA adapters trained with cross-entropy over the answer set, a strictly
proper scoring rule, using the scorer's own code path. Three new tasks (tool selection,
passage relevance, frustration). The gate: trained raw ECE on never-trained tasks must be
under the untrained model's temperature-scaled ECE.

**Round one found.** Trained tasks reach 1.000 accuracy with a temperature of 0.05: template
memorisation, marked as such. Gridworld, with random grids, shows real learning (0.6B
0.35 to 0.84). The unseen task improves 14 to 20 points but fails the gate at every size.
Clean labels teach a model that it is always right.

**Round two changed.** Template hold-out (disjoint template halves for training and
evaluation), 30% ambiguous training examples with soft labels equal to the generator's
posterior, a second never-trained task (duplicates), a real dataset (banking77, human
labels), outcome-reward training with no labels, shared-prefix cached scoring, and five
backbones from three families.

**Round two found.** Memorisation gone; fitted temperatures fall from about 10 to between
1 and 2. At 4B the gate passes on frustration (0.44 to 0.63, ECE 0.54 to 0.25) and
duplicates (0.73 to 0.997, ECE 0.26 to 0.04). Every size reaches 0.90 or better on real
banking data with ECE at or under 0.03. At 0.6B duplicates regresses through a
"shared cues mean yes" shortcut, diagnosed by confusion matrix. Outcome reward without
labels reaches 0.81 on gridworld against 0.84 supervised. Cached scoring is exact and
1.6 to 2.3× faster from 1.7B up. Phi-4-mini is the strongest untrained backbone.

**Decision.** The objective works; data decides what it learns. Next round is data.

## Round three

*2026-09-19. Done; results in TRAINING.md.*

**Question.** Does a contrastive claim cure the shortcut, does the strongest backbone
train as well as it evaluates, does a head over hidden states help, and does a delayed
outcome teach anything a label cannot?

**Delivered.** The matching task and not-human phrases (contrastive claims), Phi-4-mini
trained with the same recipe (LoRA targets detected per architecture), the typed head
(input-conditional temperature plus per-candidate residual, zero-initialised to the plain
scorer), six-step outcome episodes, and the trajectory recorder for the demo. Four pods
in parallel.

**Found.** Contrastive data cured the negative transfer: 0.6B duplicates 0.64 to 0.97
with nothing else changed. Phi-4-mini trained is the strongest model in the project
(0.96 on ticket category, 0.99 on duplicates, 0.92 on real banking). The typed head
costs accuracy at 0.6B and is pure gain at 4B (passages 0.88 to 0.93 with raw ECE 0.09
to 0.01), so it is kept from about 4B up.

## Milestone 3, the demo

*2026-09-19. Built.*

**Question.** Can someone watch the model decide, tick by tick, with its probabilities
visible?

**Delivered.** [demo.html](demo.html) replays recorded gridworld games: the grid, the
four move probabilities, the chosen move, whether it was optimal, and the latency of
each decision, with play, step and reset, for a trained and an untrained model.
`record.py` produces the trajectories from any model or adapter.

**Limit.** It is a replay of recorded games, because the page is static and the model
needs a GPU. A live version, the page asking a local scorer each tick, is the first item
below.

## Next

- A live demo served from the scorer.
- A head without the language-model term, scoring from hidden states alone.
- Distilled labels on real tickets and tool-call logs, with disagreements kept.
- Multi-step credit assignment: discounting or a value estimate for longer episodes.
- Phi-4-mini as the working backbone throughout.

## Principles that held

- **Measure before training.** The kit existed before the first adapter, and every
  training claim is a held-out number it produced.
- **The gate refuses.** Where the numbers did not support a claim, the document says so,
  including the round-one memorisation and the round-two negative transfer.
- **The model learns exactly the rules its data contains.** Every failure in the project
  was a data failure and every fix was a data fix: ambiguity for overconfidence,
  counterexamples for shortcuts.
- **Any open backbone.** The same code ran five models from three families; the working
  backbone is chosen by the table, not by habit.
