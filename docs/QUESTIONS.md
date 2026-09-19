# Questions: the API

Brier answers **questions** about a **state**. A question is one of three kinds, every
question in a call is answered in the same forward pass, and every answer carries a full
probability distribution. This page is the reference; the [architecture doc](ARCHITECTURE.md)
explains the mechanism and the [primer](PRIMER.md) the mathematics.

**Contents**: [State](#state) · [Choice](#choice) · [Score](#score) · [Claim](#claim) · [Asking](#asking) · [Answers](#answers) · [Confidence](#confidence) · [Independence](#independence) · [Writing good questions](#writing-good-questions) · [Limits](#limits)

## State

The state is whatever the questions are about: a string, or any JSON-serialisable
object. Objects are rendered as indented JSON in the prompt, so field names are visible to
the model and can be referred to in instructions.

```python
state = "Webhooks are firing twice since 3am. Need this today."

state = {
    "request": "send me the invoice for September",
    "tools": {"get_invoice": "Fetch a billing invoice by month", "reset_password": "..."},
}
```

Keep the state to what the questions need. Irrelevant fields are distractors for a model
that reads everything in one pass.

## Choice

*Which of these options?*

```python
from brier import Choice

Choice("category", "which team should handle this ticket",
       ("billing", "bug", "feature", "account", "other"))

Choice("tool", "which tool should be called to handle this request",
       {"get_invoice": "Fetch a billing invoice by month",
        "reset_password": "Send a password reset email",
        "none": "No tool applies; reply conversationally"})
```

- `id`: the key the answer comes back under.
- `instructions`: the question, as a phrase. It is rendered in parentheses after the id.
- `options`: a tuple of labels, or a dict of label to description when the labels are not
  self-explanatory (tool names, codes). Descriptions are rendered next to the labels.

The answer's `value` is one of the labels. It cannot be anything else.

## Score

*Which level on an ordered scale?*

```python
from brier import Score

Score("priority", "how fast this must be handled", ("low", "medium", "high", "urgent"))
Score("frustration", "how frustrated is the customer", ("calm", "annoyed", "angry", "furious"))
```

- `levels`: ordered lowest to highest. A dict of label to description is accepted for
  documentation, but only the labels are rendered: measured on Qwen3-4B, inline
  descriptions lowered accuracy on the frustration task from 0.43 to 0.28, because the
  model treats the description text as part of the answer space.

The answer carries `value` (the top level), the distribution, and `expected`, the
probability-weighted mean position (0 for the lowest level). A distribution split
between `annoyed` and `angry` has `expected ≈ 1.5`, which is the honest answer where the
argmax has to pick one.

## Claim

*Is this statement true of the state?*

```python
from brier import Claim

Claim("needs_human", "the customer is asking for a human agent rather than an automated reply")
Claim("answers", "the passage contains the information needed to answer the question")
Claim("refund", "a refund is requested", true="asks for money back", false="anything else")
```

- `instructions`: the statement, phrased so that "yes" means it holds.
- `true` / `false`: optional clarifications of what yes and no mean. Rendered only when
  given.

The answer's `value` is the probability that the statement is true, a float in [0, 1].
Because it is one distribution over `{yes, no}`, the probability of the negation is
exactly `1 - value`. That invariant holds by construction here and does not, in general,
hold for a hosted decision model that answers a statement and its negation as two
separate questions.

## Asking

```python
from brier import Scorer, ask

scorer = Scorer("Qwen/Qwen3-4B")
answers = ask(scorer, state, [q1, q2, q3])
```

One call, one batched forward pass, all questions. The number of sequences in the batch
is the total number of allowed answers across the questions, so latency grows with the
size of the option sets, not with how many questions there are or how long an answer
would be to write.

`ask` accepts `temperatures={id: T}` to apply per-question calibration; the
[calibration kit](CALIBRATION.md) produces these and applies them for you.

## Answers

```python
a = answers["category"]
a.value            # 'bug'
a.probabilities    # {'billing': 0.07, 'bug': 0.86, 'feature': 0.0, 'account': 0.05, 'other': 0.02}
a.confidence       # 0.86
a.margin           # 0.79
a.scores           # raw log-probabilities per option, before softmax
a.top(2)           # [('bug', 0.86), ('billing', 0.07)]
a.tier             # 'auto' | 'review' | 'human', set when a Calibration has been applied
answers.latency_ms # for the whole call
```

Every field is a plain Python value. Nothing needs parsing.

## Confidence

`confidence` is the top probability. `margin` is the top probability minus the
runner-up. Both are stated here because "confidence" is not a standard quantity and a
number whose definition is hidden cannot be reasoned about. Neither is meaningful until
calibrated: out of the box the model's probabilities are overconfident, and the
[calibration kit](CALIBRATION.md) is how they are made honest and how thresholds on them
are chosen.

## Independence

Every question is answered from the same prompt with no knowledge of the other
questions' answers. Adding or removing a question does not change the others' results.
This is what makes one pass possible. It also means a question cannot say "if the
category is billing, then …"; put that logic in your code, on the returned
probabilities, where it is explicit.

## Writing good questions

- **Atomic.** One judgement per question. "Is this a bug report" and "is it urgent" are
  two Claims, not one Choice over four combinations.
- **Literal.** The model answers what is written. If the boundary matters, put it in the
  instructions: "the customer explicitly asks for a person" rather than "needs a human."
- **Bounded.** Choice options should be exhaustive; include an `other` or `none` when
  the real world has one, or the model will force a wrong fit.
- **Short labels.** Labels that tokenize to one or two tokens score on equal footing.
  Deliberately long labels among short ones are penalised by the extra tokens; the
  [primer](PRIMER.md#8-scoring-a-candidate-the-chain-rule-applied) explains why.
- **No arithmetic.** Counting, comparing numbers, and date arithmetic are not things a
  single forward pass does. Compute them in code and ask the model the semantic part.

## Limits

- Options and levels must be finite and enumerated. Free-text answers are not a
  question type and will not be.
- Latency is one forward pass over `sum(len(options))` sequences of prompt length. On
  an RTX 4090 with Qwen3-4B and an eleven-candidate call: about 100 ms.
- Any Hugging Face causal language model is a valid backbone. Instruction-tuned models
  work best out of the box because the JSON answer prefix is in their training
  distribution.
