# The decision corpus

Stage two of the [roadmap](ROADMAP.md): public labelled datasets converted into typed
questions by one spec table (`tasks/hub.py`). Each spec names the dataset, its train and
evaluation splits, how a row becomes a state, one question (Pick, Grade or Claim) and how the
row's label maps onto that question's answers. There is no loader per dataset: one generic
`load(spec, …)` builds every one of them, and each is registered as `hub_<name>`.

**63 datasets load and pass** (acceptability 1, code 2, commonsense 11, emotion 3, factcheck 3, intent 2, nli 7, ordinal 3, paraphrase 3, qa 6, sarcasm 1, sentiment 8, spam 2, stance 3, topic 4, toxicity 4); 8 are held out of training.

## How the corpus is drawn

Every loader takes `(n, seed=0, split="all", uncertain=0.0)` like every other task. `split="train"`
draws the train split; anything else draws the evaluation split. Within that:

- **Seeded shuffle.** Sources are streamed through a seeded shuffle, so the same seed gives the
  same draw. A streaming shuffle only mixes within its buffer, so sources stored in label order
  (star ratings, intent sets) are loaded whole and shuffled globally instead.
- **Balanced classes.** Claim tasks with natural imbalance (toxicity, spam, passage relevance,
  acceptability) draw equal numbers of each class, so accuracy is never just the majority rate.
- **Disjoint halves.** A source with only one usable split is streamed once and cut in two, so
  the training half and the evaluation half share no row. Both draws use the same seed: a
  different seed per split would reshuffle and put training rows back into evaluation. Where a
  source repeats itself verbatim (`sms_spam` and `app_reviews` both do) the same text can still
  appear on both sides; the rows are disjoint, the strings are the source's own duplicates.
- **Options per row.** Multiple-choice sources become Pick questions whose options are that row's
  own candidate texts, so the answer set changes with every example rather than being fixed.
- **Long states are cut** to a fixed character budget with a marker, so no one source dominates
  the sequence length.

`hub_mixed` draws evenly across every trainable spec and shuffles the result: a corpus that is
one question kind per example but many kinds across the draw. Held-out specs never appear in it.

## Why the audit exists

A dataset that loads is not a dataset that teaches. A silently broken draw looks exactly like a
working one at the call site: the rows arrive, the labels are strings, training runs, and the
resulting accuracy is meaningless because every row carried the same label, or because the
answer set the question offered never matched the label the source gave. Both failures are
invisible until the numbers are read, and by then the run has been paid for.

`python3 corpus_audit.py --n 200` draws every spec for both splits before anything trains on it
and reports the label histogram, the number of distinct classes seen, the majority-class rate and
the mean state length. It flags a **single-class** draw and a **skewed** one (majority above 0.8),
checks that the first rows' labels are answers the question actually allows, and reports a spec
that fails to load rather than guessing at its schema. Three label-ordered sources were caught
this way and are now loaded whole and shuffled.

This table is that run's output; nothing is flagged.

## Every dataset

| dataset | family | kind | options | classes seen train / eval | majority train / eval | state chars | held out | flags | note |
|---|---|---|---|---|---|---|---|---|---|
| `hub_cola` | acceptability | claim | 2 | 2 / 2 | 0.50 / 0.50 | 45 |  |  |  |
| `hub_code_clone` | code | claim | 2 | 2 / 2 | 0.50 / 0.50 | 1987 |  |  |  |
| `hub_code_defect` | code | claim | 2 | 2 / 2 | 0.57 / 0.56 | 1042 |  |  |  |
| `hub_arc_easy` | commonsense | pick | 4 per row | 197 / 197 | 0.01 / 0.01 | 132 |  |  |  |
| `hub_copa` | commonsense | pick | 2 per row | 200 / 100 | 0.01 / 0.01 | 72 |  |  |  |
| `hub_cosmos_qa` | commonsense | pick | 4 per row | 187 / 184 | 0.07 / 0.09 | 404 |  |  |  |
| `hub_hellaswag` | commonsense | pick | 4 per row | 200 / 200 | 0.01 / 0.01 | 120 |  |  |  |
| `hub_openbookqa` | commonsense | pick | 4 per row | 199 / 198 | 0.01 / 0.01 | 76 |  |  |  |
| `hub_piqa` | commonsense | pick | 2 per row | 200 / 200 | 0.01 / 0.01 | 46 |  |  |  |
| `hub_sciq` | commonsense | pick | 4 per row | 186 / 188 | 0.01 / 0.01 | 516 |  |  |  |
| `hub_social_iqa` | commonsense | pick | 3 per row | 196 / 199 | 0.01 / 0.01 | 139 |  |  |  |
| `hub_swag` | commonsense | pick | 4 per row | 200 / 200 | 0.01 / 0.01 | 88 |  |  |  |
| `hub_winogrande` | commonsense | pick | 2 per row | 128 / 140 | 0.03 / 0.03 | 115 |  |  |  |
| `hub_emotion_setfit` | emotion | pick | 6 | 6 / 6 | 0.34 / 0.34 | 101 |  |  |  |
| `hub_go_emotions` | emotion | pick | 28 | 24 / 26 | 0.34 / 0.31 | 71 |  |  | single-label rows only |
| `hub_tweet_emotion` | emotion | pick | 4 | 4 / 4 | 0.43 / 0.43 | 96 |  |  |  |
| `hub_climate_fever` | factcheck | pick | 4 | 4 / 4 | 0.42 / 0.39 | 341 |  |  | one split, cut into disjoint halves |
| `hub_health_fact` | factcheck | pick | 4 | 4 / 4 | 0.51 / 0.46 | 1605 |  |  |  |
| `hub_liar` | factcheck | grade | 6 | 6 / 6 | 0.26 / 0.28 | 187 |  |  |  |
| `hub_clinc` | intent | pick | 151 | 114 / 104 | 0.03 / 0.15 | 42 |  |  | 150 intents plus out-of-scope; names from features |
| `hub_snips` | intent | pick | 7 | 7 / 7 | 0.18 / 0.17 | 48 |  |  |  |
| `hub_cb` | nli | pick | 3 | 3 / 3 | 0.48 / 0.50 | 367 |  |  |  |
| `hub_hans` | nli | claim | 2 | 2 / 2 | 0.51 / 0.51 | 119 |  |  |  |
| `hub_rte` | nli | claim | 2 | 2 / 2 | 0.54 / 0.53 | 367 |  |  |  |
| `hub_scitail` | nli | claim | 2 | 2 / 2 | 0.68 / 0.61 | 211 |  |  |  |
| `hub_sick` | nli | pick | 3 | 3 / 3 | 0.51 / 0.50 | 118 |  |  |  |
| `hub_wnli` | nli | claim | 2 | 2 / 2 | 0.54 / 0.56 | 184 |  |  |  |
| `hub_amazon_reviews_multi` | ordinal | grade | 5 | 5 / 5 | 0.22 / 0.23 | 192 |  |  |  |
| `hub_app_reviews` | ordinal | grade | 5 | 5 / 5 | 0.45 / 0.53 | 164 |  |  | one split, cut into disjoint halves |
| `hub_stsb` | ordinal | grade | 6 | 6 / 6 | 0.27 / 0.21 | 148 |  |  |  |
| `hub_medical_pairs` | paraphrase | claim | 2 | 2 / 2 | 0.53 / 0.54 | 238 |  |  | one split, cut into disjoint halves |
| `hub_mrpc` | paraphrase | claim | 2 | 2 / 2 | 0.60 / 0.69 | 268 |  |  |  |
| `hub_ade` | qa | claim | 2 | 2 / 2 | 0.50 / 0.50 | 142 |  |  |  |
| `hub_dream` | qa | pick | 3 per row | 199 / 200 | 0.01 / 0.01 | 648 |  |  |  |
| `hub_qnli` | qa | claim | 2 | 2 / 2 | 0.54 / 0.54 | 262 |  |  |  |
| `hub_race` | qa | pick | 4 per row | 197 / 198 | 0.01 / 0.01 | 1071 |  |  |  |
| `hub_wiki_qa` | qa | claim | 2 | 2 / 2 | 0.50 / 0.50 | 213 |  |  |  |
| `hub_amazon_polarity` | sentiment | claim | 2 | 2 / 2 | 0.52 / 0.56 | 481 |  |  |  |
| `hub_counterfactual` | sentiment | claim | 2 | 2 / 2 | 0.50 / 0.50 | 101 |  |  |  |
| `hub_customer_reviews` | sentiment | claim | 2 | 2 / 2 | 0.68 / 0.63 | 97 |  |  |  |
| `hub_poem_sentiment` | sentiment | pick | 4 | 4 / 3 | 0.64 / 0.66 | 41 |  |  |  |
| `hub_rotten_tomatoes` | sentiment | claim | 2 | 2 / 2 | 0.50 / 0.51 | 120 |  |  |  |
| `hub_subjectivity` | sentiment | claim | 2 | 2 / 2 | 0.56 / 0.57 | 127 |  |  |  |
| `hub_tweet_sentiment` | sentiment | grade | 3 | 3 / 3 | 0.54 / 0.47 | 110 |  |  |  |
| `hub_enron_spam` | spam | claim | 2 | 2 / 2 | 0.53 / 0.51 | 872 |  |  |  |
| `hub_stance_abortion` | stance | pick | 3 | 3 / 3 | 0.55 / 0.69 | 107 |  |  |  |
| `hub_stance_climate` | stance | pick | 3 | 3 / 3 | 0.53 / 0.73 | 102 |  |  |  |
| `hub_stance_feminist` | stance | pick | 3 | 3 / 3 | 0.52 / 0.66 | 109 |  |  |  |
| `hub_bbc_news` | topic | pick | 5 | 5 / 5 | 0.25 / 0.24 | 1445 |  |  |  |
| `hub_newsgroups` | topic | pick | 20 | 20 / 20 | 0.11 / 0.09 | 793 |  |  |  |
| `hub_yahoo_topics` | topic | pick | 10 | 10 / 10 | 0.14 / 0.27 | 525 |  |  |  |
| `hub_hate_offensive` | toxicity | pick | 3 | 3 / 3 | 0.74 / 0.74 | 89 |  |  |  |
| `hub_toxic_conversations` | toxicity | claim | 2 | 2 / 2 | 0.50 / 0.50 | 290 |  |  |  |
| `hub_tweet_hate` | toxicity | claim | 2 | 2 / 2 | 0.60 / 0.58 | 124 |  |  |  |
| `hub_tweet_offensive` | toxicity | claim | 2 | 2 / 2 | 0.50 / 0.50 | 141 |  |  |  |
| `hub_commonsense_qa` | commonsense | pick | 5 per row | 188 / 188 | 0.01 / 0.01 | 86 | **yes** |  |  |
| `hub_anli` | nli | pick | 3 | 3 / 3 | 0.42 / 0.36 | 425 | **yes** |  |  |
| `hub_paws` | paraphrase | claim | 2 | 2 / 2 | 0.58 / 0.57 | 253 | **yes** |  |  |
| `hub_boolq` | qa | claim | 2 | 2 / 2 | 0.62 / 0.66 | 629 | **yes** |  |  |
| `hub_tweet_irony` | sarcasm | claim | 2 | 2 / 2 | 0.51 / 0.57 | 80 | **yes** |  |  |
| `hub_financial_news` | sentiment | grade | 3 | 3 / 3 | 0.73 / 0.61 | 91 | **yes** |  |  |
| `hub_sms_spam` | spam | claim | 2 | 2 / 2 | 0.50 / 0.50 | 116 | **yes** |  | one split, cut into disjoint halves |
| `hub_trec` | topic | pick | 6 | 6 / 6 | 0.24 / 0.28 | 50 | **yes** |  |  |

`per row` means the options are that row's own candidate texts rather than a fixed set, which is
why those rows show a classes count near `n`: almost every example has a different answer string.

## Held out

Never drawn by `hub_mixed`. They span all three question kinds, so transfer can be measured to
a question type the model was not trained on rather than only to unseen wording:

- `hub_trec` — topic, pick
- `hub_financial_news` — sentiment, grade
- `hub_anli` — nli, pick
- `hub_paws` — paraphrase, claim
- `hub_sms_spam` — spam, claim
- `hub_boolq` — qa, claim
- `hub_commonsense_qa` — commonsense, pick
- `hub_tweet_irony` — sarcasm, claim

## Running it

```
python3 corpus_audit.py --n 200                     # audit every spec, rewrite this file
python3 corpus_audit.py --n 50 --only piqa,boolq    # one or two specs, no rewrite
```

Train on the whole corpus, or on any single spec:

```
python3 train.py --tasks hub_mixed --examples 5500 --model <backbone>
python3 train.py --tasks hub_piqa,hub_liar --examples 2000 --model <backbone>
```
