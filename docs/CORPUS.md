# The decision corpus

Stage two of the [roadmap](ROADMAP.md): public labelled datasets from the Hub converted into
typed questions by one spec table (`tasks/hub.py`). Each spec names the dataset, its train and
evaluation splits, how a row becomes a state, one question (Pick, Grade or Claim) and how the
row's label maps onto that question's answers. Multiple-choice sources become Pick questions whose
options are the candidate texts of that row, so the answer set changes with every example.
Sources with one split are cut into disjoint halves by seed. Claim tasks with natural imbalance
draw balanced classes, so accuracy is never the majority rate; label-sorted sources are loaded
whole and shuffled. Held-out specs never enter training: they measure transfer to question types
the model has not seen.

`python3 corpus_audit.py --n 200` produced this table; it draws every spec for train and
evaluation and flags a single class or a majority above 0.8. Train on the corpus with
`python3 train.py --tasks hub_mixed --examples <total> …` (an even draw across every trainable spec) or on
any single spec with `--tasks hub_<name>`.

**63 datasets load and pass** (acceptability 1, code 2, commonsense 11, emotion 3, factcheck 3, intent 2, nli 7, ordinal 3, paraphrase 3, qa 6, sarcasm 1, sentiment 8, spam 2, stance 3, topic 4, toxicity 4); 8 are held out.

| dataset | family | kind | options | classes seen | majority train / eval | state chars | held out | flags |
|---|---|---|---|---|---|---|---|---|
| `hub_cola` | acceptability | claim | 2 | 2 / 2 | 0.50 / 0.50 | 45 |  |  |
| `hub_code_clone` | code | claim | 2 | 2 / 2 | 0.50 / 0.50 | 1987 |  |  |
| `hub_code_defect` | code | claim | 2 | 2 / 2 | 0.57 / 0.56 | 1042 |  |  |
| `hub_arc_easy` | commonsense | pick | 4 | 197 / 197 | 0.01 / 0.01 | 132 |  |  |
| `hub_copa` | commonsense | pick | 2 | 200 / 100 | 0.01 / 0.01 | 72 |  |  |
| `hub_cosmos_qa` | commonsense | pick | 4 | 187 / 184 | 0.07 / 0.09 | 404 |  |  |
| `hub_hellaswag` | commonsense | pick | 4 | 200 / 200 | 0.01 / 0.01 | 120 |  |  |
| `hub_openbookqa` | commonsense | pick | 4 | 199 / 198 | 0.01 / 0.01 | 76 |  |  |
| `hub_piqa` | commonsense | pick | 2 | 200 / 200 | 0.01 / 0.01 | 46 |  |  |
| `hub_sciq` | commonsense | pick | 4 | 186 / 188 | 0.01 / 0.01 | 516 |  |  |
| `hub_social_iqa` | commonsense | pick | 3 | 196 / 199 | 0.01 / 0.01 | 139 |  |  |
| `hub_swag` | commonsense | pick | 4 | 200 / 200 | 0.01 / 0.01 | 88 |  |  |
| `hub_winogrande` | commonsense | pick | 2 | 128 / 140 | 0.03 / 0.03 | 115 |  |  |
| `hub_emotion_setfit` | emotion | pick | 6 | 6 / 6 | 0.34 / 0.34 | 101 |  |  |
| `hub_go_emotions` | emotion | pick | 28 | 24 / 26 | 0.34 / 0.31 | 71 |  |  |
| `hub_tweet_emotion` | emotion | pick | 4 | 4 / 4 | 0.43 / 0.43 | 96 |  |  |
| `hub_climate_fever` | factcheck | pick | 4 | 4 / 4 | 0.42 / 0.45 | 341 |  |  |
| `hub_health_fact` | factcheck | pick | 4 | 4 / 4 | 0.51 / 0.46 | 1605 |  |  |
| `hub_liar` | factcheck | grade | 6 | 6 / 6 | 0.26 / 0.28 | 187 |  |  |
| `hub_clinc` | intent | pick | 151 | 52 / 104 | 0.04 / 0.14 | 40 |  |  |
| `hub_snips` | intent | pick | 7 | 3 / 7 | 0.41 / 0.17 | 55 |  |  |
| `hub_cb` | nli | pick | 3 | 3 / 3 | 0.48 / 0.50 | 367 |  |  |
| `hub_hans` | nli | claim | 2 | 2 / 2 | 0.51 / 0.51 | 119 |  |  |
| `hub_rte` | nli | claim | 2 | 2 / 2 | 0.54 / 0.53 | 367 |  |  |
| `hub_scitail` | nli | claim | 2 | 2 / 2 | 0.68 / 0.61 | 211 |  |  |
| `hub_sick` | nli | pick | 3 | 3 / 3 | 0.51 / 0.50 | 118 |  |  |
| `hub_wnli` | nli | claim | 2 | 2 / 2 | 0.54 / 0.56 | 184 |  |  |
| `hub_amazon_reviews_multi` | ordinal | grade | 5 | 1 / 5 | 1.00 / 0.23 | 170 |  | SINGLE-CLASS |
| `hub_app_reviews` | ordinal | grade | 5 | 5 / 5 | 0.45 / 0.45 | 164 |  |  |
| `hub_stsb` | ordinal | grade | 6 | 6 / 6 | 0.27 / 0.21 | 148 |  |  |
| `hub_medical_pairs` | paraphrase | claim | 2 | 2 / 2 | 0.53 / 0.52 | 238 |  |  |
| `hub_mrpc` | paraphrase | claim | 2 | 2 / 2 | 0.60 / 0.69 | 268 |  |  |
| `hub_ade` | qa | claim | 2 | 2 / 2 | 0.50 / 0.50 | 142 |  |  |
| `hub_dream` | qa | pick | 3 | 199 / 200 | 0.01 / 0.01 | 648 |  |  |
| `hub_qnli` | qa | claim | 2 | 2 / 2 | 0.54 / 0.54 | 262 |  |  |
| `hub_race` | qa | pick | 4 | 197 / 198 | 0.01 / 0.01 | 1071 |  |  |
| `hub_wiki_qa` | qa | claim | 2 | 2 / 2 | 0.50 / 0.50 | 213 |  |  |
| `hub_amazon_polarity` | sentiment | claim | 2 | 2 / 2 | 0.52 / 0.56 | 481 |  |  |
| `hub_counterfactual` | sentiment | claim | 2 | 2 / 2 | 0.50 / 0.50 | 101 |  |  |
| `hub_customer_reviews` | sentiment | claim | 2 | 2 / 2 | 0.68 / 0.63 | 97 |  |  |
| `hub_poem_sentiment` | sentiment | pick | 4 | 4 / 3 | 0.64 / 0.66 | 41 |  |  |
| `hub_rotten_tomatoes` | sentiment | claim | 2 | 2 / 2 | 0.50 / 0.51 | 120 |  |  |
| `hub_subjectivity` | sentiment | claim | 2 | 2 / 2 | 0.56 / 0.57 | 127 |  |  |
| `hub_tweet_sentiment` | sentiment | grade | 3 | 3 / 3 | 0.54 / 0.47 | 110 |  |  |
| `hub_enron_spam` | spam | claim | 2 | 2 / 2 | 0.53 / 0.51 | 872 |  |  |
| `hub_stance_abortion` | stance | pick | 3 | 3 / 3 | 0.55 / 0.69 | 107 |  |  |
| `hub_stance_climate` | stance | pick | 3 | 3 / 3 | 0.53 / 0.73 | 102 |  |  |
| `hub_stance_feminist` | stance | pick | 3 | 3 / 3 | 0.52 / 0.66 | 109 |  |  |
| `hub_bbc_news` | topic | pick | 5 | 5 / 5 | 0.25 / 0.24 | 1445 |  |  |
| `hub_newsgroups` | topic | pick | 20 | 20 / 20 | 0.11 / 0.09 | 793 |  |  |
| `hub_yahoo_topics` | topic | pick | 10 | 10 / 10 | 0.14 / 0.27 | 525 |  |  |
| `hub_hate_offensive` | toxicity | pick | 3 | 3 / 3 | 0.74 / 0.74 | 89 |  |  |
| `hub_toxic_conversations` | toxicity | claim | 2 | 2 / 2 | 0.50 / 0.50 | 290 |  |  |
| `hub_tweet_hate` | toxicity | claim | 2 | 2 / 2 | 0.60 / 0.58 | 124 |  |  |
| `hub_tweet_offensive` | toxicity | claim | 2 | 2 / 2 | 0.50 / 0.50 | 141 |  |  |
| `hub_commonsense_qa` | commonsense | pick | 5 | 188 / 188 | 0.01 / 0.01 | 86 | **yes** |  |
| `hub_anli` | nli | pick | 3 | 3 / 3 | 0.42 / 0.36 | 425 | **yes** |  |
| `hub_paws` | paraphrase | claim | 2 | 2 / 2 | 0.58 / 0.57 | 253 | **yes** |  |
| `hub_boolq` | qa | claim | 2 | 2 / 2 | 0.62 / 0.66 | 629 | **yes** |  |
| `hub_tweet_irony` | sarcasm | claim | 2 | 2 / 2 | 0.51 / 0.57 | 80 | **yes** |  |
| `hub_financial_news` | sentiment | grade | 3 | 3 / 3 | 0.73 / 0.61 | 91 | **yes** |  |
| `hub_sms_spam` | spam | claim | 2 | 2 / 2 | 0.50 / 0.50 | 116 | **yes** |  |
| `hub_trec` | topic | pick | 6 | 6 / 6 | 0.24 / 0.28 | 50 | **yes** |  |

## Held out

Never drawn by `hub_mixed`; used only to evaluate transfer:

- `hub_trec` (topic, pick)
- `hub_financial_news` (sentiment, grade)
- `hub_anli` (nli, pick)
- `hub_paws` (paraphrase, claim)
- `hub_sms_spam` (spam, claim)
- `hub_boolq` (qa, claim)
- `hub_commonsense_qa` (commonsense, pick)
- `hub_tweet_irony` (sarcasm, claim)
