"""The decision corpus, stage two: public labelled datasets from the Hub converted into typed
questions by a single spec table. Each spec names the dataset, its train and evaluation splits,
how the state is built from a row, one question (Pick, Grade or Claim), and how the row's label
maps onto that question's answers. Loaders are data-driven from the table; nothing here is
hand-written per dataset beyond the spec. Held-out specs never enter training, so transfer to
question types the model has not seen can be measured. Every spec is checked by
corpus_audit.py, which reports label histograms and flags skew before anything is trained on."""
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from brier.questions import Pick, Grade, Claim
from .public import _stream, _balance, _cut, _splits


@dataclass
class Spec:
    name: str
    path: str
    config: Optional[str]
    train: str
    eval: str
    kind: str                                   # pick | grade | claim
    qid: str
    instructions: str
    state: Callable                             # row -> state (str or dict)
    label: Callable                             # row -> label name (str), level name (str) or bool
    labels: tuple = ()                          # fixed answer set (Pick options / Grade levels); () when options come from the row
    options: Optional[Callable] = None          # row -> tuple of option texts (multiple choice over candidates)
    claim: tuple = ("yes", "no")                # Claim wording for true / false
    balance: bool = False                       # equal classes (Claim tasks with natural imbalance)
    ordered: bool = False                       # source stored in label order: load whole and shuffle
    filt: Optional[Callable] = None
    held_out: bool = False
    family: str = ""                            # topic | sentiment | nli | paraphrase | intent | toxicity | spam | factcheck | qa | commonsense | acceptability | emotion | sarcasm | stance | ordinal | code
    note: str = ""


def _mc(row_opts, row_ans):
    """Multiple choice over candidate texts: options from the row, the answer is one of them."""
    return row_opts, row_ans


def _text(*fields):
    def f(r):
        if len(fields) == 1:
            return _cut(r[fields[0]])
        return {k: _cut(r[k]) for k in fields}
    return f


def _label_text(names):
    return lambda r: names[int(r["label"])]


SPECS = [
    # ---- topic ----
    Spec("bbc_news", "SetFit/bbc-news", None, "train", "test", "pick", "section", "the section of the newspaper this article belongs to",
         _text("text"), lambda r: r["label_text"], labels=("business", "entertainment", "politics", "sport", "tech"), family="topic"),
    Spec("newsgroups", "SetFit/20_newsgroups", None, "train", "test", "pick", "group", "which discussion group this post was made in",
         _text("text"), lambda r: r["label_text"], labels=("alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"),
         filt=lambda r: len(r["text"].strip()) > 40, family="topic"),
    Spec("yahoo_topics", "community-datasets/yahoo_answers_topics", None, "train", "test", "pick", "topic", "the topic category of this question and its best answer",
         lambda r: {"question": _cut(r["question_title"] + " " + r["question_content"]), "answer": _cut(r["best_answer"])},
         _label_text(("Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government")),
         labels=("Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"), family="topic"),
    Spec("trec", "CogComp/trec", None, "train", "test", "pick", "asks_for", "what kind of thing this question asks for",
         _text("text"), lambda r: ("abbreviation", "entity", "description", "human", "location", "number")[int(r["coarse_label"])],
         labels=("abbreviation", "entity", "description", "human", "location", "number"), held_out=True, family="topic"),
    Spec("banking_intent_full", "PolyAI/banking77", None, "train", "test", "pick", "intent", "what the customer wants",
         _text("text"), lambda r: r["label"], labels=(), options=None, family="intent", note="77 intents, names from the dataset features"),
    # ---- sentiment / ordinal ----
    Spec("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", None, "train", "test", "claim", "positive", "this review is positive about the film",
         _text("text"), lambda r: int(r["label"]) == 1, family="sentiment"),
    Spec("amazon_polarity", "fancyzhx/amazon_polarity", None, "train", "test", "claim", "positive", "this product review is positive",
         lambda r: {"title": r["title"], "review": _cut(r["content"])}, lambda r: int(r["label"]) == 1, family="sentiment"),
    Spec("tweet_sentiment", "cardiffnlp/tweet_eval", "sentiment", "train", "test", "grade", "sentiment", "the sentiment of this tweet",
         _text("text"), lambda r: ("negative", "neutral", "positive")[int(r["label"])], labels=("negative", "neutral", "positive"), family="sentiment"),
    Spec("financial_phrasebank", "takala/financial_phrasebank", "sentences_allagree", "train", "train", "grade", "sentiment", "the sentiment of this financial news sentence for the company",
         _text("sentence"), lambda r: ("negative", "neutral", "positive")[int(r["label"])], labels=("negative", "neutral", "positive"), held_out=True, family="sentiment",
         note="one split; train and eval are disjoint halves by seed"),
    Spec("app_reviews", "sealuzh/app_reviews", None, "train", "train", "grade", "stars", "the star rating this app review gives",
         lambda r: {"app": r["package_name"], "review": _cut(r["review"])}, lambda r: str(int(r["star"])), labels=("1", "2", "3", "4", "5"), family="ordinal",
         note="one split; disjoint halves by seed"),
    Spec("amazon_reviews_multi", "mteb/amazon_reviews_multi", "en", "train", "test", "grade", "stars", "the star rating this product review gives",
         _text("text"), lambda r: str(int(r["label"]) + 1), labels=("1", "2", "3", "4", "5"), family="ordinal"),
    Spec("poem_sentiment", "google-research-datasets/poem_sentiment", None, "train", "test", "pick", "sentiment", "the sentiment of this line of verse",
         _text("verse_text"), lambda r: ("negative", "positive", "no impact", "mixed")[int(r["label"])], labels=("negative", "positive", "no impact", "mixed"), family="sentiment"),
    Spec("dynasent", "dynabench/dynasent", "dynabench.dynasent.r1.all", "train", "test", "grade", "sentiment", "the sentiment of this sentence",
         _text("sentence"), lambda r: r["gold_label"], labels=("negative", "neutral", "positive"), family="sentiment"),
    Spec("counterfactual", "SetFit/amazon_counterfactual_en", None, "train", "test", "claim", "counterfactual", "this review sentence describes something that did not happen or a hypothetical",
         _text("text"), lambda r: int(r["label"]) == 1, balance=True, family="sentiment"),
    Spec("subjectivity", "SetFit/subj", None, "train", "test", "claim", "subjective", "this sentence expresses an opinion rather than a fact",
         _text("text"), lambda r: r["label_text"] == "subjective", family="sentiment"),
    Spec("customer_reviews", "SetFit/CR", None, "train", "test", "claim", "positive", "this product review sentence is positive",
         _text("text"), lambda r: int(r["label"]) == 1, family="sentiment"),
    # ---- NLI / entailment ----
    Spec("rte", "nyu-mll/glue", "rte", "train", "validation", "claim", "entailed", "the second sentence follows from the first",
         lambda r: {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, lambda r: int(r["label"]) == 0, family="nli"),
    Spec("wnli", "nyu-mll/glue", "wnli", "train", "validation", "claim", "entailed", "the second sentence follows from the first",
         lambda r: {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, lambda r: int(r["label"]) == 1, family="nli"),
    Spec("qnli", "nyu-mll/glue", "qnli", "train", "validation", "claim", "answers", "the sentence contains the answer to the question",
         lambda r: {"question": r["question"], "sentence": r["sentence"]}, lambda r: int(r["label"]) == 0, family="qa"),
    Spec("anli", "facebook/anli", None, "train_r1", "test_r1", "pick", "relation", "how the hypothesis relates to the premise",
         lambda r: {"premise": _cut(r["premise"]), "hypothesis": r["hypothesis"]}, lambda r: ("entailment", "neutral", "contradiction")[int(r["label"])],
         labels=("entailment", "neutral", "contradiction"), held_out=True, family="nli"),
    Spec("scitail", "allenai/scitail", "tsv_format", "train", "test", "claim", "entailed", "the hypothesis follows from the premise",
         lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, lambda r: r["label"] == "entails", family="nli"),
    Spec("hans", "jhu-cogsci/hans", None, "train", "validation", "claim", "entailed", "the hypothesis follows from the premise",
         lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, lambda r: int(r["label"]) == 0, family="nli"),
    Spec("sick", "RobZamp/sick", None, "train", "test", "pick", "relation", "how the second sentence relates to the first",
         lambda r: {"first": r["sentence_A"], "second": r["sentence_B"]}, lambda r: ("entailment", "neutral", "contradiction")[int(r["label"])],
         labels=("entailment", "neutral", "contradiction"), family="nli"),
    Spec("cb", "aps/super_glue", "cb", "train", "validation", "pick", "relation", "how the hypothesis relates to the premise",
         lambda r: {"premise": _cut(r["premise"]), "hypothesis": r["hypothesis"]}, lambda r: ("entailment", "contradiction", "neutral")[int(r["label"])],
         labels=("entailment", "contradiction", "neutral"), family="nli"),
    # ---- paraphrase / duplicates ----
    Spec("mrpc", "nyu-mll/glue", "mrpc", "train", "validation", "claim", "paraphrase", "the two sentences say the same thing",
         lambda r: {"first": r["sentence1"], "second": r["sentence2"]}, lambda r: int(r["label"]) == 1, family="paraphrase"),
    Spec("paws", "google-research-datasets/paws", "labeled_final", "train", "test", "claim", "paraphrase", "the two sentences mean the same thing",
         lambda r: {"first": r["sentence1"], "second": r["sentence2"]}, lambda r: int(r["label"]) == 1, held_out=True, family="paraphrase"),
    Spec("stsb", "nyu-mll/glue", "stsb", "train", "validation", "grade", "similarity", "how similar in meaning the two sentences are",
         lambda r: {"first": r["sentence1"], "second": r["sentence2"]}, lambda r: str(min(5, int(round(float(r["label"]))))), labels=("0", "1", "2", "3", "4", "5"), family="ordinal"),
    Spec("medical_pairs", "curaihealth/medical_questions_pairs", None, "train", "train", "claim", "same_question", "the two medical questions are asking the same thing",
         lambda r: {"first": r["question_1"], "second": r["question_2"]}, lambda r: int(r["label"]) == 1, family="paraphrase", note="one split; disjoint halves by seed"),
    # ---- intent ----
    Spec("clinc", "clinc/clinc_oos", "plus", "train", "test", "pick", "intent", "what the user wants",
         _text("text"), lambda r: r["intent"], labels=(), family="intent", note="150 intents plus out-of-scope; names from features"),
    Spec("snips", "benayas/snips", None, "train", "test", "pick", "intent", "what the user wants",
         _text("text"), lambda r: r["category"], labels=("AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"), family="intent"),
    # ---- toxicity / hate / offense ----
    Spec("hate_offensive", "SetFit/hate_speech_offensive", None, "train", "test", "pick", "content", "how this tweet should be classified",
         _text("text"), lambda r: r["label_text"], labels=("hate speech", "offensive language", "neither"), family="toxicity"),
    Spec("toxic_conversations", "SetFit/toxic_conversations", None, "train", "test", "claim", "toxic", "this comment is toxic",
         _text("text"), lambda r: int(r["label"]) == 1, balance=True, family="toxicity"),
    Spec("tweet_hate", "cardiffnlp/tweet_eval", "hate", "train", "test", "claim", "hateful", "this tweet is hateful towards immigrants or women",
         _text("text"), lambda r: int(r["label"]) == 1, family="toxicity"),
    Spec("tweet_offensive", "cardiffnlp/tweet_eval", "offensive", "train", "test", "claim", "offensive", "this tweet is offensive",
         _text("text"), lambda r: int(r["label"]) == 1, balance=True, family="toxicity"),
    Spec("ethos", "iamollas/ethos", "binary", "train", "train", "claim", "hateful", "this comment contains hate speech",
         _text("text"), lambda r: int(r["label"]) == 1, family="toxicity", note="one split; disjoint halves by seed"),
    # ---- spam ----
    Spec("sms_spam", "ucirvine/sms_spam", None, "train", "train", "claim", "spam", "this text message is spam",
         _text("sms"), lambda r: int(r["label"]) == 1, balance=True, held_out=True, family="spam", note="one split; disjoint halves by seed"),
    Spec("enron_spam", "SetFit/enron_spam", None, "train", "test", "claim", "spam", "this email is spam",
         lambda r: {"subject": r["subject"], "body": _cut(r["text"])}, lambda r: int(r["label"]) == 1, family="spam"),
    # ---- fact-checking / claims ----
    Spec("liar", "ucsbnlp/liar", None, "train", "test", "grade", "truthfulness", "how truthful this political statement is",
         lambda r: {"speaker": r["speaker"], "statement": r["statement"], "context": r["context"]},
         lambda r: LIAR[int(r["label"])], labels=("pants on fire", "false", "barely true", "half true", "mostly true", "true"), family="factcheck"),
    Spec("health_fact", "ImperialCollegeLondon/health_fact", None, "train", "test", "pick", "verdict", "the verdict on this health claim given the evidence",
         lambda r: {"claim": r["claim"], "evidence": _cut(r["main_text"])}, lambda r: ("false", "mixture", "true", "unproven")[int(r["label"])],
         labels=("false", "mixture", "true", "unproven"), filt=lambda r: int(r["label"]) >= 0, family="factcheck"),
    Spec("climate_fever", "tdiggelm/climate_fever", None, "test", "test", "pick", "verdict", "what the evidence says about the claim",
         lambda r: {"claim": r["claim"], "evidence": _cut(r["evidences"][0]["evidence"])}, lambda r: ("supports", "refutes", "not enough info", "disputed")[int(r["claim_label"])],
         labels=("supports", "refutes", "not enough info", "disputed"), family="factcheck", note="one split; disjoint halves by seed"),
    # ---- question answering / relevance ----
    Spec("boolq", "google/boolq", None, "train", "validation", "claim", "answer", "the answer to the question, given the passage, is yes",
         lambda r: {"question": r["question"], "passage": _cut(r["passage"])}, lambda r: bool(r["answer"]), held_out=True, family="qa"),
    Spec("wiki_qa", "microsoft/wiki_qa", None, "train", "test", "claim", "answers", "the sentence answers the question",
         lambda r: {"question": r["question"], "sentence": r["answer"]}, lambda r: int(r["label"]) == 1, balance=True, family="qa"),
    # ---- commonsense multiple choice over candidate texts ----
    Spec("piqa", "ybisk/piqa", None, "train", "validation", "pick", "solution", "which solution achieves the goal",
         lambda r: {"goal": r["goal"]}, lambda r: (r["sol1"], r["sol2"])[int(r["label"])], options=lambda r: (r["sol1"], r["sol2"]), family="commonsense"),
    Spec("hellaswag", "Rowan/hellaswag", None, "train", "validation", "pick", "continuation", "which continuation is the most plausible",
         lambda r: {"context": r["ctx"]}, lambda r: r["endings"][int(r["label"])], options=lambda r: tuple(r["endings"]), family="commonsense"),
    Spec("copa", "aps/super_glue", "copa", "train", "validation", "pick", "alternative", "which alternative is the more plausible cause or effect",
         lambda r: {"premise": r["premise"], "asking_for": r["question"]}, lambda r: (r["choice1"], r["choice2"])[int(r["label"])], options=lambda r: (r["choice1"], r["choice2"]), family="commonsense"),
    Spec("winogrande", "allenai/winogrande", "winogrande_xl", "train", "validation", "pick", "fills_blank", "which option fills the blank",
         lambda r: {"sentence": r["sentence"]}, lambda r: (r["option1"], r["option2"])[int(r["answer"]) - 1], options=lambda r: (r["option1"], r["option2"]), family="commonsense"),
    Spec("commonsense_qa", "tau/commonsense_qa", None, "train", "validation", "pick", "answer", "the answer to the question",
         lambda r: {"question": r["question"]}, lambda r: r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])], options=lambda r: tuple(r["choices"]["text"]), held_out=True, family="commonsense"),
    Spec("openbookqa", "allenai/openbookqa", "main", "train", "test", "pick", "answer", "the answer to the question",
         lambda r: {"question": r["question_stem"]}, lambda r: r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])], options=lambda r: tuple(r["choices"]["text"]), family="commonsense"),
    Spec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "test", "pick", "answer", "the answer to the science question",
         lambda r: {"question": r["question"]}, lambda r: r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])], options=lambda r: tuple(r["choices"]["text"]), family="commonsense"),
    Spec("sciq", "allenai/sciq", None, "train", "test", "pick", "answer", "the answer to the science question, given the support",
         lambda r: {"question": r["question"], "support": _cut(r["support"])}, lambda r: r["correct_answer"],
         options=lambda r: tuple(sorted([r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]])), family="commonsense"),
    Spec("swag", "allenai/swag", "regular", "train", "validation", "pick", "continuation", "which continuation is the most plausible",
         lambda r: {"context": r["startphrase"]}, lambda r: (r["ending0"], r["ending1"], r["ending2"], r["ending3"])[int(r["label"])],
         options=lambda r: (r["ending0"], r["ending1"], r["ending2"], r["ending3"]), family="commonsense"),
    Spec("cosmos_qa", "allenai/cosmos_qa", None, "train", "validation", "pick", "answer", "the answer to the question about the passage",
         lambda r: {"passage": _cut(r["context"]), "question": r["question"]}, lambda r: (r["answer0"], r["answer1"], r["answer2"], r["answer3"])[int(r["label"])],
         options=lambda r: (r["answer0"], r["answer1"], r["answer2"], r["answer3"]), family="commonsense"),
    Spec("social_iqa", "allenai/social_i_qa", None, "train", "validation", "pick", "answer", "the answer to the question about the situation",
         lambda r: {"situation": r["context"], "question": r["question"]}, lambda r: (r["answerA"], r["answerB"], r["answerC"])[int(r["label"]) - 1],
         options=lambda r: (r["answerA"], r["answerB"], r["answerC"]), family="commonsense"),
    Spec("race", "ehovy/race", "middle", "train", "test", "pick", "answer", "the answer to the question about the passage",
         lambda r: {"passage": _cut(r["article"]), "question": r["question"]}, lambda r: r["options"]["ABCD".index(r["answer"])], options=lambda r: tuple(r["options"]), family="qa"),
    Spec("dream", "dataset-org/dream", None, "train", "test", "pick", "answer", "the answer to the question about the dialogue",
         lambda r: {"dialogue": _cut("\n".join(r["dialogue"])), "question": r["question"]}, lambda r: r["answer"], options=lambda r: tuple(r["choice"]), family="qa"),
    # ---- acceptability ----
    Spec("cola", "nyu-mll/glue", "cola", "train", "validation", "claim", "acceptable", "this sentence is grammatically acceptable",
         _text("sentence"), lambda r: int(r["label"]) == 1, balance=True, family="acceptability"),
    # ---- emotion ----
    Spec("go_emotions", "google-research-datasets/go_emotions", "simplified", "train", "test", "pick", "emotion", "the main emotion this comment expresses",
         _text("text"), lambda r: GO_EMOTIONS[int(r["labels"][0])], labels=(), filt=lambda r: len(r["labels"]) == 1, family="emotion", note="single-label rows only; 28 emotions"),
    Spec("tweet_emotion", "cardiffnlp/tweet_eval", "emotion", "train", "test", "pick", "emotion", "the emotion this tweet expresses",
         _text("text"), lambda r: ("anger", "joy", "optimism", "sadness")[int(r["label"])], labels=("anger", "joy", "optimism", "sadness"), family="emotion"),
    Spec("emotion_setfit", "SetFit/emotion", None, "train", "test", "pick", "emotion", "the emotion this message expresses",
         _text("text"), lambda r: r["label_text"], labels=("sadness", "joy", "love", "anger", "fear", "surprise"), family="emotion"),
    # ---- sarcasm / irony ----
    Spec("tweet_irony", "cardiffnlp/tweet_eval", "irony", "train", "test", "claim", "ironic", "this tweet is ironic",
         _text("text"), lambda r: int(r["label"]) == 1, held_out=True, family="sarcasm"),
    # ---- stance ----
    Spec("stance_abortion", "cardiffnlp/tweet_eval", "stance_abortion", "train", "test", "pick", "stance", "the tweet's stance on abortion",
         _text("text"), lambda r: ("none", "against", "favor")[int(r["label"])], labels=("none", "against", "favor"), family="stance"),
    Spec("stance_climate", "cardiffnlp/tweet_eval", "stance_climate", "train", "test", "pick", "stance", "the tweet's stance on climate change being a concern",
         _text("text"), lambda r: ("none", "against", "favor")[int(r["label"])], labels=("none", "against", "favor"), family="stance"),
    Spec("stance_feminist", "cardiffnlp/tweet_eval", "stance_feminist", "train", "test", "pick", "stance", "the tweet's stance on the feminist movement",
         _text("text"), lambda r: ("none", "against", "favor")[int(r["label"])], labels=("none", "against", "favor"), family="stance"),
    # ---- code ----
    Spec("code_defect", "google/code_x_glue_cc_defect_detection", None, "train", "test", "claim", "defective", "this C function contains a defect or vulnerability",
         _text("func"), lambda r: bool(r["target"]), family="code"),
    Spec("code_clone", "google/code_x_glue_cc_clone_detection_big_clone_bench", None, "train", "test", "claim", "clone", "the two Java functions do the same thing",
         lambda r: {"first": _cut(r["func1"]), "second": _cut(r["func2"])}, lambda r: bool(r["label"]), balance=True, family="code"),
    # ---- misc: review helpfulness / questions ----
    Spec("ade", "SetFit/ade_corpus_v2_classification", None, "train", "test", "claim", "adverse_effect", "this sentence reports an adverse drug effect",
         _text("text"), lambda r: int(r["label"]) == 1, balance=True, family="qa"),
]

LIAR = {0: "false", 1: "half true", 2: "mostly true", 3: "true", 4: "barely true", 5: "pants on fire"}
GO_EMOTIONS = ("admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral")


def _fixed_labels(spec, rows):
    """Answer sets that come from the dataset's own feature names (many-intent sets)."""
    if spec.labels:
        return spec.labels
    if spec.name == "go_emotions":
        return GO_EMOTIONS
    from datasets import load_dataset_builder
    feats = load_dataset_builder(spec.path, spec.config).info.features
    col = "intent" if "intent" in feats else "label"
    return tuple(feats[col].names)


def _question(spec, row, labels):
    if spec.kind == "pick":
        opts = spec.options(row) if spec.options else labels
        return Pick(spec.qid, spec.instructions, tuple(opts))
    if spec.kind == "grade":
        return Grade(spec.qid, spec.instructions, tuple(labels))
    return Claim(spec.qid, spec.instructions)


def _answer(spec, row, labels, feat_names):
    v = spec.label(row)
    if spec.kind == "claim":
        return bool(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return feat_names[v]
    return str(v)


def load(spec, n, seed=0, split="all", uncertain=0.0):
    part = _splits(split, spec.train, spec.eval)
    same = spec.train == spec.eval
    # one-split sources: the train draw and the eval draw use different seeds and disjoint halves
    draw_seed = seed if not same else (seed + (0 if split == "train" else 7919))
    want = 2 * n + 200 if same else n
    pool = want * (30 if spec.balance else 1)
    rows = _stream(spec.path, spec.config, part, pool, draw_seed, filt=spec.filt, ordered=spec.ordered)
    if same:
        half = len(rows) // 2
        rows = rows[:half] if split == "train" else rows[half:]
    if spec.balance:
        rows = _balance(rows, n, lambda r: bool(spec.label(r)))
    rows = rows[:n]
    labels = None if spec.options else _fixed_labels(spec, rows)
    feat_names = labels if labels else ()
    out = []
    for r in rows:
        q = _question(spec, r, labels)
        ans = _answer(spec, r, labels, feat_names)
        out.append({"state": spec.state(r), "questions": [q], "labels": {spec.qid: ans}})
    return out


def _loader(spec):
    return lambda n, seed=0, split="all", uncertain=0.0: load(spec, n, seed, split, uncertain)


HUB = {f"hub_{s.name}": _loader(s) for s in SPECS}
BY_NAME = {s.name: s for s in SPECS}
TRAINABLE = [s for s in SPECS if not s.held_out]
HELD_OUT = [s for s in SPECS if s.held_out]


def hub_mixed(n, seed=0, split="all", uncertain=0.0, specs=None):
    """An even draw across every trainable spec (held-out specs never appear), n in total."""
    specs = specs or TRAINABLE
    per = max(1, n // len(specs))
    out = []
    for s in specs:
        try:
            out.extend(load(s, per, seed, split, uncertain))
        except Exception as e:      # a source that is down should not sink the corpus; the audit reports it
            print(f"hub: {s.name} skipped ({type(e).__name__}: {str(e)[:80]})")
    random.Random(seed).shuffle(out)
    return out


HUB["hub_mixed"] = hub_mixed
