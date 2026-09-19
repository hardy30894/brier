"""Public labelled datasets as Pick, Grade and Claim questions. Each loader streams a deterministic
sample from the Hugging Face Hub (parquet-backed sets only, no dataset scripts) and returns
{state, questions, labels} like every other task. Soft labels where the source has them."""
import random

from brier.questions import Pick, Grade, Claim

MAX_CHARS = 1500


def _stream(name, cfg, split, n, seed, filt=None):
    from datasets import load_dataset
    ds = load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=seed, buffer_size=5000)
    out = []
    for row in ds:
        if filt is None or filt(row):
            out.append(row)
            if len(out) >= n:
                break
    return out


def _cut(t):
    t = t if isinstance(t, str) else str(t)
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS] + " …"


def _splits(split, train, evals):
    return train if split == "train" else evals


AG = ("World", "Sports", "Business", "Sci/Tech")
def ag_news(n, seed=0, split="all", uncertain=0.0):
    q = [Pick("topic", "the topic of this news article", AG)]
    rows = _stream("fancyzhx/ag_news", None, _splits(split, "train", "test"), n, seed)
    return [{"state": _cut(r["text"]), "questions": q, "labels": {"topic": AG[r["label"]]}} for r in rows]


DB = ("Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork")
def dbpedia(n, seed=0, split="all", uncertain=0.0):
    q = [Pick("kind", "what kind of thing this encyclopedia article describes", DB)]
    rows = _stream("fancyzhx/dbpedia_14", None, _splits(split, "train", "test"), n, seed)
    return [{"state": {"title": r["title"], "article": _cut(r["content"])}, "questions": q, "labels": {"kind": DB[r["label"]]}} for r in rows]


SST = ("very negative", "negative", "neutral", "positive", "very positive")
def sst5(n, seed=0, split="all", uncertain=0.0):
    q = [Grade("sentiment", "the sentiment of this movie review sentence", SST)]
    rows = _stream("SetFit/sst5", None, _splits(split, "train", "test"), n, seed)
    return [{"state": r["text"], "questions": q, "labels": {"sentiment": SST[r["label"]]}} for r in rows]


STARS = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")
def yelp(n, seed=0, split="all", uncertain=0.0):
    q = [Grade("stars", "the star rating this review gives", STARS)]
    rows = _stream("Yelp/yelp_review_full", None, _splits(split, "train", "test"), n, seed)
    return [{"state": _cut(r["text"]), "questions": q, "labels": {"stars": STARS[r["label"]]}} for r in rows]


NLI = ("entailment", "neutral", "contradiction")
def _nli(name, cfg, train_split, eval_split):
    def gen(n, seed=0, split="all", uncertain=0.0):
        q = [Pick("relation", "how the hypothesis relates to the premise", NLI), Claim("entailed", "the hypothesis follows from the premise")]
        rows = _stream(name, cfg, _splits(split, train_split, eval_split), n, seed, filt=lambda r: r["label"] in (0, 1, 2))
        return [{"state": {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "questions": q, "labels": {"relation": NLI[r["label"]], "entailed": r["label"] == 0}} for r in rows]
    return gen
snli = _nli("stanfordnlp/snli", None, "train", "validation")
mnli = _nli("nyu-mll/glue", "mnli", "train", "validation_matched")


def qqp(n, seed=0, split="all", uncertain=0.0):
    q = [Claim("duplicate", "the two questions are asking the same thing")]
    rows = _stream("nyu-mll/glue", "qqp", _splits(split, "train", "validation"), n, seed)
    return [{"state": {"question_1": r["question1"], "question_2": r["question2"]}, "questions": q, "labels": {"duplicate": r["label"] == 1}} for r in rows]


def civil(n, seed=0, split="all", uncertain=0.0):
    """Toxicity is the fraction of annotators who flagged the comment: a soft label for training,
    a 0.5 cut for evaluation."""
    q = [Claim("toxic", "this comment is toxic")]
    rows = _stream("google/civil_comments", None, _splits(split, "train", "validation"), n, seed, filt=lambda r: len(r["text"]) > 20)
    out = []
    for r in rows:
        t = float(r["toxicity"])
        label = {True: t, False: 1 - t} if split == "train" and 0.1 < t < 0.9 else (t >= 0.5)
        out.append({"state": _cut(r["text"]), "questions": q, "labels": {"toxic": label}})
    return out


def msmarco(n, seed=0, split="all", uncertain=0.0):
    q = [Claim("answers", "the passage contains the answer to the query")]
    rows = _stream("microsoft/ms_marco", "v1.1", _splits(split, "train", "validation"), n, seed, filt=lambda r: any(r["passages"]["is_selected"]))
    rng = random.Random(seed)
    out = []
    for r in rows:
        ps, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        i = rng.randrange(len(ps))
        out.append({"state": {"query": r["query"], "passage": _cut(ps[i])}, "questions": q, "labels": {"answers": bool(sel[i])}})
    return out


EMO = ("sadness", "joy", "love", "anger", "fear", "surprise")
def emotion(n, seed=0, split="all", uncertain=0.0):
    q = [Pick("emotion", "the emotion this message expresses", EMO)]
    rows = _stream("dair-ai/emotion", None, _splits(split, "train", "validation"), n, seed)
    return [{"state": r["text"], "questions": q, "labels": {"emotion": EMO[r["label"]]}} for r in rows]


def imdb(n, seed=0, split="all", uncertain=0.0):
    q = [Claim("positive", "this review recommends the film")]
    rows = _stream("stanfordnlp/imdb", None, _splits(split, "train", "test"), n, seed)
    return [{"state": _cut(r["text"]), "questions": q, "labels": {"positive": r["label"] == 1}} for r in rows]


MASSIVE = ("alarm_set", "weather_query", "play_music", "calendar_set", "email_sendemail", "transport_ticket", "general_quirky", "qa_factoid", "iot_hue_lightoff", "cooking_recipe")
def massive(n, seed=0, split="all", uncertain=0.0):
    q = [Pick("intent", "what the user wants the assistant to do", MASSIVE)]
    rows = _stream("SetFit/amazon_massive_intent_en-US", None, _splits(split, "train", "validation"), n, seed, filt=lambda r: r["label_text"] in MASSIVE)
    return [{"state": r["text"], "questions": q, "labels": {"intent": r["label_text"]}} for r in rows]


PUBLIC = {"ag_news": ag_news, "dbpedia": dbpedia, "sst5": sst5, "yelp": yelp, "snli": snli, "mnli": mnli, "qqp": qqp,
          "civil": civil, "msmarco": msmarco, "emotion": emotion, "imdb": imdb, "massive": massive}
