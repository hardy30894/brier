"""RAG passage relevance: does this passage answer the question? A Noul."""
import random

from brier.questions import Noul

QUESTIONS = [Noul("answers", "the passage contains the information needed to answer the question")]

FACTS = [
    ("What year was the company founded?", "The company was founded in {year} by two former engineers.", "The company's headquarters moved to a larger office in {year}."),
    ("How many employees does the company have?", "As of last quarter the company employs {num} people across four offices.", "The company hosted {num} attendees at its annual conference."),
    ("What is the refund window?", "Customers may request a full refund within {num} days of purchase.", "Customers receive a {pct}% discount when paying annually."),
    ("Which database does the service use?", "All records are stored in {db}, replicated across three regions.", "The team evaluated several options, including {db}, during the migration planning."),
    ("What is the maximum file upload size?", "Uploads are capped at {num} MB per file on the free tier.", "The changelog notes that {num} bugs were fixed in the release."),
    ("Who is the CEO?", "{name} has served as chief executive since the company's founding.", "{name} joined the advisory board earlier this year."),
    ("What is the API rate limit?", "Each API key may make up to {num} requests per minute.", "The API documentation was viewed {num} times last month."),
    ("When does the free trial end?", "The free trial lasts {num} days and converts automatically unless cancelled.", "The onboarding webinar runs for {num} minutes."),
]
FILL = {"year": ["2014", "2017", "2019", "2021"], "num": ["14", "30", "250", "1200"], "pct": ["10", "15", "20"], "db": ["PostgreSQL", "DynamoDB", "MongoDB"], "name": ["Priya Raman", "Marcus Hale", "Dana Whitfield"]}
FILLER = ["The office kitchen was recently renovated.", "Quarterly planning happens every March.", "The support team works in two shifts.", "A new logo was announced on the blog."]


def _fill(s, rng):
    for k, vs in FILL.items():
        s = s.replace("{" + k + "}", rng.choice(vs))
    return s


def generate(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        q, yes, near = rng.choice(FACTS)
        r = rng.random()
        if r < 0.45:
            passage, label = _fill(yes, rng), True
        elif r < 0.8:
            passage, label = _fill(near, rng), False
        else:
            passage, label = _fill(rng.choice(FACTS)[1], rng), False
            if passage.split()[0] == yes.split()[0]:
                passage, label = _fill(near, rng), False
        parts = [passage] + rng.sample(FILLER, rng.randint(0, 2))
        rng.shuffle(parts)
        out.append({"state": {"question": q, "passage": " ".join(parts)}, "questions": QUESTIONS, "labels": {"answers": label}})
    return out
