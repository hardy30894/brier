"""The pointer head: one forward pass per request, every question and every option scored from
the prompt's own hidden states.

The prompt lists each question with its options. For a question, the hidden state at the end
of its line is the query; the hidden state at the last token of each listed option is that
option's key; the score is a scaled bilinear form of the two. No candidate rows, no vocabulary
projection: the cost of a request is one pass over the prompt, whatever the number of questions
or options. The head only scores options that were listed, so the type guarantee is kept."""
import json
import queue
import threading
import time
from dataclasses import dataclass

import torch

from .questions import render_state, Pick, Grade, Claim, Answer, Answers, confidence_of
from .scorer import softmax


class PointerHead(torch.nn.Module):
    """Bilinear option scoring, plus an ordinal term for Grade questions: the query predicts a
    position on the scale and options are scored by their distance from it, so the head can say
    "somewhere around medium" even though the option words carry no order of their own."""

    def __init__(self, d: int, dh: int = 256):
        super().__init__()
        self.q = torch.nn.Linear(d, dh, bias=False)
        self.k = torch.nn.Linear(d, dh, bias=False)
        self.scale = dh ** -0.5
        self.dh = dh
        self.mu = torch.nn.Linear(d, 1)
        self.log_lam = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, h_query, h_keys, ordinal=False):
        """h_query: (d,), h_keys: (n, d) -> (n,) scores."""
        s = (self.k(h_keys.float()) @ self.q(h_query.float())) * self.scale
        if ordinal and h_keys.shape[0] > 1:
            n = h_keys.shape[0]
            pos = torch.arange(n, device=s.device, dtype=s.dtype) / (n - 1)
            mu = torch.sigmoid(self.mu(h_query.float())).squeeze(-1)
            s = s - torch.exp(self.log_lam) * (pos - mu) ** 2 * n
        return s


@dataclass
class Layout:
    text: str
    query_char: list[int]            # per question: char index (exclusive end) of the question line
    option_chars: list[list[int]]    # per question, per option: char index (exclusive end) of the option mention


def render(state, questions) -> Layout:
    """The same prompt the row scorer uses, but with the character position of every option and
    of every question's end recorded as it is written."""
    head = ("Read the state and answer every question with exactly one of its allowed answers. "
            "Reply with a single JSON object keyed by question id and nothing else.\n\n"
            f"State:\n{render_state(state)}\n\nQuestions:\n")
    parts, qc, oc = [head], [], []
    pos = len(head)
    for q in questions:
        if isinstance(q, Claim):
            clar = "" if (q.true, q.false) == ("yes", "no") else f" (yes means: {q.true}; no means: {q.false})"
            line = f'- "{q.id}" ({q.instructions}) '
            ends = []
            for i, lab in enumerate(("yes", "no")):
                line += lab
                ends.append(pos + len(line))
                line += " or " if i == 0 else ""
            line += clar
        else:
            lead = f'- "{q.id}" ({q.instructions}) ' + ("one of, from lowest to highest: " if isinstance(q, Grade) else "one of: ")
            line = lead
            ends = []
            labels = q.labels
            for i, lab in enumerate(labels):
                line += lab
                ends.append(pos + len(line))
                if i < len(labels) - 1:
                    line += ", "
        line += "\n"
        parts.append(line)
        oc.append(ends)
        qc.append(pos + len(line))
        pos += len(line)
    return Layout("".join(parts), qc, oc)


def char_to_token(offsets, char_end):
    """Index of the last token whose span ends at or before char_end."""
    best = 0
    for i, (a, b) in enumerate(offsets):
        if b <= char_end and b > 0:
            best = i
        elif a >= char_end:
            break
    return best


def encode(scorer, state, questions):
    lay = render(state, questions)
    full = scorer.chat_prefix(lay.text)
    base = full.find(lay.text)
    enc = scorer.tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    qpos = [char_to_token(offs, base + c) for c in lay.query_char]
    opos = [[char_to_token(offs, base + c) for c in row] for row in lay.option_chars]
    return ids, qpos, opos


def decoder(model):
    """The transformer body without the vocabulary head, through PEFT and DDP wrappers."""
    m = model
    for _ in range(6):
        if hasattr(m, "embed_tokens"):
            return m
        m = getattr(m, "module", None) or getattr(m, "base_model", None) or getattr(m, "model", None) or m
        if m is None:
            break
    raise RuntimeError("no decoder body found")


BUCKETS = (128, 256, 512, 1024, 2048)
BATCH_BUCKETS = (1, 2, 4, 8, 16)


def bucket(n, buckets):
    return next((b for b in buckets if b >= n), n)


class Captured:
    """The transformer body replayed from a CUDA graph, one graph per input shape. A capture
    costs a few forward passes; a replay skips every per-layer kernel launch, which at small
    batch sizes is most of the time. Inputs are padded to fixed buckets so the number of graphs
    stays small. Outputs are the eager outputs to the bit."""

    def __init__(self, body):
        self.body = body
        self.graphs = {}
        self._static_rope()

    def _static_rope(self):
        """Some rotary embeddings check the sequence length against a threshold on every call,
        which reads a device value and cannot run inside a capture. Below the threshold the
        check is a no-op, and every bucket is below it, so call the undecorated forward."""
        import inspect
        for m in self.body.modules():
            if getattr(m, "rope_type", None) != "longrope":
                continue
            limit = getattr(getattr(m, "config", None), "original_max_position_embeddings", None)
            if limit is not None and max(BUCKETS) > limit:
                continue
            raw = inspect.unwrap(type(m).forward)
            m.forward = raw.__get__(m, type(m))

    def capture(self, input_ids):
        static = input_ids.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.body(input_ids=static)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.body(input_ids=static).last_hidden_state
        return graph, static, out

    def __call__(self, input_ids):
        key = tuple(input_ids.shape)
        if key not in self.graphs:
            self.graphs[key] = self.capture(input_ids)
        graph, static, out = self.graphs[key]
        static.copy_(input_ids)
        graph.replay()
        return out


def captured_body(scorer):
    if getattr(scorer, "_captured_body", None) is None:
        scorer._captured_body = Captured(decoder(scorer.model))
    return scorer._captured_body


def hidden_states(scorer, rows, grad=False):
    """One forward over a batch of token-id rows, right-padded to a common length; returns
    (B, L, d) hidden states. Padding sits after every real token, so under causal attention
    the real positions are unchanged by it and no attention mask is needed. Replayed from a
    CUDA graph, with bucketed shapes, when the scorer says so."""
    n = max(len(r) for r in rows)
    cap = getattr(scorer, "capture", False) and not grad and scorer.device == "cuda"
    L = bucket(n, BUCKETS) if cap else n
    B = bucket(len(rows), BATCH_BUCKETS) if cap else len(rows)
    pad = scorer.tok.pad_token_id
    x = torch.tensor([r + [pad] * (L - len(r)) for r in rows] + [[pad] * L] * (B - len(rows)), device=scorer.device)
    if cap:
        return captured_body(scorer)(x)
    return decoder(scorer.model)(input_ids=x).last_hidden_state


def pointer_scores_batch(scorer, head, requests, grad=False):
    """Several requests, one forward pass. requests: list of (state, questions); returns, per
    request, a list (per question) of score tensors (per option)."""
    encs = [encode(scorer, st, qs) for st, qs in requests]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hs = hidden_states(scorer, [ids for ids, _, _ in encs], grad)
        out = []
        for (st, qs), (ids, qpos, opos), h in zip(requests, encs, hs):
            out.append([head(h[qp], h[torch.tensor(op, device=scorer.device)], ordinal=isinstance(q, Grade)) for q, qp, op in zip(qs, qpos, opos)])
        return out


def pointer_scores(scorer, head, state, questions, grad=False):
    """One forward pass over the prompt, through the transformer body only (no vocabulary
    projection); returns a list (per question) of score tensors (per option)."""
    return pointer_scores_batch(scorer, head, [(state, questions)], grad)[0]


def answers_from_scores(questions, raw, temperatures=None) -> dict:
    temperatures = temperatures or {}
    out = {}
    for q, s in zip(questions, raw):
        s = s.tolist()
        probs = softmax(s, temperatures.get(q.id, 1.0))
        dist = dict(zip(q.labels, probs))
        conf, margin = confidence_of(probs)
        if isinstance(q, Claim):
            out[q.id] = Answer(q.id, "claim", dist["yes"], dist, conf, margin, dict(zip(q.labels, s)))
        else:
            best = max(dist, key=dist.get)
            exp = sum(i * p for i, p in enumerate(probs)) if isinstance(q, Grade) else None
            out[q.id] = Answer(q.id, "grade" if isinstance(q, Grade) else "pick", best, dist, conf, margin, dict(zip(q.labels, s)), expected=exp)
    return out


def ask_pointer(scorer, head, state, questions, temperatures=None) -> Answers:
    t0 = time.perf_counter()
    raw = pointer_scores(scorer, head, state, questions)
    return Answers(answers_from_scores(questions, raw, temperatures), (time.perf_counter() - t0) * 1000)


def warm(scorer, head=None, max_batch=1, max_len=512):
    """Capture one graph per (batch, length) bucket up to the given sizes so the first real
    request of each shape does not pay for its capture."""
    if not getattr(scorer, "capture", False) or scorer.device != "cuda":
        return 0
    pad = scorer.tok.pad_token_id
    n = 0
    with torch.no_grad():
        for L in BUCKETS:
            if L > max_len:
                break
            for B in BATCH_BUCKETS:
                if B > bucket(max_batch, BATCH_BUCKETS):
                    break
                captured_body(scorer)(torch.full((B, L), pad, device=scorer.device))
                n += 1
    torch.cuda.synchronize()
    return n


class Batcher:
    """Concurrent requests share forward passes. Callers submit from any thread and block for
    their own Answers; one worker drains the queue, waits a few milliseconds for company, and
    runs every waiting Pick/Claim set through one padded pass. Grade questions still go through
    the row scorer, serialised on the same worker so the model is never used from two threads."""

    def __init__(self, scorer, max_batch=8, wait_ms=2.0):
        self.scorer = scorer
        self.max_batch = max(1, max_batch)
        self.wait = wait_ms / 1000
        self.q: queue.Queue = queue.Queue()
        self.batches = 0
        self.served = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def ask(self, state, questions, temperatures=None) -> Answers:
        done = threading.Event()
        job = {"state": state, "questions": questions, "temperatures": temperatures, "done": done, "out": None, "err": None, "t0": time.perf_counter()}
        self.q.put(job)
        done.wait()
        if job["err"] is not None:
            raise job["err"]
        return job["out"]

    def _take(self):
        jobs = [self.q.get()]
        deadline = time.perf_counter() + self.wait
        while len(jobs) < self.max_batch:
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            try:
                jobs.append(self.q.get(timeout=left))
            except queue.Empty:
                break
        return jobs

    def _run(self):
        from .questions import _ask_rows
        sc = self.scorer
        while True:
            jobs = self._take()
            try:
                use_ptr = sc.pointer is not None
                fast = [[q for q in j["questions"] if use_ptr and (sc.pointer_grade or not isinstance(q, Grade))] for j in jobs]
                slow = [[q for q in j["questions"] if q not in f] for j, f in zip(jobs, fast)]
                raws = pointer_scores_batch(sc, sc.pointer, [(j["state"], f) for j, f in zip(jobs, fast) if f]) if any(fast) else []
                it = iter(raws)
                for j, f, s in zip(jobs, fast, slow):
                    out = answers_from_scores(f, next(it), j["temperatures"]) if f else {}
                    if s:
                        out.update(_ask_rows(sc, j["state"], s, j["temperatures"]).answers)
                    j["out"] = Answers({q.id: out[q.id] for q in j["questions"]}, (time.perf_counter() - j["t0"]) * 1000)
                self.batches += 1
                self.served += len(jobs)
            except Exception as e:
                for j in jobs:
                    j["err"] = e
            finally:
                for j in jobs:
                    j["done"].set()
