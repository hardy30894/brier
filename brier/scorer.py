import json
import math
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import Schema, Decision, FieldDecision


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def softmax(scores: list[float], T: float = 1.0) -> list[float]:
    m = max(scores)
    z = [math.exp((s - m) / T) for s in scores]
    t = sum(z)
    return [v / t for v in z]


class Scorer:
    """Scores candidate continuations with a causal LM. The core is score_groups: one batched
    forward pass over every candidate of every group, returning summed log-probabilities."""

    def __init__(self, model_id="Qwen/Qwen3-0.6B", device=None, debias=False, model=None, tok=None, shared_prefix=False):
        self.device = device or pick_device()
        self.debias = debias
        self.shared_prefix = shared_prefix
        self._prior: dict[str, list[list[float]]] = {}
        self.tok = tok or AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if model is None:
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model = model.eval()
        self.head = None
        self._enc_cache: dict[str, list[int]] = {}

    def _enc(self, s: str) -> list[int]:
        ids = self._enc_cache.get(s)
        if ids is None:
            ids = self.tok(s, add_special_tokens=False).input_ids
            self._enc_cache[s] = ids
        return ids

    def chat_prefix(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def build_rows(self, prefix: str, groups: list[tuple[str, list[str]]]):
        prefix_ids = self._enc(prefix)
        seqs, spans = [], []
        for header, cands in groups:
            head = prefix_ids + self._enc(header)
            for c in cands:
                cand = self._enc(c)
                seqs.append(head + cand)
                spans.append((len(head), len(head) + len(cand)))
        return seqs, spans

    def batch(self, seqs):
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
        attn = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
        return input_ids, attn

    def candidate_logprobs(self, logits, input_ids, spans):
        rows = torch.cat([torch.full((b - a,), i, device=self.device) for i, (a, b) in enumerate(spans)])
        cols = torch.cat([torch.arange(a - 1, b - 1, device=self.device) for a, b in spans])
        tgt = torch.cat([input_ids[i, a:b] for i, (a, b) in enumerate(spans)])
        lp = torch.log_softmax(logits[rows, cols].float(), dim=-1).gather(1, tgt[:, None]).squeeze(1)
        return torch.stack([s.sum() for s in torch.split(lp, [b - a for a, b in spans])])

    def scores_from_rows(self, input_ids, attn, spans):
        """LM-head scores per candidate, corrected by the typed head when one is attached."""
        if self.head is None:
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            return self.candidate_logprobs(logits, input_ids, spans)
        out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        lm = self.candidate_logprobs(out.logits, input_ids, spans)
        h = out.hidden_states[-1]
        idx = torch.arange(len(spans), device=self.device)
        h_state = h[idx, torch.tensor([a - 1 for a, b in spans], device=self.device)]
        h_cand = h[idx, torch.tensor([b - 1 for a, b in spans], device=self.device)]
        return self.head(lm, h_state, h_cand)

    @torch.no_grad()
    def score_groups(self, prefix: str, groups: list[tuple[str, list[str]]]) -> list[list[float]]:
        flat = (self._score_cached if self.shared_prefix and self.head is None else self._score_full)(prefix, groups)
        out, k = [], 0
        for _, cands in groups:
            out.append(flat[k : k + len(cands)])
            k += len(cands)
        return out

    def _score_full(self, prefix, groups):
        seqs, spans = self.build_rows(prefix, groups)
        input_ids, attn = self.batch(seqs)
        return self.scores_from_rows(input_ids, attn, spans).tolist()

    def _score_cached(self, prefix, groups):
        """The prompt is run once; its key/value cache is shared by every candidate row, so the
        cost of a decision is one prompt pass plus a batch of short tails."""
        from transformers import DynamicCache
        prefix_ids = self._enc(prefix)
        n_pre = len(prefix_ids)
        tails, spans = [], []
        for header, cands in groups:
            h = self._enc(header)
            for c in cands:
                t = h + self._enc(c)
                tails.append(t)
                spans.append((len(h), len(t)))
        B = len(tails)
        pre = torch.tensor([prefix_ids], device=self.device)
        out = self.model(input_ids=pre, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[0, -1]
        cache.batch_repeat_interleave(B)
        L = max(len(t) for t in tails)
        pad = self.tok.pad_token_id
        tail_ids = torch.tensor([t + [pad] * (L - len(t)) for t in tails], device=self.device)
        attn = torch.tensor([[1] * n_pre + [1] * len(t) + [0] * (L - len(t)) for t in tails], device=self.device)
        pos = torch.arange(n_pre, n_pre + L, device=self.device)[None].expand(B, -1)
        logits = self.model(input_ids=tail_ids, attention_mask=attn, past_key_values=cache, position_ids=pos, use_cache=True).logits
        full = torch.cat([last_logits[None, None].expand(B, 1, -1), logits[:, :-1]], dim=1)
        rows = torch.cat([torch.full((b - a,), i, device=self.device) for i, (a, b) in enumerate(spans)])
        cols = torch.cat([torch.arange(a, b, device=self.device) for a, b in spans])
        tgt = torch.cat([tail_ids[i, a:b] for i, (a, b) in enumerate(spans)])
        lp = torch.log_softmax(full[rows, cols].float(), dim=-1).gather(1, tgt[:, None]).squeeze(1)
        return torch.stack([x.sum() for x in torch.split(lp, [b - a for a, b in spans])]).tolist()

    # ---- schema interface -------------------------------------------------

    def schema_prompt(self, context: str, schema: Schema) -> str:
        return self.chat_prefix(
            "Read the input and answer every field with exactly one of its allowed values. "
            "Reply with a single JSON object and nothing else.\n\n"
            f"Input:\n{context}\n\nFields:\n{schema.render()}"
        )

    def _schema_groups(self, schema: Schema):
        return [(f'{{"{f.name}": "', list(f.choices)) for f in schema.fields]

    def _prior_scores(self, schema: Schema) -> list[list[float]]:
        key = "|".join(f"{f.name}:{','.join(f.choices)}" for f in schema.fields)
        if key not in self._prior:
            runs = [self.score_groups(self.schema_prompt(c, schema), self._schema_groups(schema)) for c in ("N/A", "", "[no input]")]
            self._prior[key] = [[sum(r[g][i] for r in runs) / len(runs) for i in range(len(f.choices))] for g, f in enumerate(schema.fields)]
        return self._prior[key]

    def decide(self, context: str, schema: Schema) -> Decision:
        t0 = time.perf_counter()
        raw = self.score_groups(self.schema_prompt(context, schema), self._schema_groups(schema))
        prior = self._prior_scores(schema) if self.debias else None
        fields = {}
        for g, f in enumerate(schema.fields):
            scores = raw[g] if prior is None else [r - p for r, p in zip(raw[g], prior[g])]
            probs = softmax(scores)
            dist = dict(zip(f.choices, probs))
            best = max(dist, key=dist.get)
            fields[f.name] = FieldDecision(value=f.decode(best), choice=best, prob=dist[best], dist=dist, logits=dict(zip(f.choices, scores)))
        return Decision(fields=fields, latency_ms=(time.perf_counter() - t0) * 1000)

    @torch.no_grad()
    def generate_json(self, context: str, schema: Schema, max_new_tokens=80) -> Decision:
        t0 = time.perf_counter()
        prompt = (
            "Read the input and answer every field with exactly one of its allowed values. "
            "Reply with a single JSON object and nothing else. Include a \"confidence\" number from 0 to 1 "
            "giving the probability that all your answers are correct.\n\n"
            f"Input:\n{context}\n\nFields:\n{schema.render()}\n\nExample shape: {schema.json_example()}"
        )
        text = self.chat_prefix(prompt)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        reply = self.tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        lat = (time.perf_counter() - t0) * 1000
        obj, type_error = None, False
        m = re.search(r"\{.*\}", reply, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            type_error, obj = True, {}
        try:
            conf = min(max(float(obj.get("confidence", 1.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 1.0
        fields = {}
        for f in schema.fields:
            choice = f.parse(obj.get(f.name)) if f.name in obj else None
            if choice is None:
                type_error = True
                fields[f.name] = FieldDecision(value=None, choice="", prob=conf, dist={})
            else:
                fields[f.name] = FieldDecision(value=f.decode(choice), choice=choice, prob=conf, dist={choice: conf})
        return Decision(fields=fields, latency_ms=lat, type_error=type_error)
