import json
import math
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import Schema, Decision, FieldDecision

PREFIX = (
    "You are a decision function. Read the input and answer every field with exactly one of its allowed values.\n\n"
    "Input:\n{context}\n\nFields:\n{fields}\n\nAnswers:\n"
)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Scorer:
    def __init__(self, model_id="Qwen/Qwen3-0.6B", device=None, style="json", debias=False):
        self.device = device or pick_device()
        self.style = style
        self.debias = debias
        self._prior: dict[int, dict[str, list[float]]] = {}
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self._enc_cache: dict[str, list[int]] = {}

    def _enc(self, s: str) -> list[int]:
        ids = self._enc_cache.get(s)
        if ids is None:
            ids = self.tok(s, add_special_tokens=False).input_ids
            self._enc_cache[s] = ids
        return ids

    def _prefix(self, context: str, schema: Schema) -> str:
        if self.style == "raw":
            return PREFIX.format(context=context, fields=schema.render())
        if self.style == "json":
            user = (
                "Read the input and answer every field with exactly one of its allowed values. "
                "Reply with a single JSON object and nothing else.\n\n"
                f"Input:\n{context}\n\nFields:\n{schema.render()}"
            )
        else:
            user = (
                "Read the input and answer every field with exactly one of its allowed values. "
                "Write one line per field in the form `name: value` and nothing else.\n\n"
                f"Input:\n{context}\n\nFields:\n{schema.render()}"
            )
        msgs = [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _prior_logits(self, schema: Schema) -> dict[str, list[float]]:
        key = id(schema)
        if key not in self._prior:
            saved, self.debias = self.debias, False
            runs = [self._score(c, schema) for c in ("N/A", "", "[no input]")]
            self.debias = saved
            self._prior[key] = {
                f.name: [sum(r[f.name][i] for r in runs) / len(runs) for i in range(len(f.choices))]
                for f in schema.fields
            }
        return self._prior[key]

    @torch.no_grad()
    def _score(self, context: str, schema: Schema) -> dict[str, list[float]]:
        prefix_ids = self._enc(self._prefix(context, schema))
        seqs, spans, owner = [], [], []
        for fi, f in enumerate(schema.fields):
            head = prefix_ids + self._enc(f'{{"{f.name}": "' if self.style == "json" else f"{f.name}:")
            for c in f.choices:
                cand = self._enc(c if self.style == "json" else " " + c)
                seqs.append(head + cand)
                spans.append((len(head), len(head) + len(cand)))
                owner.append(fi)
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
        attn = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        rows = torch.cat([torch.full((b - a,), i, device=self.device) for i, (a, b) in enumerate(spans)])
        cols = torch.cat([torch.arange(a - 1, b - 1, device=self.device) for a, b in spans])
        tgt = torch.cat([input_ids[i, a:b] for i, (a, b) in enumerate(spans)])
        lp = torch.log_softmax(logits[rows, cols].float(), dim=-1).gather(1, tgt[:, None]).squeeze(1)
        lens = torch.tensor([b - a for a, b in spans], device=self.device)
        scores = torch.split(lp, lens.tolist())
        scores = [s.sum().item() for s in scores]
        out, k = {}, 0
        for f in schema.fields:
            out[f.name] = scores[k : k + len(f.choices)]
            k += len(f.choices)
        return out

    def decide(self, context: str, schema: Schema) -> Decision:
        t0 = time.perf_counter()
        raw_all = self._score(context, schema)
        prior = self._prior_logits(schema) if self.debias else None
        fields = {}
        for f in schema.fields:
            raw = raw_all[f.name]
            if prior:
                raw = [r - p for r, p in zip(raw, prior[f.name])]
            m = max(raw)
            z = [math.exp(r - m) for r in raw]
            s = sum(z)
            dist = {c: p / s for c, p in zip(f.choices, z)}
            best = max(dist, key=dist.get)
            fields[f.name] = FieldDecision(
                value=f.decode(best), choice=best, prob=dist[best], dist=dist,
                logits={c: r for c, r in zip(f.choices, raw)},
            )
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
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        reply = self.tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        lat = (time.perf_counter() - t0) * 1000
        fields, type_error = {}, False
        obj = None
        m = re.search(r"\{.*\}", reply, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            type_error = True
            obj = {}
        conf = obj.get("confidence", 1.0)
        try:
            conf = float(conf)
            conf = min(max(conf, 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 1.0
        for f in schema.fields:
            choice = f.parse(obj.get(f.name)) if f.name in obj else None
            if choice is None:
                type_error = True
                fields[f.name] = FieldDecision(value=None, choice="", prob=conf, dist={})
            else:
                fields[f.name] = FieldDecision(value=f.decode(choice), choice=choice, prob=conf, dist={choice: conf})
        return Decision(fields=fields, latency_ms=lat, type_error=type_error)
