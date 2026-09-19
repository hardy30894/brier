"""A small HTTP server that puts the scorer behind two endpoints and serves the live demo page.

    python3 -m brier.serve --model microsoft/Phi-4-mini-instruct --adapter adapters/<q> \\
        --grid-model Qwen/Qwen3-0.6B --grid-adapter adapters/<rl> --calibration results/cal_tickets_<tag>.json

    python3 -m brier.serve --mock          # no model; random distributions, for developing the page

POST /ask        {"state": str|obj, "questions": [{"kind": "choice"|"score"|"claim", "id", "instructions", "options"|"levels"|"true"/"false"}]}
POST /grid/new   {}                      -> a fresh game
POST /grid/step  {"game": <as returned>} -> the move distribution and the game after applying the top move
GET  /            the demo page; GET /health
"""
import argparse
import json
import math
import os
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_questions(items):
    from .questions import Choice, Score, Claim
    qs = []
    for it in items:
        kind = it.get("kind", "choice")
        qid, instr = str(it["id"]).strip(), str(it.get("instructions", "")).strip()
        if kind == "choice":
            opts = it["options"]
            qs.append(Choice(qid, instr, tuple(opts) if isinstance(opts, list) else dict(opts)))
        elif kind == "score":
            lv = it["levels"]
            qs.append(Score(qid, instr, tuple(lv) if isinstance(lv, list) else dict(lv)))
        else:
            qs.append(Claim(qid, instr, it.get("true", "yes"), it.get("false", "no")))
    return qs


class Engine:
    def __init__(self, a):
        self.mock = a.mock
        self.cal = None
        self.names = {"questions": a.model + (f" + {os.path.basename(a.adapter)}" if a.adapter else ""),
                      "gridworld": (a.grid_model or a.model) + (f" + {os.path.basename(a.grid_adapter)}" if a.grid_adapter else "")}
        if self.mock:
            return
        from .__main__ import load_scorer
        from .calibrate import Calibration
        self.q_scorer = load_scorer(a.model, a.adapter)
        if a.grid_model or a.grid_adapter:
            self.g_scorer = load_scorer(a.grid_model or a.model, a.grid_adapter)
        else:
            self.g_scorer = self.q_scorer
        if a.calibration and os.path.exists(a.calibration):
            self.cal = Calibration.load(a.calibration)

    def ask(self, state, items):
        qs = parse_questions(items)
        t0 = time.perf_counter()
        if self.mock:
            out = []
            for q in qs:
                raw = [random.gauss(0, 1.5) for _ in q.labels]
                m = max(raw); z = [math.exp(r - m) for r in raw]; s = sum(z)
                probs = dict(zip(q.labels, [v / s for v in z]))
                top = max(probs, key=probs.get)
                kind = "claim" if q.__class__.__name__ == "Claim" else q.__class__.__name__.lower()
                out.append({"id": q.id, "kind": kind, "value": probs["yes"] if kind == "claim" else top, "probabilities": probs,
                            "confidence": probs[top], "tier": None, "expected": None})
            time.sleep(0.05)
            return {"answers": out, "latency_ms": (time.perf_counter() - t0) * 1000, "model": self.names["questions"]}
        from .questions import ask
        temps = self.cal.temperatures if self.cal else None
        ans = ask(self.q_scorer, state, qs, temperatures=temps)
        if self.cal:
            self.cal.apply(ans)
        out = [{"id": a.id, "kind": a.kind, "value": a.value, "probabilities": a.probabilities, "confidence": a.confidence,
                "margin": a.margin, "expected": a.expected, "tier": a.tier} for a in ans]
        return {"answers": out, "latency_ms": ans.latency_ms, "model": self.names["questions"]}

    def generate(self, state, items):
        """The baseline, for the race: the same weights writing the answers as JSON."""
        from .schema import Schema, Enum, Bool
        from .questions import render_state
        fields = []
        for it in items:
            kind = it.get("kind", "choice")
            if kind == "claim":
                fields.append(Bool(it["id"], it.get("instructions", "")))
            else:
                fields.append(Enum(it["id"], tuple(it.get("options") or it.get("levels")), it.get("instructions", "")))
        schema = Schema(tuple(fields))
        if self.mock:
            time.sleep(1.2)
            text = "{" + ", ".join(f'"{f.name}": "{f.choices[0]}"' for f in fields) + ', "confidence": 0.9}'
            return {"text": text, "latency_ms": 1200.0, "type_error": False, "model": self.names["questions"]}
        d = self.q_scorer.generate_json(render_state(state), schema)
        return {"text": d.text, "latency_ms": d.latency_ms, "type_error": d.type_error, "model": self.names["questions"]}

    def grid_new(self):
        sys.path.insert(0, str(ROOT))
        from tasks import gridworld
        ex = gridworld.generate_q(1, seed=random.randrange(10 ** 9))[0]
        grid, player, goal, monster = gridworld.parse_state(ex)
        return {"grid": ["".join(r) for r in grid], "player": list(player), "exit": list(goal), "monster": list(monster), "status": "ok", "steps": 0}

    def grid_step(self, game):
        sys.path.insert(0, str(ROOT))
        from tasks import gridworld
        from .questions import ask
        grid = [list(r) for r in game["grid"]]
        player, goal, monster = tuple(game["player"]), tuple(game["exit"]), tuple(game["monster"])
        state = gridworld.render_state(grid, player, goal, monster)
        q = gridworld.QUESTIONS[0]
        t0 = time.perf_counter()
        if self.mock:
            goods = [m for m in gridworld.DIRS if gridworld.outcome(grid, player, goal, monster, m)]
            probs = {m: (0.55 if m in goods else 0.15) for m in gridworld.DIRS}
            s = sum(probs.values()); probs = {k: v / s for k, v in probs.items()}
            time.sleep(0.04)
            lat = (time.perf_counter() - t0) * 1000
        else:
            ans = ask(self.g_scorer, state, [q])
            probs, lat = ans["move"].probabilities, ans.latency_ms
        move = max(probs, key=probs.get)
        optimal = bool(gridworld.outcome(grid, player, goal, monster, move))
        new_player, status = gridworld.step(grid, player, monster, move)
        return {"probabilities": probs, "move": move, "optimal": optimal, "latency_ms": lat, "model": self.names["gridworld"],
                "game": {"grid": game["grid"], "player": list(new_player), "exit": list(goal), "monster": list(monster), "status": status, "steps": game.get("steps", 0) + 1}}


def make_handler(engine, page):
    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/static/"):
                f = (ROOT / "docs" / os.path.basename(self.path)).resolve()
                if f.exists() and f.suffix in (".png", ".svg", ".css", ".js"):
                    body = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", {"png": "image/png", "svg": "image/svg+xml", "css": "text/css", "js": "text/javascript"}[f.suffix[1:]])
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(404, {"error": "not found"})
            elif self.path == "/health":
                self._json(200, {"ok": True, "mock": engine.mock, "models": engine.names, "calibrated": engine.cal is not None})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/ask":
                    self._json(200, engine.ask(data.get("state", ""), data.get("questions", [])))
                elif self.path == "/generate":
                    self._json(200, engine.generate(data.get("state", ""), data.get("questions", [])))
                elif self.path == "/grid/new":
                    self._json(200, engine.grid_new())
                elif self.path == "/grid/step":
                    self._json(200, engine.grid_step(data["game"]))
                else:
                    self._json(404, {"error": "not found"})
            except Exception as e:
                self._json(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--grid-model", default=None)
    ap.add_argument("--grid-adapter", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--page", default=str(ROOT / "docs" / "live.html"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    engine = Engine(a)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(engine, Path(a.page)))
    print(f"serving on :{a.port}  models={engine.names}  mock={engine.mock}", file=sys.stderr)
    srv.serve_forever()
