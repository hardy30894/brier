"""Play gridworld episodes with a model and record every step's distribution for the demo."""
import argparse
import json

from brier import ask
from brier.__main__ import load_scorer
from tasks import gridworld
from tasks.gridworld import parse_state, step, render_state, outcome

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter", default=None)
ap.add_argument("--episodes", type=int, default=12)
ap.add_argument("--max-steps", type=int, default=10)
ap.add_argument("--out", default="results/trajectories.json")
a = ap.parse_args()
sc = load_scorer(a.model, a.adapter)
q = gridworld.QUESTIONS[0]
episodes = []
for k, ex in enumerate(gridworld.generate_q(a.episodes, seed=4242)):
    grid, player, goal, monster = parse_state(ex)
    steps, status = [], "ok"
    for t in range(a.max_steps):
        state = render_state(grid, player, goal, monster)
        answers = ask(sc, state, [q])
        ans = answers["move"]
        move = ans.value
        good = outcome(grid, player, goal, monster, move)
        steps.append({"grid": state.split("\n")[1:], "player": player, "probs": ans.probabilities, "move": move, "confidence": round(ans.confidence, 3), "optimal": bool(good), "latency_ms": round(answers.latency_ms, 1)})
        player, status = step(grid, player, monster, move)
        if status in ("exit", "dead"):
            break
    episodes.append({"grid": [list(r) for r in ex["state"].split("\n")[1:]], "exit": goal, "monster": monster, "steps": steps, "result": status})
    print(f"episode {k}: {len(steps)} steps -> {status}")
json.dump({"model": a.adapter or a.model, "episodes": episodes}, open(a.out, "w"))
print("wrote", a.out)
