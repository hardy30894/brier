import random
from collections import deque

from brier.schema import Schema, Enum, Bool

SCHEMA = Schema((
    Enum("move", ("up", "down", "left", "right"), "the single step that gets the player closer to the exit along the shortest safe path"),
    Bool("danger", "is a monster in one of the four cells directly next to the player"),
))

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _bfs(grid, start, goal):
    n = len(grid)
    dist = {goal: 0}
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _render(grid):
    return "\n".join("".join(row) for row in grid)


def generate(n, seed=0, size=5):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        grid = [["." for _ in range(size)] for _ in range(size)]
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        for r, c in cells[: rng.randint(2, 5)]:
            grid[r][c] = "#"
        free = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == "."]
        if len(free) < 4:
            continue
        rng.shuffle(free)
        player, goal, monster = free[0], free[1], free[2]
        grid[goal[0]][goal[1]] = "E"
        grid[monster[0]][monster[1]] = "M"
        grid2 = [row[:] for row in grid]
        grid2[monster[0]][monster[1]] = "#"
        dist = _bfs(grid2, player, goal)
        if player not in dist:
            continue
        best = []
        for name, (dr, dc) in DIRS.items():
            nxt = (player[0] + dr, player[1] + dc)
            if nxt in dist and dist[nxt] == dist[player] - 1:
                best.append(name)
        if not best:
            continue
        danger = any((player[0] + dr, player[1] + dc) == monster for dr, dc in DIRS.values())
        grid[player[0]][player[1]] = "P"
        context = (
            f"A {size}x{size} grid. P = player, E = exit, M = monster (never step onto it), # = wall, . = open floor. "
            f"Rows go top to bottom, columns left to right. The player is at row {player[0] + 1}, column {player[1] + 1}; "
            f"the exit is at row {goal[0] + 1}, column {goal[1] + 1}; the monster is at row {monster[0] + 1}, column {monster[1] + 1}.\n"
            + _render(grid)
        )
        out.append({"context": context, "labels": {"move": tuple(best), "danger": "yes" if danger else "no"}})
    return out


def parse_state(ex):
    """Recover grid, player, exit and monster positions from a generated example's text."""
    grid = [list(r) for r in ex["state"].split("\n")[1:]]
    find = lambda ch: next((r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == ch)
    return grid, find("P"), find("E"), find("M")


def outcome(grid, player, goal, monster, move):
    """Environment signal, no label: 1 if the move is legal and reduces the shortest safe
    distance to the exit, else 0."""
    g = [row[:] for row in grid]
    g[monster[0]][monster[1]] = "#"
    g[player[0]][player[1]] = "."
    dist = _bfs(g, player, goal)
    dr, dc = DIRS[move]
    nxt = (player[0] + dr, player[1] + dc)
    if nxt not in dist:
        return 0.0
    return 1.0 if dist[nxt] < dist.get(player, 10 ** 9) else 0.0


def step(grid, player, monster, move):
    """Apply a move. Returns (new_player, status) with status in {'ok', 'blocked', 'dead', 'exit'}."""
    dr, dc = DIRS[move]
    n = len(grid)
    nxt = (player[0] + dr, player[1] + dc)
    if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or grid[nxt[0]][nxt[1]] == "#":
        return player, "blocked"
    if nxt == monster:
        return nxt, "dead"
    if grid[nxt[0]][nxt[1]] == "E":
        return nxt, "exit"
    return nxt, "ok"


def render_state(grid, player, goal, monster):
    g = [row[:] for row in grid]
    for r in range(len(g)):
        for c in range(len(g[r])):
            if g[r][c] in "PEM":
                g[r][c] = "."
    g[goal[0]][goal[1]] = "E"
    g[monster[0]][monster[1]] = "M"
    g[player[0]][player[1]] = "P"
    size = len(g)
    return (
        f"A {size}x{size} grid. P = player, E = exit, M = monster (never step onto it), # = wall, . = open floor. "
        f"Rows go top to bottom, columns left to right. The player is at row {player[0] + 1}, column {player[1] + 1}; "
        f"the exit is at row {goal[0] + 1}, column {goal[1] + 1}; the monster is at row {monster[0] + 1}, column {monster[1] + 1}.\n"
        + _render(g)
    )


from brier.questions import Choice, Claim  # noqa: E402

QUESTIONS = [
    Choice("move", "the single step that gets the player closer to the exit along the shortest safe path", ("up", "down", "left", "right")),
    Claim("danger", "a monster is in one of the four cells directly next to the player"),
]


def generate_q(n, seed=0, size=5, split="all", uncertain=0.0):
    return [{"state": ex["context"], "questions": QUESTIONS, "labels": ex["labels"]} for ex in generate(n, seed, size)]
