"""One line per tracked pod: minutes left, whether its job runs, DONE markers, last log line."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / ".pod.json").read_text()) if (ROOT / ".pod.json").exists() else {}
names = sys.argv[1:] or sorted(d)
for n in names:
    p = d.get(n)
    if not p:
        print(f"{n:>10}  (no pod)")
        continue
    left = (p["deadline"] - time.time()) / 60
    cmd = "cd /workspace/brier 2>/dev/null && (pgrep -f '^bash jobs' >/dev/null && echo RUN || echo idle); ls results/*/DONE_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\\n' ' '; echo; tail -c 300 job.log 2>/dev/null | tr '\\n' ' ' | tail -c 160"
    try:
        r = subprocess.run([sys.executable, str(ROOT / "pod.py"), "--name", n, "exec", cmd], capture_output=True, text=True, timeout=40)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        state = lines[0] if lines else "?"
        done = lines[1] if len(lines) > 2 else ""
        last = lines[-1][-150:] if len(lines) > 1 else ""
    except Exception as e:
        state, done, last = f"unreachable ({type(e).__name__})", "", ""
    print(f"{n:>10}  {left:5.0f} min  {state:>5}  {done:<28} {last}")
