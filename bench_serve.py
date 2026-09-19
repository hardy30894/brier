"""Throughput of a running server: C concurrent clients each send N /ask requests; reports
requests per second and latency percentiles as the clients see them."""
import argparse
import json
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--clients", type=int, default=8)
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--questions", type=int, default=3)
a = ap.parse_args()

state = "Webhooks are firing twice for every event since around 3am UTC. I need this sorted today if at all possible. We're on the Team plan."
base = [{"id": "category", "kind": "pick", "instructions": "which team should handle this ticket", "options": ["billing", "bug", "feature", "account", "other"]},
        {"id": "needs_human", "kind": "claim", "instructions": "the customer is asking for a human agent"},
        {"id": "refund", "kind": "claim", "instructions": "the customer asks for money back"}]
qs = [dict(base[i % 3], id=f"{base[i % 3]['id']}_{i}") for i in range(a.questions)]
body = json.dumps({"state": state, "questions": qs}).encode()

def call():
    req = urllib.request.Request(a.url + "/ask", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

call()
lat = []
lock = threading.Lock()
def client():
    for _ in range(a.n):
        t0 = time.perf_counter(); call(); dt = (time.perf_counter() - t0) * 1000
        with lock:
            lat.append(dt)
ths = [threading.Thread(target=client) for _ in range(a.clients)]
t0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
wall = time.perf_counter() - t0
lat.sort()
p = lambda f: lat[min(len(lat) - 1, int(f * len(lat)))]
print(f"clients={a.clients} requests={len(lat)} questions={a.questions}  {len(lat) / wall:.1f} req/s  p50 {p(0.5):.1f} ms  p95 {p(0.95):.1f} ms  max {lat[-1]:.1f} ms")
