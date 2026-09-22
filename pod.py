"""RunPod lifecycle with a hard time-to-live. Every pod created here is terminated by a
detached watchdog at its deadline even if this session dies; `down --all` is the panic button."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import runpod

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".pod.json"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REMOTE = "/workspace/brier"
MAX_TTL_MIN = 360   # the hard ceiling a watchdog will always enforce; training runs of several hours are deliberate


def load_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                os.environ["RUNPOD_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing: put it in .env")
    runpod.api_key = key


def _all():
    if not STATE.exists():
        return {}
    d = json.loads(STATE.read_text())
    return {"main": d} if "id" in d else d


def _save_all(d):
    if d:
        STATE.write_text(json.dumps(d))
    elif STATE.exists():
        STATE.unlink()


NAME = "main"


def state():
    return _all().get(NAME)


def set_state(pod):
    """Read-modify-write under a file lock, so several pods can be brought up at once."""
    import fcntl
    with open(ROOT / ".pod.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = _all()
        if pod is None:
            d.pop(NAME, None)
        else:
            d[NAME] = pod
        _save_all(d)


def ssh_endpoint(pod):
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None


def cmd_up(a):
    ttl = min(a.ttl, MAX_TTL_MIN)
    if state():
        sys.exit(f"pod {state()['id']} already recorded under name {NAME!r}; run `pod.py down --name {NAME}` first")
    pubkey = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    pod = runpod.create_pod(
        name=f"brier-{NAME}-{int(time.time())}", image_name=IMAGE, gpu_type_id=a.gpu, cloud_type=a.cloud,
        container_disk_in_gb=40, volume_in_gb=0, ports="22/tcp" + (f",{a.http}/http" if a.http else ""), env={"PUBLIC_KEY": pubkey}, gpu_count=a.gpus,
        min_memory_in_gb=16,
    )
    deadline = time.time() + ttl * 60
    set_state({"id": pod["id"], "created": time.time(), "deadline": deadline, "gpu": a.gpu})
    log = ROOT / ".watchdog.log"
    subprocess.Popen(
        [sys.executable, str(ROOT / "pod.py"), "watchdog", pod["id"], str(deadline)],
        stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print(f"created {pod['id']} ({a.gpus}× {a.gpu}); hard TTL {ttl} min, watchdog armed")
    for _ in range(60):
        time.sleep(5)
        p = runpod.get_pod(pod["id"])
        ep = ssh_endpoint(p)
        if ep:
            print(f"ssh ready: ssh -p {ep[1]} root@{ep[0]}")
            if a.http:
                print(f"http: https://{pod['id']}-{a.http}.proxy.runpod.net")
            return
    print("pod created but ssh not up after 5 min; check `pod.py status`")


def cmd_watchdog(a):
    deadline = float(a.deadline)
    while time.time() < deadline:
        time.sleep(min(60, max(1, deadline - time.time())))
    try:
        runpod.terminate_pod(a.pod_id)
        print(f"{time.ctime()} watchdog terminated {a.pod_id}")
    except Exception as e:
        print(f"{time.ctime()} watchdog: {a.pod_id} already gone ({e})")
    d = _all()
    for k, v in list(d.items()):
        if v["id"] == a.pod_id:
            d.pop(k)
    _save_all(d)


def cmd_extend(a):
    s = state()
    if not s:
        sys.exit("no tracked pod")
    total = (time.time() - s["created"]) / 60 + a.minutes
    if total > MAX_TTL_MIN:
        sys.exit(f"refusing: total lifetime would be {total:.0f} min > cap {MAX_TTL_MIN}")
    subprocess.run(["pkill", "-f", f"pod.py watchdog {s['id']}"])
    deadline = time.time() + a.minutes * 60
    s["deadline"] = deadline
    set_state(s)
    subprocess.Popen([sys.executable, str(ROOT / "pod.py"), "watchdog", s["id"], str(deadline)],
                     stdout=open(ROOT / ".watchdog.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"{s['id']}: new hard deadline in {a.minutes} min (total lifetime {total:.0f} min), watchdog re-armed")


def cmd_status(a):
    pods = runpod.get_pods()
    mine = {v["id"]: k for k, v in _all().items()}
    if not pods:
        print("no pods on account")
    for p in pods:
        up = p.get("runtime") or {}
        mins = up.get("uptimeInSeconds", 0) / 60
        tag = f"(this repo: {mine[p['id']]})" if p["id"] in mine else "(UNTRACKED)"
        print(f"{p['id']} {p['name']} {p.get('desiredStatus')} gpu={p.get('machine', {}).get('gpuDisplayName')} "
              f"up={mins:.0f}min ${p.get('costPerHr', 0):.2f}/hr {tag}")
    for k, v in _all().items():
        left = (v["deadline"] - time.time()) / 60
        print(f"tracked pod {k} {v['id']}: {left:.0f} min until watchdog terminates it")


def cmd_down(a):
    s = state()
    if s and not a.all and not a.force:
        marker = ROOT / f".pulled-{s['id']}"
        if not marker.exists():
            sys.exit(f"refusing: results of {s['id']} were never pulled; run `pod.py --name {NAME} pull` first, or down --force")
    ids = [p["id"] for p in runpod.get_pods() if p["name"].startswith("brier-")] if a.all else ([s["id"]] if s else [])
    if not ids:
        print("nothing to terminate")
    for pid in ids:
        try:
            runpod.terminate_pod(pid)
            print(f"terminated {pid}")
        except Exception as e:
            print(f"{pid}: {e}")
    if a.all:
        _save_all({})
    else:
        set_state(None)
    time.sleep(3)
    left = runpod.get_pods()
    print(f"pods remaining on account: {len(left)}")
    for p in left:
        print(f"  {p.get('desiredStatus')}: {p['id']} {p['name']} ${p.get('costPerHr', 0):.2f}/hr (not created by this repo)")


def _get_pod_retry(pod_id, tries=6):
    for i in range(tries):
        try:
            return runpod.get_pod(pod_id)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def _endpoint_or_die():
    s = state()
    if not s:
        sys.exit("no tracked pod")
    ep = ssh_endpoint(_get_pod_retry(s["id"]))
    if not ep:
        sys.exit("pod has no public ssh port yet")
    return ep


def cmd_ssh(a):
    ip, port = _endpoint_or_die()
    print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{ip}")


def cmd_sync(a):
    ip, port = _endpoint_or_die()
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"root@{ip}"]
    tar = subprocess.Popen(["tar", "-C", str(ROOT), "-czf", "-", "--exclude", ".git", "--exclude", "results",
                            "--exclude", ".env", "--exclude", ".pod.json", "--exclude", "__pycache__",
                            "--exclude", ".watchdog.log", "--exclude", "adapters", "."], stdout=subprocess.PIPE)
    subprocess.run(ssh + [f"mkdir -p {REMOTE} && tar -C {REMOTE} -xzf -"], stdin=tar.stdout, check=True)
    tar.wait()
    print("synced")


def cmd_push(a):
    """Copy one local path (a directory such as adapters/ptr) to the same place on the pod."""
    ip, port = _endpoint_or_die()
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"root@{ip}"]
    for path in a.paths:
        rel = os.path.relpath(path, ROOT)
        tar = subprocess.Popen(["tar", "-C", str(ROOT), "-czf", "-", rel], stdout=subprocess.PIPE)
        subprocess.run(ssh + [f"mkdir -p {REMOTE} && tar -C {REMOTE} -xzf -"], stdin=tar.stdout, check=True)
        tar.wait()
        print(f"pushed {rel}")


def cmd_exec(a):
    ip, port = _endpoint_or_die()
    r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"root@{ip}",
                        f"cd {REMOTE} && {a.command}"])
    sys.exit(r.returncode)


def cmd_pull(a):
    ip, port = _endpoint_or_die()
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), f"root@{ip}"]
    dirs = "results adapters" if a.adapters else "results"
    remote = subprocess.Popen(ssh + [f"cd {REMOTE} && tar -czf - $(ls -d {dirs} 2>/dev/null)"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-C", str(ROOT), "-xzf", "-"], stdin=remote.stdout, check=True)
    remote.wait()
    if remote.returncode:
        sys.exit(f"pull failed: the remote tar exited {remote.returncode}; nothing was replaced")
    remote.wait()
    (ROOT / f".pulled-{state()['id']}").write_text(str(time.time()))
    print(f"pulled {dirs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    u.add_argument("--cloud", default="ALL")
    u.add_argument("--ttl", type=int, default=90, help="minutes until forced termination")
    u.add_argument("--http", type=int, default=None, help="also expose this port over RunPod's HTTPS proxy")
    u.add_argument("--gpus", type=int, default=1, help="GPUs in the pod (data-parallel training via torchrun)")
    w = sub.add_parser("watchdog")
    w.add_argument("pod_id")
    w.add_argument("deadline")
    sub.add_parser("status")
    x = sub.add_parser("extend")
    x.add_argument("minutes", type=int)
    d = sub.add_parser("down")
    d.add_argument("--all", action="store_true")
    d.add_argument("--force", action="store_true", help="terminate even if results were never pulled")
    sub.add_parser("ssh")
    sub.add_parser("sync")
    ps = sub.add_parser("push")
    ps.add_argument("paths", nargs="+")
    e = sub.add_parser("exec")
    e.add_argument("command")
    pl = sub.add_parser("pull")
    pl.add_argument("--adapters", action="store_true", help="also pull adapters/")
    ap.add_argument("--name", default="main", help="which tracked pod this command refers to")
    a = ap.parse_args()
    NAME = a.name
    load_key()
    globals()[f"cmd_{a.cmd}"](a)
