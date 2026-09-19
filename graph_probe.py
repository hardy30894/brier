"""Which way of removing launch overhead is worth its setup cost: Inductor per static shape,
Inductor with dynamic shapes, or plain CUDA-graph capture per shape (no compiler)."""
import argparse
import time

import torch

from brier.__main__ import load_scorer
from brier.pointer import decoder

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter", required=True)
a = ap.parse_args()
sc = load_scorer(a.model, a.adapter)
body = decoder(sc.model)
pad = sc.tok.pad_token_id
shapes = [(1, 128), (1, 256), (4, 256), (8, 256)]
ref = {}

def timeit(fn, k=9):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(k):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return sorted(ts)[k // 2]

def inputs(B, L):
    g = torch.Generator(device="cpu").manual_seed(B * 1000 + L)
    return torch.randint(0, 1000, (B, L), generator=g).to(sc.device)

print(f"{'variant':>16} {'shape':>8} {'setup s':>8} {'p50 ms':>7} {'max|diff|':>9}")
with torch.no_grad():
    for B, L in shapes:
        x = inputs(B, L)
        ref[(B, L)] = body(input_ids=x).last_hidden_state.float()
        print(f"{'eager':>16} {f'{B}x{L}':>8} {0:>8.1f} {timeit(lambda: body(input_ids=x)):>7.1f} {0:>9.1e}")

    class Graphed:
        def __init__(self, body):
            self.body, self.g = body, {}
        def __call__(self, input_ids):
            key = tuple(input_ids.shape)
            if key not in self.g:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                sx = input_ids.clone()
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.body(input_ids=sx)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.body(input_ids=sx).last_hidden_state
                self.g[key] = (graph, sx, out)
            graph, sx, out = self.g[key]
            sx.copy_(input_ids); graph.replay()
            return out
    gb = Graphed(body)
    for B, L in shapes:
        x = inputs(B, L)
        t0 = time.perf_counter(); out = gb(x); torch.cuda.synchronize(); setup = time.perf_counter() - t0
        d = (out.float() - ref[(B, L)]).abs().max().item()
        print(f"{'cuda-graph':>16} {f'{B}x{L}':>8} {setup:>8.1f} {timeit(lambda: gb(x)):>7.1f} {d:>9.1e}")

    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    cd = torch.compile(body, mode="reduce-overhead", dynamic=True)
    for B, L in shapes:
        x = inputs(B, L)
        try:
            t0 = time.perf_counter(); out = cd(input_ids=x).last_hidden_state; torch.cuda.synchronize(); setup = time.perf_counter() - t0
            d = (out.float() - ref[(B, L)]).abs().max().item()
            print(f"{'compile-dynamic':>16} {f'{B}x{L}':>8} {setup:>8.1f} {timeit(lambda: cd(input_ids=x)):>7.1f} {d:>9.1e}")
        except Exception as e:
            print(f"{'compile-dynamic':>16} {f'{B}x{L}':>8} failed: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
            break

    torch._dynamo.reset()
    cs = torch.compile(body, mode="reduce-overhead", dynamic=False)
    for B, L in shapes[:2]:
        x = inputs(B, L)
        t0 = time.perf_counter(); out = cs(input_ids=x).last_hidden_state; torch.cuda.synchronize(); setup = time.perf_counter() - t0
        d = (out.float() - ref[(B, L)]).abs().max().item()
        print(f"{'compile-static':>16} {f'{B}x{L}':>8} {setup:>8.1f} {timeit(lambda: cs(input_ids=x)):>7.1f} {d:>9.1e}")
