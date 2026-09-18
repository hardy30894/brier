import math

import numpy as np


def ece(confs, corrects, bins=10):
    confs, corrects = np.asarray(confs, float), np.asarray(corrects, float)
    edges = np.linspace(0, 1, bins + 1)
    total, n = 0.0, len(confs)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.any():
            gap = abs(confs[m].mean() - corrects[m].mean())
            total += m.sum() / n * gap
            rows.append((lo, hi, int(m.sum()), float(confs[m].mean()), float(corrects[m].mean())))
    return float(total), rows


def brier(confs, corrects):
    confs, corrects = np.asarray(confs, float), np.asarray(corrects, float)
    return float(((confs - corrects) ** 2).mean())


def softmax_t(logits, T):
    a = np.asarray(logits, float) / T
    a -= a.max()
    e = np.exp(a)
    return e / e.sum()


def fit_temperature(records, grid=None):
    grid = grid if grid is not None else np.exp(np.linspace(math.log(0.05), math.log(10), 120))
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = 0.0
        for logits, label_idx in records:
            p = softmax_t(logits, T)
            nll -= math.log(max(p[label_idx], 1e-12))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def summarize(confs, corrects, latencies, type_errors=0):
    lat = np.asarray(latencies, float)
    e, rows = ece(confs, corrects)
    return {
        "n": len(confs),
        "accuracy": float(np.mean(corrects)),
        "ece": e,
        "brier": brier(confs, corrects),
        "mean_conf": float(np.mean(confs)),
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p95_ms": float(np.percentile(lat, 95)),
        "type_errors": type_errors,
        "reliability": rows,
    }
