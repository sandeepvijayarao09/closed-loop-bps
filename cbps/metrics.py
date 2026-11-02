"""Metrics: velocity R^2, Webgrid achieved bitrate, bootstrap CIs."""
from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean over output dims of 1 - SSE/SST (same convention as NLB vel R^2)."""
    y = y.reshape(-1, y.shape[-1])
    yhat = yhat.reshape(-1, yhat.shape[-1])
    sse = ((y - yhat) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - sse / sst))


def achieved_bitrate(n_targets: int, n_correct: int, n_incorrect: int, seconds: float) -> float:
    """Webgrid / BrainGate 'achieved bitrate' (Nuyujukian et al. 2015; Pandarinath et al. 2017).

    B = log2(N - 1) * max(Sc - Si, 0) / t

    N is the number of selectable targets on screen. Wrong selections cancel
    right ones, so a decoder that clicks randomly scores 0, not a positive rate.
    """
    if n_targets < 2 or seconds <= 0:
        raise ValueError("need N >= 2 targets and t > 0")
    return float(np.log2(n_targets - 1) * max(n_correct - n_incorrect, 0) / seconds)


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over independent units (sessions / seeds / trials)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def r2_per_trial_bootstrap(y: np.ndarray, yhat: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """CI on pooled R^2 by resampling whole held-out trials ([N, T, 2] inputs)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [r2(y[i], yhat[i]) for i in rng.integers(0, n, (n_boot, n))]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return r2(y, yhat), float(lo), float(hi)
