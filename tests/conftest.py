import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbps.encoding import ENC_LAGS, EncodingModel  # noqa: E402


@pytest.fixture
def toy_encoder():
    """Small synthetic encoding model: 40 cosine-tuned units, mild latent noise."""
    rng = np.random.default_rng(0)
    U, L = 40, len(ENC_LAGS)
    B = np.zeros((U, 1 + 3 * L))
    B[:, 0] = np.log(rng.uniform(0.1, 0.4, U))  # counts per 20 ms bin
    pd = rng.uniform(0, 2 * np.pi, U)
    w = np.stack([np.cos(pd), np.sin(pd), 0.3 * np.ones(U)], 1)
    for j in range(L):
        B[:, 1 + 3 * j:4 + 3 * j] = w * 0.4
    C = 0.1 * rng.standard_normal((U, 2))
    return EncodingModel(B, C, np.array([0.5, 0.3]), 0.05 * np.ones(U))


@pytest.fixture
def toy_data(toy_encoder):
    """Smooth random reaches -> spikes, aligned like the real data (X[t] encodes V at the leads)."""
    rng = np.random.default_rng(1)
    N, T = 60, 120
    V = np.zeros((N, T + max(ENC_LAGS), 2))
    for i in range(N):
        a = rng.uniform(0, 2 * np.pi)
        tt = np.arange(V.shape[1])
        V[i] = 400 * np.sin(np.pi * np.clip((tt - 20) / 25, 0, 1))[:, None] * [np.cos(a), np.sin(a)]
    Vs = np.stack([V[:, l:l + T] for l in ENC_LAGS], 2)
    X = toy_encoder.simulate_from_stack(Vs, rng)
    return X, Vs[:, :, ENC_LAGS.index(4)], Vs
