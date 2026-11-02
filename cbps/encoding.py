"""Generative encoding model fit to real MC_Maze neurons, used by the simulator.

Real M1 units are tuned to hand velocity over a range of leads (~0-160 ms),
and history-based decoders exploit that. A single-lag model throws it away:
with one lag, a Wiener filter trained on real data scored R^2 0.30 on
simulated spikes vs 0.64 on real ones, even without extra noise. So each
neuron gets a multi-lag velocity kernel:

  log lambda_i(t) = b_i + sum_l w_il . f(v_{t+l}) + c_i . z_t + e_i(t) + d_i(t),  l in ENC_LAGS
  y_i(t) ~ Poisson(lambda_i(t))

  f(v)  = [vx, vy, |v|] / V_SCALE
  z_t   shared latent noise, AR(1) per factor (factor analysis on GLM residuals)
  e_i   private log-normal overdispersion (iid per bin)
  d_i   slow per-neuron drift (log-rate random walk; zero unless a scenario adds it)

In the simulator the user issues a command c_t, defined as the velocity the
hand would have max(ENC_LAGS) bins later (c_t == v_{t+8}). The neural bin at t
then depends on c_t, c_{t-2}, ..., c_{t-8}: one command unfolds in the
population over 160 ms. Decoders are trained to output v_{t+DECODE_LAG}, which
in the simulator is c_{t-3}, so there is a built-in 60 ms neural-to-decoder
latency on top of visual feedback delay.

Not modelled (see README): preparatory/rotational dynamics, position tuning,
nonlinear speed-direction interactions, the user learning the decoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import FactorAnalysis

V_SCALE = 300.0  # mm/s; keeps GLM features O(1)
ENC_LAGS = (0, 2, 4, 6, 8)  # bins of velocity lead encoded by each neural bin
CMD_LEAD = max(ENC_LAGS)
LBAR_FLOOR = 0.05  # counts/bin (2.5 Hz), used only when converting Pearson loadings to log-rate
MAX_LOG_STD = 1.0


def features(V: np.ndarray) -> np.ndarray:
    """V [..., 2] -> [..., 3]."""
    V = np.asarray(V, dtype=np.float64)
    sp = np.linalg.norm(V, axis=-1, keepdims=True)
    return np.concatenate([V, sp], -1) / V_SCALE


def stacked_features(Vs: np.ndarray) -> np.ndarray:
    """Vs [..., L, 2] velocities at ENC_LAGS -> [..., 3L]."""
    f = features(Vs)
    return f.reshape(f.shape[:-2] + (-1,))


def fit_poisson_glm(F: np.ndarray, Y: np.ndarray, l2: float = 1e-2, iters: int = 30):
    """Vectorised Newton/IRLS for U independent Poisson GLMs sharing a design matrix.

    F [n, k] features (intercept added here), Y [n, U] counts. Returns B [U, k+1].
    """
    n, k = F.shape
    Xd = np.concatenate([np.ones((n, 1)), F], 1)
    U = Y.shape[1]
    B = np.zeros((U, k + 1))
    B[:, 0] = np.log(Y.mean(0) + 1e-3)
    reg = l2 * np.eye(k + 1)
    reg[0, 0] = 0
    outer = (Xd[:, :, None] * Xd[:, None, :]).reshape(n, -1)  # [n, (k+1)^2]
    for _ in range(iters):
        lam = np.exp(np.clip(Xd @ B.T, -20, 5))  # [n, U]
        g = Xd.T @ (Y - lam) - (reg @ B.T)  # [k+1, U]
        Hs = (outer.T @ lam).T.reshape(U, k + 1, k + 1) + reg
        step = np.linalg.solve(Hs, g.T[..., None])[..., 0]
        B += step
        if np.abs(step).max() < 1e-6:
            break
    return B


def poisson_ll(Y, lam):
    from scipy.special import gammaln
    return float((Y * np.log(lam + 1e-12) - lam - gammaln(Y + 1)).sum())


def bits_per_spike(Y, lam, lam0):
    return (poisson_ll(Y, lam) - poisson_ll(Y, lam0)) / Y.sum() / np.log(2)


@dataclass
class EncodingModel:
    B: np.ndarray            # [U, 1 + 3L] GLM weights
    C: np.ndarray            # [U, k] latent loadings in log-rate space
    phi: np.ndarray          # [k] AR(1) coefficients of the latents
    sigma_priv: np.ndarray   # [U] private log-rate noise std
    info: dict = field(default_factory=dict)

    @property
    def n_units(self):
        return self.B.shape[0]

    @property
    def log_var(self):
        return (self.C ** 2).sum(1) + self.sigma_priv ** 2

    def log_rate(self, Vs: np.ndarray) -> np.ndarray:
        """Vs [..., L, 2] velocities at ENC_LAGS -> GLM log-rate [..., U]."""
        return self.B[:, 0] + stacked_features(Vs) @ self.B[:, 1:].T

    def sampler(self, rng: np.random.Generator, noise_scale: float = 1.0, drift_per_10min: float = 0.0,
                bin_s: float = 0.02) -> "SpikeSampler":
        return SpikeSampler(self, rng, noise_scale, drift_per_10min, bin_s)

    def simulate_from_stack(self, Vs: np.ndarray, rng: np.random.Generator, noise_scale: float = 1.0):
        """Spikes for real velocity stacks Vs [N, T, L, 2] (used to validate the model)."""
        N, T = Vs.shape[:2]
        base = self.log_rate(Vs)  # [N, T, U]
        out = np.zeros_like(base, dtype=np.float32)
        for i in range(N):
            s = self.sampler(rng, noise_scale)
            for t in range(T):
                out[i, t] = s.sample(base[i, t])
        return out


class SpikeSampler:
    """Streaming spike generator: command history, latent noise and drift state."""

    def __init__(self, enc: EncodingModel, rng, noise_scale, drift_per_10min, bin_s):
        self.enc, self.rng, self.ns = enc, rng, noise_scale
        k = len(enc.phi)
        self.z = rng.standard_normal(k)  # start at stationarity
        self.innov = np.sqrt(1 - enc.phi ** 2)
        # Random walk in log-rate: std after 10 min equals drift_per_10min.
        self.drift_step = drift_per_10min / np.sqrt(600.0 / bin_s)
        self.d = np.zeros(enc.n_units)
        self.cmd = np.zeros((CMD_LEAD + 1, 2))  # cmd[j] = c_{t-j}
        self._lag_rows = np.array([CMD_LEAD - l for l in ENC_LAGS])  # v_{t+l} == c_{t-(8-l)}

    def sample(self, base_log_rate: np.ndarray) -> np.ndarray:
        e = self.enc
        self.z = e.phi * self.z + self.innov * self.rng.standard_normal(len(e.phi))
        if self.drift_step > 0:
            self.d += self.drift_step * self.rng.standard_normal(e.n_units)
        # -0.5*var keeps E[rate] at the GLM rate for any noise scale.
        lr = (base_log_rate + self.ns * (e.C @ self.z + e.sigma_priv * self.rng.standard_normal(e.n_units))
              - 0.5 * self.ns ** 2 * e.log_var + self.d)
        return self.rng.poisson(np.exp(np.minimum(lr, 4.0))).astype(np.float32)

    def step(self, command: np.ndarray) -> np.ndarray:
        self.cmd = np.roll(self.cmd, 1, axis=0)
        self.cmd[0] = command
        return self.sample(self.enc.log_rate(self.cmd[self._lag_rows]))


def fit_encoding_model(X: np.ndarray, Vs: np.ndarray, n_factors: int = 8, seed: int = 0) -> EncodingModel:
    """X [N,T,U] counts; Vs [N,T,L,2] velocity at each ENC_LAGS lead of bin t."""
    N, T, U = X.shape
    F = stacked_features(Vs).reshape(N * T, -1)
    Y = X.reshape(-1, U).astype(np.float64)
    B = fit_poisson_glm(F, Y)
    lam = np.exp(B[:, 0] + F @ B[:, 1:].T)
    resid = (Y - lam) / np.sqrt(lam + 1e-6)  # Pearson residuals
    fa = FactorAnalysis(n_components=n_factors, random_state=seed).fit(resid)
    Cp = fa.components_.T  # [U, k] loadings on unit-variance factors (Pearson scale)
    psi = fa.noise_variance_  # private Pearson variance; pure Poisson gives 1
    # First-order delta method: Pearson residual ~ sqrt(lambda) * (log-rate perturbation).
    # It is unreliable for very sparse units (a 0.2 Hz unit gets a huge loading),
    # so the rate used in the conversion is floored and each unit's total
    # log-rate noise std is capped. Both are crude; scripts/03_fit_encoder.py
    # reports how simulated decoding compares with real held-out decoding.
    lbar = np.maximum(lam.mean(0), LBAR_FLOOR)
    C = Cp / np.sqrt(lbar)[:, None]
    sigma_priv = np.sqrt(np.maximum(psi - 1.0, 0.0) / lbar)
    tot = np.sqrt((C ** 2).sum(1) + sigma_priv ** 2)
    shrink = np.minimum(1.0, MAX_LOG_STD / np.maximum(tot, 1e-9))
    C, sigma_priv = C * shrink[:, None], sigma_priv * shrink
    zs = fa.transform(resid).reshape(N, T, n_factors)
    a, b = zs[:, :-1].reshape(-1, n_factors), zs[:, 1:].reshape(-1, n_factors)
    phi = np.clip((a * b).sum(0) / (a * a).sum(0), 0.0, 0.99)
    info = {"mean_rate_hz": float(Y.mean() / 0.02), "phi": phi.round(3).tolist(),
            "median_sigma_priv": float(np.median(sigma_priv)), "n_units_noise_capped": int((shrink < 1).sum())}
    return EncodingModel(B, C, phi, sigma_priv, info)
