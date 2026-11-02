"""Streaming velocity decoders with one interface.

Every decoder consumes one 20 ms bin of raw spike counts per call to `step`
and returns a 2-D velocity (mm/s). `predict` just loops `step` over trials, so
offline R^2 is computed with exactly the causal code path the closed-loop
simulator and the C++ port use. Nothing peeks at future bins.
"""
from __future__ import annotations

import copy

import numpy as np


class Decoder:
    name = "decoder"

    def reset(self) -> None: ...

    def step(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape[:-1] + (2,), dtype=np.float64)
        for i in range(X.shape[0]):
            self.reset()
            for t in range(X.shape[1]):
                out[i, t] = self.step(X[i, t])
        return out

    def clone(self) -> "Decoder":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------
# Velocity Kalman filter (Wu et al. 2006; velocity-only state as in ReFIT).
# --------------------------------------------------------------------------
class KalmanVelocity(Decoder):
    """x_t = A x_{t-1} + w,  y_t - mu = H x_t + q.  Runs at steady-state gain.

    Steady state: x_t = M1 x_{t-1} + M2 (y_t - mu) with M1 = (I - K H) A,
    M2 = K. That is a 2x2 and a 2xU mat-vec per bin, which is what runs in the
    C++ loop. Kalman gains converge within ~1 s of bins, so the steady-state
    and time-varying filters agree after the burn-in we already exclude.
    """
    name = "KF"

    def __init__(self, A, W, H, Q, mu):
        self.A, self.W, self.H, self.Q, self.mu = A, W, H, Q, mu
        self.K = steady_state_gain(A, W, H, Q)
        self.M1 = (np.eye(2) - self.K @ H) @ A
        self.M2 = self.K
        self.x = np.zeros(2)

    @classmethod
    def fit(cls, X: np.ndarray, V: np.ndarray, AW: tuple | None = None, ridge: float = 1e-3):
        """X [N,T,U] counts, V [N,T,2] velocity. AW optionally fixes the dynamics."""
        Y = X.reshape(-1, X.shape[-1]).astype(np.float64)
        Vf = V.reshape(-1, 2).astype(np.float64)
        mu = Y.mean(0)
        Yc = Y - mu
        if AW is None:
            x0 = V[:, :-1].reshape(-1, 2)
            x1 = V[:, 1:].reshape(-1, 2)
            A = np.linalg.lstsq(x0, x1, rcond=None)[0].T
            W = np.cov((x1 - x0 @ A.T).T)
        else:
            A, W = AW
        H = np.linalg.lstsq(Vf, Yc, rcond=None)[0].T  # [U,2]
        R = Yc - Vf @ H.T
        Q = np.cov(R.T) + ridge * np.eye(Y.shape[1])
        return cls(A, W, H, Q, mu)

    def reset(self):
        self.x = np.zeros(2)

    def step(self, y):
        self.x = self.M1 @ self.x + self.M2 @ (y - self.mu)
        return self.x

    def predict(self, X):  # vectorized over trials, same recursion
        Yc = X - self.mu
        out = np.zeros(X.shape[:-1] + (2,))
        x = np.zeros((X.shape[0], 2))
        for t in range(X.shape[1]):
            x = x @ self.M1.T + Yc[:, t] @ self.M2.T
            out[:, t] = x
        return out

    @property
    def n_params(self):
        return self.M1.size + self.M2.size + self.mu.size


def steady_state_gain(A, W, H, Q, iters: int = 2000, tol: float = 1e-10):
    P = W.copy()
    Qi = np.linalg.inv(Q)
    K = None
    for _ in range(iters):
        Pp = A @ P @ A.T + W
        # Information form: avoids inverting the U x U innovation covariance each step.
        Pn = np.linalg.inv(np.linalg.inv(Pp) + H.T @ Qi @ H)
        Kn = Pn @ H.T @ Qi
        if K is not None and np.abs(Kn - K).max() < tol:
            K = Kn
            break
        K, P = Kn, Pn
    return K


# --------------------------------------------------------------------------
# Wiener filter / optimal linear estimator with spike history.
# --------------------------------------------------------------------------
class Wiener(Decoder):
    name = "Wiener"

    def __init__(self, Wt, b, mu, n_hist):
        self.Wt, self.b, self.mu, self.n_hist = Wt, b, mu, n_hist
        self.buf = np.zeros((n_hist, len(mu)))

    @staticmethod
    def design(X, mu, n_hist):
        Xc = X - mu
        N, T, U = X.shape
        out = np.zeros((N, T, n_hist * U))
        for k in range(n_hist):
            out[:, k:, k * U:(k + 1) * U] = Xc[:, :T - k]
        return out

    @classmethod
    def fit(cls, X, V, n_hist=5, alpha=1.0):
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        vm = V[:, n_hist:].reshape(-1, 2).mean(0)
        # Accumulate normal equations in chunks of trials: the full design matrix
        # is ~n_trials*T x (n_hist*U) and would not fit in memory at long histories.
        k = n_hist * X.shape[-1]
        G, b, n = np.zeros((k, k)), np.zeros((k, 2)), 0
        for i in range(0, len(X), 100):
            D = cls.design(X[i:i + 100], mu, n_hist)[:, n_hist:].reshape(-1, k)
            G += D.T @ D
            b += D.T @ (V[i:i + 100, n_hist:].reshape(-1, 2) - vm)
            n += len(D)
        Wt = np.linalg.solve(G + alpha * n * np.eye(k), b)
        return cls(Wt, vm, mu, n_hist)

    def reset(self):
        self.buf[:] = 0

    def step(self, y):
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = y - self.mu
        return self.buf.reshape(-1) @ self.Wt + self.b

    def predict(self, X):
        return self.design(X, self.mu, self.n_hist) @ self.Wt + self.b

    @property
    def n_params(self):
        return self.Wt.size + self.b.size + self.mu.size


# --------------------------------------------------------------------------
# GRU: weights trained in torch (see cbps/train_gru.py), inference in numpy.
# --------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUDecoder(Decoder):
    """Single-layer GRU, PyTorch gate layout (r, z, n), linear readout.

    h_t = GRU(h_{t-1}, (y_t - mu) / sd);  v_t = (W_o h_t + b_o) * v_sd
    """

    def __init__(self, W_ih, W_hh, b_ih, b_hh, W_o, b_o, mu, sd, v_sd):
        self.W_ih, self.W_hh, self.b_ih, self.b_hh = W_ih, W_hh, b_ih, b_hh
        self.W_o, self.b_o, self.mu, self.sd, self.v_sd = W_o, b_o, mu, sd, v_sd
        self.hidden = W_hh.shape[1]
        self.name = f"GRU-{self.hidden}"
        self.h = np.zeros(self.hidden)

    def reset(self):
        self.h = np.zeros(self.hidden)

    def _cell(self, x, h):
        H = self.hidden
        gi = x @ self.W_ih.T + self.b_ih
        gh = h @ self.W_hh.T + self.b_hh
        r = _sigmoid(gi[..., :H] + gh[..., :H])
        z = _sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
        return (1 - z) * n + z * h

    def step(self, y):
        self.h = self._cell((y - self.mu) / self.sd, self.h)
        return (self.W_o @ self.h + self.b_o) * self.v_sd

    def predict(self, X):
        Xn = (X - self.mu) / self.sd
        h = np.zeros((X.shape[0], self.hidden))
        out = np.zeros(X.shape[:-1] + (2,))
        for t in range(X.shape[1]):
            h = self._cell(Xn[:, t], h)
            out[:, t] = (h @ self.W_o.T + self.b_o) * self.v_sd
        return out

    @property
    def n_params(self):
        return sum(a.size for a in (self.W_ih, self.W_hh, self.b_ih, self.b_hh, self.W_o, self.b_o, self.mu, self.sd))
