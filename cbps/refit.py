"""ReFIT recalibration (Gilja et al. 2012) inside the simulator, for the KF and the GRU.

1. Run the offline-trained KF closed loop for a calibration block.
2. Relabel each bin's intended velocity: keep the decoded speed, rotate it to
   point from the cursor at the target; set it to zero while the cursor is on
   the target (the user is trying to hold still).
3. ReFIT-KF: refit the observation model (H, Q) on (spikes, labels), keeping
   the offline dynamics (A, W).
   ReFIT-GRU: train a GRU on the same (spikes, labels) stream. Same data, same
   labels, so the comparison with ReFIT-KF is like for like.

Labels in the bins right after each target change are masked (switch_mask):
there the label already points at the new target while the spikes still encode
the reach to the old one.

Honest caveat: the calibration spikes come from the encoding model, not from
the monkey. Both ReFIT decoders are therefore partly fit to the simulator, and
their real-data R^2 (reported alongside) tells you how far that drifted from
the real neurons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import BIN_S
from .data import DECODE_LAG
from .decoders import KalmanVelocity
from .encoding import CMD_LEAD
from .train_gru import train_gru
from .webgrid import TaskConfig, UserModel, cell_of, run_session


def refit_labels(p, target, vhat, cfg: TaskConfig):
    e = target - p
    d = np.linalg.norm(e, axis=1, keepdims=True)
    lab = np.linalg.norm(vhat, axis=1, keepdims=True) * e / np.maximum(d, 1e-9)
    on = np.array([cell_of(pi, cfg) == cell_of(ti, cfg) for pi, ti in zip(p, target)])
    lab[on] = 0.0
    return lab


@dataclass
class CalibrationSession:
    y: np.ndarray        # [T, U] spikes
    rotated: np.ndarray  # [T, 2] ReFIT labels (decoded speed, pointed at the target)
    oracle: np.ndarray   # [T, 2] the user's true intent c_{t-3}: what a perfect decoder would output
    mask: np.ndarray     # [T] False for bins right after a target change


def switch_mask(target: np.ndarray, n_after: int) -> np.ndarray:
    """False for the n_after bins following every target change.

    The ReFIT label points at the new target from the bin it appears, but the
    user only reacts after react_delay, and that reaction reaches the decoded
    output after the neural lag and motor smoothing. Those bins' spikes still
    encode the old movement, so their labels are wrong by construction.
    """
    change = np.r_[False, np.any(np.diff(target, axis=0) != 0, axis=1)]
    m = np.ones(len(target), bool)
    for t in np.flatnonzero(change):
        m[t:t + n_after] = False
    return m


def mask_bins(user: UserModel) -> int:
    return user.react_delay + (CMD_LEAD - DECODE_LAG) + int(round(user.tau_m / BIN_S))


def collect(decoder, enc, cfg: TaskConfig, user: UserModel, seeds):
    """Closed-loop calibration block -> list of CalibrationSession, session stats."""
    sessions, stats = [], []
    lag = CMD_LEAD - DECODE_LAG
    for s in seeds:
        res, log = run_session(decoder.clone(), enc, cfg, user, seed=s, record=True)
        stats.append(res.as_dict())
        cmd = np.asarray(log["cmd"])
        oracle = np.r_[np.zeros((lag, 2)), cmd[:-lag]]
        # The cursor moved by gain * decoder output; label in decoder-output units.
        sessions.append(CalibrationSession(np.asarray(log["y"], dtype=np.float32),
                                           refit_labels(log["p"], log["target"], log["vhat"] / cfg.gain, cfg),
                                           oracle, switch_mask(log["target"], mask_bins(user))))
    return sessions, stats


def _stack(sessions, labels: str, masked: bool):
    Y = np.concatenate([s.y for s in sessions])
    L = np.concatenate([getattr(s, labels) for s in sessions])
    M = np.concatenate([s.mask for s in sessions]) if masked else np.ones(len(Y), bool)
    return Y, L, M


def refit_kf(kf: KalmanVelocity, sessions, labels: str = "rotated", masked: bool = True) -> KalmanVelocity:
    Y, L, M = _stack(sessions, labels, masked)
    new = KalmanVelocity.fit(Y[None], L[None], AW=(kf.A, kf.W), mask=M[None])
    new.name = "ReFIT-KF" + ("" if labels == "rotated" else "-oracle") + ("" if masked else "-unmasked")
    return new


def refit_gru(sessions, hidden: int, seed: int = 0, labels: str = "rotated", masked: bool = True):
    """Train on all but the last calibration session (loss masked like the KF fit); early-stop on the
    last one, kept as one continuous sequence. Uses the chosen labels throughout, never the oracle's
    unless labels="oracle"."""
    Y, L, M = _stack(sessions[:-1], labels, masked)
    yv, lv, _ = _stack(sessions[-1:], labels, masked)
    dec, info = train_gru(None, None, yv[None], lv[None], hidden=hidden, seed=seed,
                          stream=(Y, L, M), crops_per_epoch=800, patience=15)
    dec.name = f"ReFIT-GRU-{hidden}" + ("" if labels == "rotated" else "-oracle") + ("" if masked else "-unmasked")
    return dec, info
