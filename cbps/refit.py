"""ReFIT recalibration (Gilja et al. 2012) inside the simulator, for the KF and the GRU.

1. Run the offline-trained KF closed loop for a calibration block.
2. Relabel each bin's intended velocity: keep the decoded speed, rotate it to
   point from the cursor at the target; set it to zero while the cursor is on
   the target (the user is trying to hold still).
3. ReFIT-KF: refit the observation model (H, Q) on (spikes, labels), keeping
   the offline dynamics (A, W).
   ReFIT-GRU: train a GRU on the same (spikes, labels) stream. Same data, same
   labels, so the comparison with ReFIT-KF is like for like.

Honest caveat: the calibration spikes come from the encoding model, not from
the monkey. Both ReFIT decoders are therefore partly fit to the simulator, and
their real-data R^2 (reported alongside) tells you how far that drifted from
the real neurons.
"""
from __future__ import annotations

import numpy as np

from .decoders import KalmanVelocity
from .train_gru import train_gru
from .webgrid import TaskConfig, UserModel, cell_of, run_session


def refit_labels(p, target, vhat, cfg: TaskConfig):
    e = target - p
    d = np.linalg.norm(e, axis=1, keepdims=True)
    lab = np.linalg.norm(vhat, axis=1, keepdims=True) * e / np.maximum(d, 1e-9)
    on = np.array([cell_of(pi, cfg) == cell_of(ti, cfg) for pi, ti in zip(p, target)])
    lab[on] = 0.0
    return lab


def collect(decoder, enc, cfg: TaskConfig, user: UserModel, seeds):
    """Closed-loop calibration block -> list of (spikes [T,U], labels [T,2]) per session, session stats."""
    sessions, stats = [], []
    for s in seeds:
        res, log = run_session(decoder.clone(), enc, cfg, user, seed=s, record=True)
        stats.append(res.as_dict())
        # The cursor moved by gain * decoder output; label in decoder-output units.
        sessions.append((np.asarray(log["y"], dtype=np.float32),
                         refit_labels(log["p"], log["target"], log["vhat"] / cfg.gain, cfg)))
    return sessions, stats


def refit_kf(kf: KalmanVelocity, sessions) -> KalmanVelocity:
    Y = np.concatenate([s[0] for s in sessions])[None]
    L = np.concatenate([s[1] for s in sessions])[None]
    new = KalmanVelocity.fit(Y, L, AW=(kf.A, kf.W))
    new.name = "ReFIT-KF"
    return new


def refit_gru(sessions, hidden: int, seed: int = 0):
    """Train on all but the last calibration session; early-stop on the last one (labels only, no oracle)."""
    tr = sessions[:-1]
    Y = np.concatenate([s[0] for s in tr])
    L = np.concatenate([s[1] for s in tr])
    yv, lv = sessions[-1]
    dec, info = train_gru(None, None, yv[None], lv[None], hidden=hidden, seed=seed,
                          stream=(Y, L, np.ones(len(Y), bool)), crops_per_epoch=800, patience=15)
    dec.name = f"ReFIT-GRU-{hidden}"
    return dec, info
