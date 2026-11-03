"""Load MC_Maze (DANDI 000128) and bin it at 20 ms.

We read the NWB file with h5py directly (no pynwb object graph) and cache:
  spikes  [B, U] uint8   spike counts per 20 ms bin, whole session
  vel     [B, 2] float32 hand velocity (mm/s), mean over the bin
  pos     [B, 2] float32 hand position (mm), mean over the bin
  valid   [B]    bool    bin has kinematic samples
  trial table: move-onset bin, start/stop bin, first target position, #barriers

Splits are chronological (first 70% of trials train, next 15% val, last 15%
test). This is stricter than NLB's interleaved val split: it keeps
within-session nonstationarity from leaking between train and test.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from . import BIN_S

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mc_maze_train.nwb"
PROCESSED = ROOT / "data" / "processed" / "mc_maze_20ms.npz"
DANDI_URL = "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
RAW_SHA256 = "a27f84f9c31f077011062fd99a6ced199fdf19e656151f056c3c6244a408a3c4"

# Trial window around movement onset, in bins. Decoders see 1.2 s before onset
# (target/delay period: no movement, preparatory activity) and 800 ms after.
# R^2 is scored on [-500, +800] ms (bins EVAL_START..end); the first 700 ms are
# burn-in so recurrent/filtered decoders are not scored on their transients.
# Scoring the still period matters: a decoder that drifts while the user holds
# still is unusable in closed loop, and the NLB [-250, +450] ms window hides it.
PRE_BINS = 60
EVAL_START = 35
POST_BINS = 40
SPLIT_FRACS = (0.70, 0.15, 0.15)
MAX_LAG = 8  # trial set is fixed by the largest lag we search, so splits never depend on lag


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _bin_mean(ts: np.ndarray, x: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.floor(ts / BIN_S + 1e-9).astype(np.int64)
    keep = (idx >= 0) & (idx < n_bins)
    idx, x = idx[keep], x[keep]
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    out = np.stack([np.bincount(idx, weights=x[:, d], minlength=n_bins) for d in range(x.shape[1])], 1)
    with np.errstate(invalid="ignore"):
        out = out / cnt[:, None]
    return out.astype(np.float32), cnt > 0


def preprocess(raw: Path = RAW, out: Path = PROCESSED) -> Path:
    with h5py.File(raw, "r") as f:
        tr = f["intervals/trials"]
        start = tr["start_time"][:]
        stop = tr["stop_time"][:]
        onset = tr["move_onset_time"][:]
        n_barriers = tr["num_barriers"][:]
        tp = tr["target_pos"][:]
        tp_idx = tr["target_pos_index"][:]
        active = tr["active_target"][:]
        # target_pos is a ragged column; pick the active (reached) target.
        lo = np.concatenate([[0], tp_idx[:-1]]).astype(np.int64)
        target = np.stack([tp[lo[i] + active[i]] for i in range(len(lo))]).astype(np.float32)

        session_end = float(stop.max()) + 1.0
        n_bins = int(np.ceil(session_end / BIN_S))

        st = f["units/spike_times"][:]
        st_idx = f["units/spike_times_index"][:]
        n_units = len(st_idx)
        spikes = np.zeros((n_bins, n_units), dtype=np.uint8)
        lo = 0
        for u, hi in enumerate(st_idx):
            b = np.floor(st[lo:hi] / BIN_S).astype(np.int64)
            b = b[(b >= 0) & (b < n_bins)]
            c = np.bincount(b, minlength=n_bins)
            spikes[:, u] = np.minimum(c, 255)
            lo = hi

        ts = f["processing/behavior/hand_vel/timestamps"][:]
        vel, valid = _bin_mean(ts, f["processing/behavior/hand_vel/data"][:], n_bins)
        ts = f["processing/behavior/hand_pos/timestamps"][:]
        pos, _ = _bin_mean(ts, f["processing/behavior/hand_pos/data"][:], n_bins)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, spikes=spikes, vel=vel, pos=pos, valid=valid,
        onset_bin=np.round(onset / BIN_S).astype(np.int64),
        start_bin=np.round(start / BIN_S).astype(np.int64),
        stop_bin=np.round(stop / BIN_S).astype(np.int64),
        target=target, n_barriers=n_barriers,
    )
    return out


@dataclass
class Trials:
    """Trial-aligned windows. X[t] are spikes in bin t; V[t] is velocity `lag` bins later."""
    X: np.ndarray  # [N, T, U] float32 counts
    V: np.ndarray  # [N, T, 2] mm/s
    P: np.ndarray  # [N, T, 2] mm (hand position, aligned with V)
    target: np.ndarray  # [N, 2]
    n_barriers: np.ndarray  # [N]
    trial_idx: np.ndarray  # [N] chronological index in session

    def subset(self, m: np.ndarray) -> "Trials":
        return Trials(self.X[m], self.V[m], self.P[m], self.target[m], self.n_barriers[m], self.trial_idx[m])

    def flat(self, burn_in: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return (self.X[:, burn_in:].reshape(-1, self.X.shape[-1]),
                self.V[:, burn_in:].reshape(-1, 2))


def load(lag: int, path: Path = PROCESSED) -> Trials:
    assert 0 <= lag <= MAX_LAG
    d = np.load(path)
    spikes, vel, pos, valid = d["spikes"], d["vel"], d["pos"], d["valid"]
    on = d["onset_bin"]
    T = PRE_BINS + POST_BINS
    xs, vs, ps, keep = [], [], [], []
    for i, o in enumerate(on):
        s = o - PRE_BINS
        kin = slice(s + lag, s + lag + T)
        if s < 0 or s + MAX_LAG + T > len(vel) or not valid[s:s + MAX_LAG + T].all():
            continue
        xs.append(spikes[s:s + T])
        vs.append(vel[kin])
        ps.append(pos[kin])
        keep.append(i)
    keep = np.array(keep)
    return Trials(np.stack(xs).astype(np.float32), np.stack(vs), np.stack(ps),
                  d["target"][keep], d["n_barriers"][keep], keep)


def split_indices(tr: Trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices of the chronological train / val / test trials."""
    n = len(tr.trial_idx)
    order = np.argsort(tr.trial_idx)
    a = int(round(SPLIT_FRACS[0] * n))
    b = int(round((SPLIT_FRACS[0] + SPLIT_FRACS[1]) * n))
    return np.sort(order[:a]), np.sort(order[a:b]), np.sort(order[b:])


def split(tr: Trials) -> tuple[Trials, Trials, Trials]:
    return tuple(tr.subset(i) for i in split_indices(tr))


# Decoders predict velocity DECODE_LAG bins after the spikes (100 ms). Chosen on
# the validation split as the lag maximising held-out Poisson-GLM likelihood
# (it is also the KF's best lag); see results/lag_selection.json.
DECODE_LAG = 5


def velocity_stack(lags, path: Path = PROCESSED) -> np.ndarray:
    """[N, T, len(lags), 2] velocities at several leads; same trial set/order as load()."""
    return np.stack([load(l, path).V for l in lags], axis=2)


def continuous(tr: Trials, lag: int = DECODE_LAG, path: Path = PROCESSED):
    """The raw session stream spanning a split's trials (first start -> last stop).

    Returns X [B, U] counts, V [B, 2] velocity `lag` bins later, valid [B].
    Training on random crops of this stream (instead of onset-aligned windows)
    removes the hold -> go -> reach timing prior that a recurrent decoder can
    otherwise learn from the trial structure.
    """
    d = np.load(path)
    i0, i1 = tr.trial_idx.min(), tr.trial_idx.max()
    s, e = int(d["start_bin"][i0]), int(d["stop_bin"][i1])
    X = d["spikes"][s:e].astype(np.float32)
    V = d["vel"][s + lag:e + lag]
    valid = d["valid"][s + lag:e + lag] & np.isfinite(V).all(1)
    return X, np.nan_to_num(V), valid
