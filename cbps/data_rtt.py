"""NLB MC_RTT (DANDI 000129): continuous random-target reaching, Indy, 130 units, 649 s.

Unlike MC_Maze there is no hold/go/reach trial structure: a new target appears roughly every
second and the monkey moves continuously, which is much closer to Webgrid. Data are binned at
20 ms, split chronologically 70/15/15 by time, and cut into non-overlapping 2 s windows so the
same decoder code (which takes [N, T, U] arrays) runs unchanged.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from . import BIN_S
from .data import ROOT, SPLIT_FRACS, sha256

RAW = ROOT / "data" / "raw_rtt" / "mc_rtt_train.nwb"
DANDI_URL = "https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/"
RAW_SHA256 = "2f78db62bd4d68b9bc737444f72bc2dfe475d7390dd7a54848aaf6a6a6ba8da5"
WIN = 100  # bins per window (2 s)


def load_continuous(path: Path = RAW):
    """spikes [B, U], vel [B, 2] mm/s, pos [B, 2] mm (cursor), target [B, 2] mm, valid [B]."""
    with h5py.File(path, "r") as f:
        b = f["processing/behavior"]

        def series(k):
            d = b[k + "/data"]
            x = d[:] * d.attrs.get("conversion", 1.0) * 1000.0  # NWB stores m or m/s; work in mm
            t0, rate = b[k + "/starting_time"][()], b[k + "/starting_time"].attrs["rate"]
            return x, t0, rate

        vel, t0, rate = series("finger_vel")
        pos, _, _ = series("cursor_pos")
        tgt, _, _ = series("target_pos")
        k = int(round(BIN_S * rate))
        n_bins = len(vel) // k
        shape = (n_bins, k, -1)
        vel_b = vel[:n_bins * k].reshape(shape)
        valid = ~np.isnan(vel_b).any((1, 2))
        vel_b = np.nan_to_num(vel_b).mean(1)
        pos_b = np.nan_to_num(pos[:n_bins * k].reshape(shape)).mean(1)
        tgt_b = tgt[:n_bins * k].reshape(shape)[:, -1]
        st, st_idx = f["units/spike_times"][:], f["units/spike_times_index"][:]
        spikes = np.zeros((n_bins, len(st_idx)), np.float32)
        lo = 0
        for u, hi in enumerate(st_idx):
            bi = np.floor((st[lo:hi] - t0) / BIN_S).astype(int)
            bi = bi[(bi >= 0) & (bi < n_bins)]
            spikes[:, u] = np.bincount(bi, minlength=n_bins)
            lo = hi
    return spikes, vel_b.astype(np.float32), pos_b.astype(np.float32), tgt_b.astype(np.float32), valid


def windows(spikes, vel, valid, lags, start, stop):
    """Non-overlapping WIN-bin windows in [start, stop): X [N,T,U], V [N,T,L,2] (velocity at each lag)."""
    xs, vs = [], []
    mx = max(lags)
    for s in range(start, stop - WIN - mx, WIN):
        if not valid[s:s + WIN + mx].all():
            continue
        xs.append(spikes[s:s + WIN])
        vs.append(np.stack([vel[s + l:s + l + WIN] for l in lags], 1))
    return np.stack(xs), np.stack(vs)


def split_bounds(n_bins):
    a = int(SPLIT_FRACS[0] * n_bins)
    b = int((SPLIT_FRACS[0] + SPLIT_FRACS[1]) * n_bins)
    return (0, a), (a, b), (b, n_bins)


def verify(path: Path = RAW) -> str:
    digest = sha256(path)
    assert digest == RAW_SHA256, f"checksum mismatch: {digest}"
    return digest
