"""Export numpy decoders to the C++ binary format read by rt/decoders.hpp.

File layout (little-endian):
  b"CBPS" | u32 version=1 | u32 kind (1=KF, 2=GRU, 3=Wiener) | u32 n_units | u32 param
  then float32 arrays, row-major:
    KF:     mu[U], M1[d,d], M2[d,U], out[2] (state indices of vx, vy)   param = d
    Wiener: mu[U], W[2, n_hist*U] (block k = k bins ago), b[2]          param = n_hist
    GRU:    mu[U], 1/sd[U], W_ih[3H,U], W_hh[3H,H], b_ih[3H], b_hh[3H],
            W_o[2,H], b_o[2], v_sd[2]                                     param = H
Reference files: u32 T | u32 U | float32 X[T,U] | float32 Y[T,2] (Python outputs).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .decoders import GRUDecoder, KalmanVelocity, Wiener


def _f32(*arrays):
    return b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes() for a in arrays)


def export(dec, path: Path) -> None:
    U = len(dec.mu)
    if isinstance(dec, KalmanVelocity):
        head, body = (1, U, dec.d), _f32(dec.mu, dec.M1, dec.M2, dec.out.astype(float))
    elif isinstance(dec, Wiener):
        head, body = (3, U, dec.n_hist), _f32(dec.mu, dec.Wt.T, dec.b)
    elif isinstance(dec, GRUDecoder):
        head = (2, U, dec.hidden)
        body = _f32(dec.mu, 1.0 / dec.sd, dec.W_ih, dec.W_hh, dec.b_ih, dec.b_hh, dec.W_o, dec.b_o, dec.v_sd)
    else:
        raise TypeError(type(dec))
    path.write_bytes(b"CBPS" + struct.pack("<4I", 1, *head) + body)


def export_reference(dec, X: np.ndarray, path: Path) -> None:
    """X [T, U] one continuous stream; Y from the Python decoder without resets."""
    dec = dec.clone()
    dec.reset()
    Y = np.stack([dec.step(x) for x in X])
    path.write_bytes(struct.pack("<2I", *X.shape) + _f32(X, Y))
