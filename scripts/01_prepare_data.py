"""Download (if needed), verify, bin MC_Maze at 20 ms, and select the decode lag on validation.

Usage: python scripts/01_prepare_data.py
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data  # noqa: E402
from cbps.decoders import KalmanVelocity, Wiener  # noqa: E402
from cbps.encoding import bits_per_spike, features, fit_poisson_glm  # noqa: E402
from cbps.io import save_json  # noqa: E402
from cbps.metrics import r2  # noqa: E402


def main():
    if not data.RAW.exists():
        data.RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MC_Maze (DANDI 000128, ~691 MB) -> {data.RAW}")
        urllib.request.urlretrieve(data.DANDI_URL, data.RAW)
    digest = data.sha256(data.RAW)
    assert digest == data.RAW_SHA256, f"checksum mismatch: {digest}"
    if not data.PROCESSED.exists():
        data.preprocess()
    E = data.EVAL_START
    rows = []
    for lag in range(data.MAX_LAG + 1):
        tr, va, _ = data.split(data.load(lag))
        B = fit_poisson_glm(features(tr.V.reshape(-1, 2)), tr.X.reshape(-1, tr.X.shape[-1]).astype(float))
        F = features(va.V.reshape(-1, 2))
        Y = va.X.reshape(-1, va.X.shape[-1])
        lam = np.exp(B[:, 0] + F @ B[:, 1:].T)
        lam0 = np.broadcast_to(tr.X.reshape(-1, Y.shape[1]).mean(0), Y.shape)
        kf = KalmanVelocity.fit(tr.X, tr.V)
        w = Wiener.fit(tr.X, tr.V, 20, 1e-3)
        rows.append({"lag_bins": lag, "lag_ms": lag * 20,
                     "glm_val_bits_per_spike": bits_per_spike(Y, lam, lam0),
                     "kf_val_r2": r2(va.V[:, E:], kf.predict(va.X)[:, E:]),
                     "wiener20_val_r2": r2(va.V[:, E:], w.predict(va.X)[:, E:])})
        print(rows[-1])
    best = max(rows, key=lambda r: r["glm_val_bits_per_spike"])["lag_bins"]
    tr, va, te = data.split(data.load(best))
    save_json("lag_selection.json", {
        "selection_rule": "argmax validation Poisson-GLM bits/spike (single-lag velocity+speed GLM)",
        "selected_lag_bins": best, "rows": rows,
        "n_trials": {"train": len(tr.X), "val": len(va.X), "test": len(te.X)},
        "n_units": int(tr.X.shape[-1]), "raw_sha256": digest})
    assert best == data.DECODE_LAG, f"selected lag {best} != DECODE_LAG {data.DECODE_LAG}; update cbps/data.py"


if __name__ == "__main__":
    main()
