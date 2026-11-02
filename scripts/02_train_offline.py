"""Fit offline decoders on real MC_Maze train trials; select hyperparameters on val; report test R^2.

Writes results/offline.json and artifacts/decoders.pkl.
Usage: python scripts/02_train_offline.py [--hidden 16 32 64 128 256]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data  # noqa: E402
from cbps.decoders import KalmanVelocity, Wiener  # noqa: E402
from cbps.io import save_json, save_pickle  # noqa: E402
from cbps.metrics import r2, r2_per_trial_bootstrap  # noqa: E402
from cbps.train_gru import train_gru  # noqa: E402

E = data.EVAL_START


def score(dec, va, te):
    pv, pt = dec.predict(va.X), dec.predict(te.X)
    m, lo, hi = r2_per_trial_bootstrap(te.V[:, E:], pt[:, E:])
    return {"val_r2": r2(va.V[:, E:], pv[:, E:]), "test_r2": m, "test_r2_ci95": [lo, hi],
            "n_params": int(dec.n_params)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = ap.parse_args()
    torch.set_num_threads(4)
    tr, va, te = data.split(data.load(data.DECODE_LAG))
    decoders, out = {}, {"decode_lag_bins": data.DECODE_LAG, "eval_window_bins": [E, tr.X.shape[1]],
                         "eval_window_ms_from_onset": [(E - data.PRE_BINS) * 20, data.POST_BINS * 20],
                         "n_trials": {"train": len(tr.X), "val": len(va.X), "test": len(te.X)},
                         "ci": "95% percentile bootstrap over held-out test trials", "decoders": {}}

    kf = KalmanVelocity.fit(tr.X, tr.V)
    decoders["KF"] = kf
    out["decoders"]["KF"] = score(kf, va, te)
    print("KF", out["decoders"]["KF"])

    grid = [(h, a) for h in (5, 10, 15, 20, 25, 30, 40) for a in (1e-4, 1e-3, 1e-2)]
    sweep = []
    for h, a in grid:
        w = Wiener.fit(tr.X, tr.V, h, a)
        sweep.append((r2(va.V[:, E:], w.predict(va.X)[:, E:]), h, a))
    _, h, a = max(sweep)
    w = Wiener.fit(tr.X, tr.V, h, a)
    decoders["Wiener"] = w
    out["decoders"]["Wiener"] = {**score(w, va, te), "n_hist": h, "alpha": a,
                                 "val_sweep": [{"n_hist": s[1], "alpha": s[2], "val_r2": s[0]} for s in sweep]}
    print("Wiener", {k: v for k, v in out["decoders"]["Wiener"].items() if k != "val_sweep"})

    for hid in args.hidden:
        t0 = time.time()
        g, info = train_gru(tr.X, tr.V, va.X, va.V, hidden=hid, seed=0)
        decoders[g.name] = g
        out["decoders"][g.name] = {**score(g, va, te), "hidden": hid, "epochs_run": info["epochs_run"],
                                   "train_seconds": time.time() - t0}
        print(g.name, out["decoders"][g.name])
        save_pickle("decoders.pkl", decoders)
        save_json("offline.json", out)
    # Ablation: same GRU-64, trained on random 4 s crops of the continuous session
    # stream (+ channel gain/offset augmentation) instead of onset-aligned windows.
    t0 = time.time()
    g, info = train_gru(tr.X, tr.V, va.X, va.V, hidden=64, seed=0, stream=data.continuous(tr))
    g.name = "GRU-64-cont"
    decoders[g.name] = g
    out["decoders"][g.name] = {**score(g, va, te), "hidden": 64, "epochs_run": info["epochs_run"],
                               "train_seconds": time.time() - t0, "training": "continuous-stream crops + augmentation"}
    print(g.name, out["decoders"][g.name])
    save_pickle("decoders.pkl", decoders)
    save_json("offline.json", out)


if __name__ == "__main__":
    main()
