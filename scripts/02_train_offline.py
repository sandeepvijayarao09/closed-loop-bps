"""Fit offline decoders on real MC_Maze train trials; select hyperparameters on val; report test R^2.

Decoders: velocity KF (the ReFIT state), lag-augmented KF, Wiener, GRU-{16..256}, and two ablations
(GRU-64 trained on continuous-session crops; Wiener / GRU-64 trained at decode lag 0).
R^2 is reported on the scored window [-500, +800] ms and on the NLB window [-250, +450] ms.

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
NLB = slice(data.PRE_BINS - 13, data.PRE_BINS + 22)  # [-260, +440) ms: the NLB [-250, +450] ms window at 20 ms
KF_LEADS = (0, 2, 4, 5, 6, 8)


def score(dec, va, te):
    pv, pt = dec.predict(va.X), dec.predict(te.X)
    m, lo, hi = r2_per_trial_bootstrap(te.V[:, E:], pt[:, E:])
    mn, lon, hin = r2_per_trial_bootstrap(te.V[:, NLB], pt[:, NLB])
    return {"val_r2": r2(va.V[:, E:], pv[:, E:]), "test_r2": m, "test_r2_ci95": [lo, hi],
            "test_r2_nlb_window": mn, "test_r2_nlb_window_ci95": [lon, hin], "n_params": int(dec.n_params)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = ap.parse_args()
    torch.set_num_threads(4)
    full = data.load(data.DECODE_LAG)
    tr, va, te = data.split(full)
    decoders, out = {}, {"decode_lag_bins": data.DECODE_LAG, "eval_window_bins": [E, tr.X.shape[1]],
                         "eval_window_ms_from_onset": [(E - data.PRE_BINS) * 20, data.POST_BINS * 20],
                         "nlb_window_ms_from_onset": [-260, 440],
                         "n_trials": {"train": len(tr.X), "val": len(va.X), "test": len(te.X)},
                         "ci": "95% percentile bootstrap over held-out test trials", "decoders": {}}

    def record(dec, extra=None, sets=(va, te)):
        decoders[dec.name] = dec
        out["decoders"][dec.name] = {**score(dec, *sets), **(extra or {})}
        print(dec.name, {k: v for k, v in out["decoders"][dec.name].items() if "sweep" not in k})
        save_pickle("decoders.pkl", decoders)
        save_json("offline.json", out)

    record(KalmanVelocity.fit(tr.X, tr.V), {"state": "velocity (2-D)"})

    # Lag-augmented KF: state = velocity at KF_LEADS bins after the spikes; output = lead DECODE_LAG.
    Vs = data.velocity_stack(KF_LEADS)
    itr = data.split_indices(full)[0]
    kfl = KalmanVelocity.fit_lagged(tr.X, Vs[itr], KF_LEADS, data.DECODE_LAG)
    record(kfl, {"state": f"velocity at leads {list(KF_LEADS)} bins ({2 * len(KF_LEADS)}-D)"})

    sweep = []
    for h in (5, 10, 15, 20, 25, 30, 40):
        alphas = (1e-4, 1e-3, 1e-2)
        for a, w in zip(alphas, Wiener.fit_path(tr.X, tr.V, h, alphas)):
            sweep.append((r2(va.V[:, E:], w.predict(va.X)[:, E:]), h, a))
    _, h, a = max(sweep)
    record(Wiener.fit(tr.X, tr.V, h, a), {"n_hist": h, "alpha": a,
                                          "val_sweep": [{"n_hist": s[1], "alpha": s[2], "val_r2": s[0]} for s in sweep]})

    for hid in args.hidden:
        t0 = time.time()
        g, info = train_gru(tr.X, tr.V, va.X, va.V, hidden=hid, seed=0)
        record(g, {"hidden": hid, "epochs_run": info["epochs_run"], "train_seconds": time.time() - t0})

    # Ablation: GRU-64 trained on random 4 s crops of the continuous session stream
    # (+ channel gain/offset augmentation) instead of onset-aligned windows.
    t0 = time.time()
    g, info = train_gru(tr.X, tr.V, va.X, va.V, hidden=64, seed=0, stream=data.continuous(tr))
    g.name = "GRU-64-cont"
    record(g, {"hidden": 64, "epochs_run": info["epochs_run"], "train_seconds": time.time() - t0,
               "training": "continuous-stream crops + augmentation"})

    # Ablation: decode lag 0 (target = velocity in the same bin as the spikes). Easier offline,
    # but in closed loop the output trails intent by 5 more bins (100 ms). R^2 here is against
    # the lag-0 target, so it is not directly comparable with the lag-5 rows.
    tr0, va0, te0 = data.split(data.load(0))
    w0 = Wiener.fit(tr0.X, tr0.V, h, a)
    w0.name = "Wiener-lag0"
    record(w0, {"decode_lag_bins": 0, "n_hist": h, "alpha": a}, sets=(va0, te0))
    g0, info = train_gru(tr0.X, tr0.V, va0.X, va0.V, hidden=64, seed=0)
    g0.name = "GRU-64-lag0"
    record(g0, {"decode_lag_bins": 0, "hidden": 64, "epochs_run": info["epochs_run"]}, sets=(va0, te0))


if __name__ == "__main__":
    main()
