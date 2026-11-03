"""Replication on NLB MC_RTT: continuous random-target reaching (DANDI 000129, 50 MB).

MC_RTT has no hold/go/reach trial structure: targets change about once a second and the monkey moves
continuously, much like Webgrid. If the MC_Maze GRUs fail in closed loop because they learned a
"brief isolated reach" prior (distribution shift), GRUs trained on MC_RTT should not.

Everything is refit on MC_RTT: decoders, encoding model, and the user's reach policy (MC_RTT reaches
are slower and the workspace smaller, so the MC_Maze user would push the encoding model far outside
its fitted range). The Webgrid is scaled to the MC_RTT workspace (140 mm, 8x8 grid), so absolute BPS
is not comparable with the MC_Maze simulator; the decoder ranking is the point.

Writes results/mc_rtt.json, artifacts/decoders_rtt.pkl, artifacts/encoder_rtt.pkl.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys  # noqa: E402
import urllib.request  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data_rtt as R  # noqa: E402
from cbps.data import DECODE_LAG, EVAL_START  # noqa: E402
from cbps.decoders import KalmanVelocity, Wiener  # noqa: E402
from cbps.encoding import ENC_LAGS, bits_per_spike, fit_encoding_model  # noqa: E402
from cbps.experiment import evaluate, pool, tune  # noqa: E402
from cbps.io import save_json, save_pickle  # noqa: E402
from cbps.metrics import r2, r2_per_trial_bootstrap  # noqa: E402
from cbps.train_gru import train_gru  # noqa: E402
from cbps.webgrid import TaskConfig, UserModel, fit_reach_policy  # noqa: E402

LAGS = (0, 2, 4, 5, 6, 8)  # union of ENC_LAGS and the KF-lag leads
KF_LEADS = (0, 2, 4, 5, 6, 8)
CAL_SEEDS = list(range(1000, 1006))
EVAL_SEEDS = list(range(0, 20))
E = EVAL_START


def fit_user(spikes, vel, pos, tgt, valid, stop):
    """Reach policy fit to MC_RTT target-to-target reaches in the training span, aligned at movement onset."""
    sp = np.linalg.norm(vel, axis=1)
    finite = np.isfinite(tgt).all(1)
    change = np.flatnonzero(np.any(np.diff(np.nan_to_num(tgt[:stop]), axis=0) != 0, axis=1)) + 1
    change = change[finite[change]]  # a few target samples are NaN; skip reaches toward them
    starts, goals, profs = [], [], []
    for c in change:
        seg = sp[c:c + 30]
        on = c + int(np.argmax(seg > 50)) if (seg > 50).any() else None
        if on is None or on + 40 > stop or not valid[on:on + 40].all():
            continue
        starts.append(pos[on])
        goals.append(tgt[c])
        profs.append(sp[on:on + 40])
    vmax, tau, tm, rmse, sim = fit_reach_policy(np.array(starts), np.array(goals), np.mean(profs, 0))
    return {"vmax": vmax, "tau": tau, "tau_m": tm, "train_rmse_mm_s": rmse, "n_reaches": len(starts),
            "real_profile": np.mean(profs, 0), "sim_profile": sim}


def main():
    torch.set_num_threads(4)
    if not R.RAW.exists():
        R.RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(R.DANDI_URL, R.RAW)
    digest = R.verify()
    spikes, vel, pos, tgt, valid = R.load_continuous()
    (a0, b0), (a1, b1), (a2, b2) = R.split_bounds(len(spikes))
    Xtr, Vtr = R.windows(spikes, vel, valid, LAGS, a0, b0)
    Xva, Vva = R.windows(spikes, vel, valid, LAGS, a1, b1)
    Xte, Vte = R.windows(spikes, vel, valid, LAGS, a2, b2)
    k = LAGS.index(DECODE_LAG)
    tr_v, va_v, te_v = Vtr[:, :, k], Vva[:, :, k], Vte[:, :, k]
    out = {"raw_sha256": digest, "n_units": int(spikes.shape[1]), "minutes": len(spikes) * 0.02 / 60,
           "n_windows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}, "window_bins": R.WIN,
           "decode_lag_bins": DECODE_LAG, "decoders": {}}

    # lag sweep (Wiener, val): same trade-off as MC_Maze, reported, not used for selection
    out["wiener_val_r2_by_lag"] = {
        str(l): r2(Vva[:, E:, i], Wiener.fit(Xtr, Vtr[:, :, i], 20, 1e-3).predict(Xva)[:, E:]) for i, l in enumerate(LAGS)}

    decs = {"KF": KalmanVelocity.fit(Xtr, tr_v),
            "KF-lag": KalmanVelocity.fit_lagged(Xtr, Vtr[:, :, [LAGS.index(l) for l in KF_LEADS]], KF_LEADS,
                                                DECODE_LAG)}
    sweep = []
    for h in (10, 20, 30):
        for a, w in zip((1e-3, 1e-2, 1e-1), Wiener.fit_path(Xtr, tr_v, h, (1e-3, 1e-2, 1e-1))):
            sweep.append((r2(va_v[:, E:], w.predict(Xva)[:, E:]), h, a))
    _, h, a = max(sweep)
    decs["Wiener"] = Wiener.fit(Xtr, tr_v, h, a)
    for hid in (16, 64):
        decs[f"GRU-{hid}"] = train_gru(Xtr, tr_v, Xva, va_v, hidden=hid, seed=0)[0]

    # encoding model + fidelity
    enc_idx = [LAGS.index(l) for l in ENC_LAGS]
    enc = fit_encoding_model(Xtr, Vtr[:, :, enc_idx])
    save_pickle("encoder_rtt.pkl", enc)
    Y = Xte.reshape(-1, Xte.shape[-1])
    lam = np.exp(enc.log_rate(Vte[:, :, enc_idx])).reshape(Y.shape)
    out["encoding_model"] = {"test_bits_per_spike": bits_per_spike(Y, lam, np.broadcast_to(Xtr.reshape(-1, Y.shape[1]).mean(0), Y.shape)),
                             **enc.info}
    Xs = enc.simulate_from_stack(Vte[:, :, enc_idx], np.random.default_rng(123))
    for n, d in decs.items():
        m, lo, hi = r2_per_trial_bootstrap(te_v[:, E:], d.predict(Xte)[:, E:])
        out["decoders"][n] = {"test_r2": m, "test_r2_ci95": [lo, hi],
                              "sim_test_r2": r2(te_v[:, E:], d.predict(Xs)[:, E:]), "n_params": int(d.n_params)}
        print(n, out["decoders"][n])
    save_pickle("decoders_rtt.pkl", decs)

    u = fit_user(spikes, vel, pos, tgt, valid, b0)
    # The click threshold scales with the user's speed (80 mm/s for the 1000 mm/s MC_Maze user).
    user = UserModel(vmax=u["vmax"], tau=u["tau"], tau_m=u["tau_m"], click_speed=80.0 * u["vmax"] / 1000.0)
    out["user_model"] = {**{k: v for k, v in u.items()}, **asdict(user)}
    print("user", {k: v for k, v in u.items() if "profile" not in k})

    click = TaskConfig(mode="click", width_mm=140.0, grid=8, seconds=120.0)
    out["task"] = asdict(click)
    names = list(decs) + ["Oracle"]
    with pool(decs, enc) as p:
        tuned = tune(p, names, replace(click, seconds=60.0), user, CAL_SEEDS)
        res = evaluate(p, names, click, user, EVAL_SEEDS, {n: tuned[n][0] for n in names})
    for n in names:
        res[n].pop("per_session_bps")
        out["decoders"].setdefault(n, {})["closed_loop_click"] = {**res[n], "cal_bps_by_gain": tuned[n][1]}
        print(f"  {n:8s} gain {res[n]['gain']:<5} BPS {res[n]['bps_mean']:.2f} [{res[n]['bps_ci95'][0]:.2f}, "
              f"{res[n]['bps_ci95'][1]:.2f}]  vs KF {res[n]['paired_diff_vs_KF']['mean']:+.2f}")
    save_json("mc_rtt.json", out)


if __name__ == "__main__":
    main()
