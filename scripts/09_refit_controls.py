"""ReFIT controls: label quality, label masking, and the simulated user's model, with CIs.

For each user model (adaptive forward-model gain = default; fixed gain = the earlier, unrealistic one):
  1. tune the KF's gain, run it closed loop for 10 x 2 min calibration sessions;
  2. refit five variants on the same data:
       ReFIT-KF            rotated labels, bins after target changes masked (the main-table version)
       ReFIT-KF-unmasked   rotated labels, no masking (the original, buggy version)
       ReFIT-KF-oracle     labels = the user's true intent c_{t-3} (upper bound: perfect relabelling)
       ReFIT-GRU-64        rotated + masked
       ReFIT-GRU-64-oracle true-intent labels
  3. tune each gain and evaluate (20 x 2 min, common seeds), paired differences vs the KF;
  4. open-loop diagnosis on Webgrid command streams: R^2 vs true intent, delivered speed ratio,
     jitter at rest -- to explain *why* a variant wins or loses.

Writes results/refit_controls.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data  # noqa: E402
from cbps.encoding import CMD_LEAD  # noqa: E402
from cbps.experiment import evaluate, pool, tune  # noqa: E402
from cbps.io import load_json, load_pickle, save_json  # noqa: E402
from cbps.metrics import r2  # noqa: E402
from cbps.refit import collect, refit_gru, refit_kf  # noqa: E402
from cbps.webgrid import Oracle, TaskConfig, UserModel, run_session  # noqa: E402

CAL_SEEDS = list(range(1000, 1006))
REFIT_SEEDS = list(range(2000, 2010))
EVAL_SEEDS = list(range(0, 20))


def open_loop(decs, enc, user, n_sessions=4):
    lag = CMD_LEAD - data.DECODE_LAG
    streams = [run_session(Oracle(), enc, TaskConfig(mode="click", seconds=60.0), user, s, record=True)[1]["cmd"]
               for s in range(500, 500 + n_sessions)]
    out = {}
    for n, d in decs.items():
        ys, yh, rest = [], [], []
        for k, cmd in enumerate(streams):
            sm, dd = enc.sampler(np.random.default_rng([k, 11])), d.clone()
            dd.reset()
            v = np.array([dd.step(sm.step(c)) for c in cmd])
            ys.append(cmd[50 - lag:len(cmd) - lag])
            yh.append(v[50:])
        ys, yh = np.concatenate(ys), np.concatenate(yh)
        mv, still = np.linalg.norm(ys, axis=1) > 100, np.linalg.norm(ys, axis=1) < 30
        out[n] = {"r2_vs_intent": r2(ys, yh),
                  "speed_ratio_when_moving": float(np.median(np.linalg.norm(yh[mv], axis=1)
                                                             / np.linalg.norm(ys[mv], axis=1))),
                  "speed_rms_when_intent_still_mm_s": float(np.sqrt((yh[still] ** 2).sum(1).mean()))}
    return out


def main():
    decs = load_pickle("decoders.pkl")
    enc = load_pickle("encoder.pkl")
    um = load_json("user_model.json")
    base_user = UserModel(vmax=um["vmax"], tau=um["tau"], tau_m=um["tau_m"])
    users = {"adaptive": base_user, "fixed_gain": replace(base_user, forward_gain="fixed")}
    _, _, te = data.split(data.load(data.DECODE_LAG))
    E = data.EVAL_START
    click = TaskConfig(mode="click", seconds=120.0)
    out = {"eval_seeds": EVAL_SEEDS, "eval_seconds": 120.0, "users": {}}
    for uk, user in users.items():
        with pool(decs, enc) as p:
            kf_gain = tune(p, ["KF"], replace(click, seconds=60.0), user, CAL_SEEDS)["KF"][0]
        sessions, _ = collect(decs["KF"], enc, replace(click, gain=kf_gain), user, REFIT_SEEDS)
        variants = {"KF": decs["KF"], "GRU-64": decs["GRU-64"]}
        for dec in (refit_kf(decs["KF"], sessions), refit_kf(decs["KF"], sessions, masked=False),
                    refit_kf(decs["KF"], sessions, labels="oracle"),
                    refit_gru(sessions, 64)[0], refit_gru(sessions, 64, labels="oracle")[0]):
            variants[dec.name] = dec
        names = list(variants)
        with pool(variants, enc) as p:
            tuned = tune(p, names, replace(click, seconds=60.0), user, CAL_SEEDS)
            res = evaluate(p, names, click, user, EVAL_SEEDS, {n: tuned[n][0] for n in names})
        diag = open_loop(variants, enc, user)
        for n in names:
            res[n]["real_test_r2"] = r2(te.V[:, E:], variants[n].predict(te.X)[:, E:])
            res[n]["open_loop_webgrid"] = diag[n]
            res[n].pop("per_session_bps")
        out["users"][uk] = {"kf_gain_during_calibration": kf_gain, "decoders": res}
        print(uk)
        for n, v in res.items():
            d = v["paired_diff_vs_KF"]
            print(f"  {n:22s} gain {v['gain']:<5} BPS {v['bps_mean']:.2f} [{v['bps_ci95'][0]:.2f}, {v['bps_ci95'][1]:.2f}]"
                  f"  vs KF {d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  real R2 {v['real_test_r2']:.3f}"
                  f"  OL R2 {v['open_loop_webgrid']['r2_vs_intent']:.2f} speed {v['open_loop_webgrid']['speed_ratio_when_moving']:.2f}"
                  f" still-rms {v['open_loop_webgrid']['speed_rms_when_intent_still_mm_s']:.0f}")
        save_json("refit_controls.json", out)


if __name__ == "__main__":
    main()
