"""Closed-loop Webgrid evaluation of every decoder in the simulator.

Protocol:
  1. Tune the offline KF's output gain on calibration seeds (click mode).
  2. ReFIT: run that KF closed loop for REFIT_SEEDS sessions, relabel intention,
     fit ReFIT-KF and ReFIT-GRU-64 on the same data. Score both on real test trials.
  3. Per selection mode: tune each decoder's output gain from GAINS on CAL_SEEDS
     (disjoint from evaluation seeds), like a clinician setting speed. Evaluate
     on EVAL_SEEDS with common random numbers across decoders. Untuned (gain=1)
     results are kept too.
  4. Drift sweep (click mode): 10-minute sessions with per-neuron log-rate drift.
  5. Probe: mean decoder response to a held 500 mm/s command (explains step 3).

Writes results/closed_loop.json, results/probe_sustained.json, artifacts/decoders_cl.pkl.
Usage: python scripts/04_closed_loop.py [--quick]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data  # noqa: E402
from cbps.io import load_json, load_pickle, save_json, save_pickle  # noqa: E402
from cbps.metrics import bootstrap_ci, r2, r2_per_trial_bootstrap  # noqa: E402
from cbps.refit import collect, refit_gru, refit_kf  # noqa: E402
from cbps.webgrid import Oracle, TaskConfig, UserModel, run_session  # noqa: E402

GAINS = [0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
CAL_SEEDS = list(range(1000, 1006))
REFIT_SEEDS = list(range(2000, 2010))
EVAL_SEEDS = list(range(0, 40))
DRIFT_LEVELS = [0.0, 0.25, 0.5, 1.0]
DRIFT_SEEDS = list(range(100, 110))

_G = {}


def _init(decs, enc):
    _G["decs"], _G["enc"] = decs, enc


def _job(args):
    name, cfg, user, seed = args
    dec = Oracle() if name == "Oracle" else _G["decs"][name].clone()
    res, _ = run_session(dec, _G["enc"], cfg, user, seed)
    return name, cfg.gain, cfg.drift_per_10min, seed, res.as_dict()


def summarize(rows):
    b = [r["bps"] for r in rows]
    m, lo, hi = bootstrap_ci(b)
    t = [r["mean_time_to_target_s"] for r in rows if np.isfinite(r["mean_time_to_target_s"])]
    return {"bps_mean": m, "bps_ci95": [lo, hi], "n_sessions": len(b),
            "correct_per_min": float(np.mean([r["correct"] / r["seconds"] * 60 for r in rows])),
            "incorrect_per_min": float(np.mean([r["incorrect"] / r["seconds"] * 60 for r in rows])),
            "time_to_target_s": float(np.mean(t)) if t else None, "per_session_bps": b}


def paired_diff(a, b):
    m, lo, hi = bootstrap_ci(np.array(a) - np.array(b))
    return {"mean": m, "ci95": [lo, hi]}


def tune(pool, names, cal_cfg, user):
    jobs = [(n, replace(cal_cfg, gain=g), user, s) for n in names for g in GAINS for s in CAL_SEEDS]
    rows = pool.map(_job, jobs, chunksize=1)
    best = {}
    for n in names:
        means = {g: float(np.mean([r[4]["bps"] for r in rows if r[0] == n and r[1] == g])) for g in GAINS}
        # ties (e.g. all zero) -> gain closest to 1
        best[n] = (max(GAINS, key=lambda g: (round(means[g], 6), -abs(np.log(g)))), means)
    return best


def probe(decs, enc, n_rep=20):
    """Hold still for 1 s, then hold a 500 mm/s rightward command for 2 s, then stop."""
    on, off = 50, 150
    cmd = np.zeros((200, 2))
    cmd[on:off] = [500.0, 0.0]
    out = {"command_mm_s": cmd[:, 0].tolist(), "onset_bin": on, "decoders": {}}
    for n, d in decs.items():
        if n == "Oracle":
            continue
        reps = []
        for s in range(n_rep):
            sm, dd = enc.sampler(np.random.default_rng([s, 99])), d.clone()
            dd.reset()
            reps.append([dd.step(sm.step(c)) for c in cmd])
        reps = np.array(reps)  # [rep, T, 2]
        m = reps[..., 0].mean(0)
        out["decoders"][n] = {
            "mean_vx": m.tolist(),
            "mean_vx_0_to_200ms": float(m[on:on + 10].mean()),
            "mean_vx_1_to_2s": float(m[on + 50:off].mean()),
            # jitter: within-run fluctuation the cursor actually shows (not across-run spread)
            "rest_speed_rms_mm_s": float(np.sqrt((reps[:, 25:on] ** 2).sum(-1).mean())),
            "moving_vy_std_mm_s": float(reps[:, on + 50:off, 1].std(1).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few seeds/short sessions, for smoke tests")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    global EVAL_SEEDS, CAL_SEEDS, DRIFT_SEEDS, REFIT_SEEDS
    eval_s, cal_s, drift_s = 120.0, 60.0, 600.0
    if args.quick:
        EVAL_SEEDS, CAL_SEEDS, DRIFT_SEEDS, REFIT_SEEDS = EVAL_SEEDS[:4], CAL_SEEDS[:2], DRIFT_SEEDS[:2], REFIT_SEEDS[:3]
        eval_s, cal_s, drift_s = 30.0, 20.0, 60.0

    decs = load_pickle("decoders.pkl")
    enc = load_pickle("encoder.pkl")
    um = load_json("user_model.json")
    user = UserModel(vmax=um["vmax"], tau=um["tau"], tau_m=um["tau_m"])
    _, va, te = data.split(data.load(data.DECODE_LAG))
    E = data.EVAL_START
    t0 = time.time()
    out = {"user_model": asdict(user), "gains": GAINS, "cal_seeds": CAL_SEEDS, "eval_seeds": EVAL_SEEDS,
           "refit_seeds": REFIT_SEEDS, "eval_seconds": eval_s, "modes": {}}

    # 1-2. ReFIT calibration (click mode), shared by both modes.
    click = TaskConfig(mode="click", seconds=eval_s)
    with Pool(args.procs, initializer=_init, initargs=(decs, enc)) as pool:
        kf_gain = tune(pool, ["KF"], replace(click, seconds=cal_s), user)["KF"][0]
    sessions, cal_stats = collect(decs["KF"], enc, replace(click, seconds=120.0, gain=kf_gain), user, REFIT_SEEDS)
    rk = refit_kf(decs["KF"], sessions)
    rg, rg_info = refit_gru(sessions, hidden=64)
    decs.update({"ReFIT-KF": rk, rg.name: rg})
    refit_info = {"kf_gain_during_calibration": kf_gain, "calibration_minutes": len(REFIT_SEEDS) * 2,
                  "calibration_sessions": cal_stats, "refit_gru_epochs": rg_info["epochs_run"],
                  "refit_gru_best_label_r2": rg_info["best_val_r2"], "real_data_r2": {}}
    for n, d in (("ReFIT-KF", rk), (rg.name, rg)):
        m, lo, hi = r2_per_trial_bootstrap(te.V[:, E:], d.predict(te.X)[:, E:])
        refit_info["real_data_r2"][n] = {"val_r2": r2(va.V[:, E:], d.predict(va.X)[:, E:]), "test_r2": m,
                                         "test_r2_ci95": [lo, hi], "n_params": int(d.n_params)}
    out["refit"] = refit_info
    save_pickle("decoders_cl.pkl", decs)
    print(f"[{time.time() - t0:.0f}s] ReFIT done: {refit_info['real_data_r2']}")

    names = list(decs) + ["Oracle"]
    with Pool(args.procs, initializer=_init, initargs=(decs, enc)) as pool:
        for mode in ["click", "dwell"]:
            # Dwell uses a 6x6 grid (40 mm targets) and 1 s dwell: on the 12x12 / 0.5 s
            # grid, cursor wander (15-22 mm per 0.5 s) is as large as a target and every
            # real decoder scores ~0 (scripts/08_diagnostics.py sweeps target size x dwell).
            base = (TaskConfig(mode=mode, seconds=eval_s) if mode == "click"
                    else TaskConfig(mode=mode, grid=6, dwell_s=1.0, seconds=eval_s))
            tuned = tune(pool, names, replace(base, seconds=cal_s), user)
            jobs = [(n, replace(base, gain=g), user, s) for n in names for g in sorted({tuned[n][0], 1.0})
                    for s in EVAL_SEEDS]
            rows = pool.map(_job, jobs, chunksize=1)
            res = {"task": asdict(base), "decoders": {}}
            for n in names:
                g = tuned[n][0]
                res["decoders"][n] = {"gain": g, "cal_bps_by_gain": tuned[n][1],
                                      **summarize([r[4] for r in rows if r[0] == n and r[1] == g]),
                                      "untuned_gain1": summarize([r[4] for r in rows if r[0] == n and r[1] == 1.0])}
            kf_b = res["decoders"]["KF"]["per_session_bps"]
            for n in names:
                res["decoders"][n]["paired_diff_vs_KF"] = paired_diff(res["decoders"][n]["per_session_bps"], kf_b)
            if mode == "click":
                jobs = [(n, replace(base, seconds=drift_s, gain=tuned[n][0], drift_per_10min=dl), user, s)
                        for n in names for dl in DRIFT_LEVELS for s in DRIFT_SEEDS]
                rows = pool.map(_job, jobs, chunksize=1)
                res["drift"] = {"session_minutes": drift_s / 60, "levels_log_rate_std_per_10min": DRIFT_LEVELS,
                                "decoders": {n: {str(dl): summarize([r[4] for r in rows if r[0] == n and r[2] == dl])
                                                 for dl in DRIFT_LEVELS} for n in names}}
            out["modes"][mode] = res
            print(f"[{time.time() - t0:.0f}s] {mode}:")
            for n, v in res["decoders"].items():
                print(f"  {n:14s} gain {v['gain']:.1f}  BPS {v['bps_mean']:.2f} [{v['bps_ci95'][0]:.2f}, "
                      f"{v['bps_ci95'][1]:.2f}]  untuned {v['untuned_gain1']['bps_mean']:.2f}")
    save_json("closed_loop_quick.json" if args.quick else "closed_loop.json", out)
    if not args.quick:
        save_json("probe_sustained.json", probe(decs, enc))


if __name__ == "__main__":
    main()
