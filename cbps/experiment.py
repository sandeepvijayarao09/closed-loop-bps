"""Shared closed-loop experiment machinery: parallel sessions, gain tuning, summaries with CIs."""
from __future__ import annotations

import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .metrics import bootstrap_ci
from .webgrid import Oracle, run_session

GAINS = [0.09, 0.13, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
_G = {}


def _init(decs, enc):
    os.environ["OMP_NUM_THREADS"] = "1"
    _G["decs"], _G["enc"] = decs, enc


def _job(args):
    key, name, cfg, user, seed = args
    dec = Oracle() if name == "Oracle" else _G["decs"][name].clone()
    res, _ = run_session(dec, _G["enc"], cfg, user, seed)
    return key, name, cfg.gain, seed, res.as_dict()


def pool(decs, enc, procs: int | None = None):
    return Pool(procs or max(1, (os.cpu_count() or 2) - 2), initializer=_init, initargs=(decs, enc))


def run_jobs(p, jobs):
    """jobs: (key, decoder name, TaskConfig, UserModel, seed) -> list of (key, name, gain, seed, result dict)."""
    return p.map(_job, jobs, chunksize=1)


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


def tune(p, names, cal_cfg, user, seeds, gains=GAINS, key="tune"):
    """Best output gain per decoder by mean calibration BPS. Ties (e.g. all zero) -> gain closest to 1."""
    jobs = [(key, n, replace(cal_cfg, gain=g), user, s) for n in names for g in gains for s in seeds]
    rows = run_jobs(p, jobs)
    best = {}
    for n in names:
        means = {g: float(np.mean([r[4]["bps"] for r in rows if r[1] == n and r[2] == g])) for g in gains}
        best[n] = (max(gains, key=lambda g: (round(means[g], 6), -abs(np.log(g)))), means)
    return best


def evaluate(p, names, cfg, user, seeds, gains: dict, key="eval"):
    """Sessions at each decoder's gain -> {name: summary}, with paired difference vs the first name."""
    jobs = [(key, n, replace(cfg, gain=gains[n]), user, s) for n in names for s in seeds]
    rows = run_jobs(p, jobs)
    out = {n: {"gain": gains[n], **summarize([r[4] for r in rows if r[1] == n])} for n in names}
    ref = out[names[0]]["per_session_bps"]
    for n in names:
        out[n]["paired_diff_vs_" + names[0]] = paired_diff(out[n]["per_session_bps"], ref)
    return out
