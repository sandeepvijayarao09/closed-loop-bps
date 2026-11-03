"""Diagnostics for the two results that look like bugs: dwell BPS ~0, and GRUs scoring worst in closed loop.

A. Dwell feasibility: BPS over target size x dwell time; error breakdown; hold-still cursor wander.
B. GRU attribution, one factor at a time:
   - gain: closed_loop.json already has every decoder's calibration BPS at each gain;
   - simulated user: forward model off, half-speed user;
   - distribution shift, with no feedback: spikes generated open-loop from (i) real MC_Maze test
     reaches and (ii) Webgrid command streams recorded from oracle sessions; R^2 of each decoder's
     output against the command it should recover.
C. Trajectories for one seed (figure) and time-to-target.

Writes results/diagnostics.json, figures/trajectories.png, figures/dwell_feasibility.png.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbps import data  # noqa: E402
from cbps.encoding import CMD_LEAD, ENC_LAGS  # noqa: E402
from cbps.io import load_json, load_pickle, save_json  # noqa: E402
from cbps.metrics import bootstrap_ci, r2  # noqa: E402
from cbps.webgrid import Oracle, TaskConfig, UserModel, run_session  # noqa: E402

SEEDS = list(range(300, 308))
DECS = ["Oracle", "ReFIT-KF", "KF", "KF-lag", "Wiener", "GRU-16", "GRU-64"]
_G = {}


def _init(decs, enc):
    _G["decs"], _G["enc"] = decs, enc


def _job(a):
    key, name, cfg, user, seed = a
    dec = Oracle() if name == "Oracle" else _G["decs"][name].clone()
    res, log = run_session(dec, _G["enc"], cfg, user, seed, record=True)
    wrong_prev, wrong_other, prev = 0, 0, None
    for _, hit, ij in log["events"]:
        if hit:
            prev = ij
        elif ij == prev:
            wrong_prev += 1
        else:
            wrong_other += 1
    return key, name, seed, {**res.as_dict(), "wrong_reselect_previous": wrong_prev, "wrong_other_cell": wrong_other}


def agg(rows):
    m, lo, hi = bootstrap_ci([r["bps"] for r in rows])
    keys = ["correct", "incorrect", "wrong_reselect_previous", "wrong_other_cell"]
    out = {"bps_mean": m, "bps_ci95": [lo, hi]}
    out.update({f"{k}_per_min": float(np.mean([r[k] / r["seconds"] * 60 for r in rows])) for k in keys})
    t = [r["mean_time_to_target_s"] for r in rows if np.isfinite(r["mean_time_to_target_s"])]
    out["time_to_target_s"] = float(np.mean(t)) if t else None
    return out


def hold_still_wander(decs, enc):
    out = {}
    for n in DECS[1:]:
        disp = []
        for s in range(20):
            sm, d = enc.sampler(np.random.default_rng([s, 7])), decs[n].clone()
            d.reset()
            v = np.array([d.step(sm.step(np.zeros(2))) for _ in range(150)])[50:]
            x = np.cumsum(v * 0.02, 0)
            disp += [np.linalg.norm(x[i + 25] - x[i]) for i in range(0, 75, 5)]
        out[n] = {"median_mm_over_0p5s": float(np.median(disp)), "p90_mm_over_0p5s": float(np.percentile(disp, 90))}
    return out


def open_loop_shift(decs, enc, user, n_sessions=4):
    """R^2 of decoder output vs the command it should recover (c_{t-3}), no feedback."""
    lag = CMD_LEAD - data.DECODE_LAG
    # (i) real test reaches: the command stream is the hand velocity CMD_LEAD bins ahead
    full = data.load(data.DECODE_LAG)
    Vs = data.velocity_stack(ENC_LAGS)
    _, _, te = data.split(full)
    pos = np.searchsorted(full.trial_idx, te.trial_idx)
    trial_cmds = Vs[pos][:, :, -1]  # [N, T, 2] velocity at the largest lead == command c_t
    # (ii) Webgrid command streams recorded from oracle sessions
    grid_cmds = [run_session(Oracle(), enc, TaskConfig(mode="click", seconds=60.0), user, s, record=True)[1]["cmd"]
                 for s in range(500, 500 + n_sessions)]
    out = {}
    for n in DECS[1:]:
        res = {}
        for label, streams in (("mc_maze_reaches", list(trial_cmds)), ("webgrid_commands", grid_cmds)):
            ys, yh = [], []
            for k, cmd in enumerate(streams):
                sm, d = enc.sampler(np.random.default_rng([k, 11])), decs[n].clone()
                d.reset()
                v = np.array([d.step(sm.step(c)) for c in cmd])
                burn = 35 if label == "mc_maze_reaches" else 50
                ys.append(cmd[burn - lag:len(cmd) - lag])
                yh.append(v[burn:])
            ys, yh = np.concatenate(ys), np.concatenate(yh)
            sp = np.linalg.norm(ys, axis=1) > 100
            res[label] = {"r2": r2(ys, yh),
                          "speed_ratio_when_moving": float(np.median(np.linalg.norm(yh[sp], axis=1)
                                                                      / np.linalg.norm(ys[sp], axis=1))),
                          "frac_bins_near_rest": float((np.linalg.norm(ys, axis=1) < 30).mean())}
        out[n] = res
    return out


def fig_trajectories(decs, enc, user, gains, seed=0, seconds=12.0):
    cfg = TaskConfig(mode="click", seconds=seconds)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, n in zip(axes, ["Oracle", "KF", "GRU-64"]):
        dec = Oracle() if n == "Oracle" else decs[n].clone()
        res, log = run_session(dec, enc, replace(cfg, gain=gains[n]), user, seed, record=True)
        p, tg = log["p"], log["target"]
        h = cfg.width_mm / 2
        for k in np.linspace(-h, h, cfg.grid + 1):
            ax.axhline(k, color="#e4e3df", lw=0.5)
            ax.axvline(k, color="#e4e3df", lw=0.5)
        ax.plot(p[:, 0], p[:, 1], color="#2a78d6", lw=1.2)
        u = np.unique(tg, axis=0, return_index=True)[1]
        tgs = tg[np.sort(u)]
        ax.plot(tgs[:, 0], tgs[:, 1], "s", color="#eb6834", ms=6, mfc="none", mew=1.5)
        for i, t in enumerate(tgs[:8]):
            ax.annotate(str(i + 1), t, xytext=(4, 4), textcoords="offset points", fontsize=7, color="#52514e")
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{n} (gain {gains[n]}): {res.correct} correct, {res.incorrect} wrong in {seconds:.0f} s",
                     fontsize=9, loc="left")
    fig.suptitle("Cursor path (blue) and target sequence (orange squares, numbered), same seed, click mode",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "trajectories.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_dwell(sweep, grids, dwells):
    fig, axes = plt.subplots(1, len(dwells), figsize=(11, 3.6), sharey=True)
    cols = {"Oracle": "#52514e", "ReFIT-KF": "#6250d6", "KF": "#2a78d6", "Wiener": "#eb6834", "GRU-16": "#1baf7a",
            "GRU-64": "#eda100"}
    for ax, dw in zip(axes, dwells):
        for n, c in cols.items():
            y = [sweep[f"g{g}_d{dw}"][n]["bps_mean"] for g in grids]
            ax.plot([240 / g for g in grids], y, marker="o", color=c, lw=2, ms=5, label=n,
                    ls="--" if n == "Oracle" else "-")
        ax.set_title(f"dwell {dw} s", fontsize=10, loc="left")
        ax.set_xlabel("target (cell) width, mm")
        ax.grid(color="#e4e3df")
    axes[0].set_ylabel("BPS (dwell selection)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "dwell_feasibility.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    decs = load_pickle("decoders_cl.pkl")
    enc = load_pickle("encoder.pkl")
    um = load_json("user_model.json")
    cl = load_json("closed_loop.json")
    user = UserModel(vmax=um["vmax"], tau=um["tau"], tau_m=um["tau_m"])
    gains = {n: (1.0 if n == "Oracle" else cl["modes"]["click"]["decoders"][n]["gain"]) for n in DECS}
    # Dwell sweep uses the gains tuned for the main dwell condition (6x6, 1 s), the same ones as the main table.
    dgains = {n: (1.0 if n == "Oracle" else cl["modes"]["dwell"]["decoders"][n]["gain"]) for n in DECS}
    out = {"seeds": SEEDS, "session_seconds": 60.0}

    grids, dwells = [4, 6, 8, 12], [0.3, 0.5, 1.0]
    users = {"default (adaptive forward model)": user, "fixed_gain_forward_model": replace(user, forward_gain="fixed"),
             "no_forward_model": replace(user, forward_model=False),
             "half_speed": replace(user, vmax=user.vmax / 2),
             # Closer to a person using a BCI cursor: much slower, slower to react.
             "human_speed": replace(user, vmax=350.0, react_delay=15)}
    jobs = []
    for g in grids:
        for dw in dwells:
            for n in DECS:
                for s in SEEDS:
                    jobs.append((f"g{g}_d{dw}", n, TaskConfig(mode="dwell", grid=g, dwell_s=dw, seconds=60.0,
                                                              gain=dgains[n]), user, s))
    for uk, uu in users.items():
        for n in DECS:
            for s in SEEDS:
                jobs.append((f"user_{uk}", n, TaskConfig(mode="click", seconds=60.0, gain=gains[n]), uu, s))
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=_init, initargs=(decs, enc)) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    keys = sorted({r[0] for r in rows})
    res = {k: {n: agg([r[3] for r in rows if r[0] == k and r[1] == n]) for n in DECS} for k in keys}
    out["dwell_sweep"] = {"grids": grids, "cell_mm": [240 / g for g in grids], "dwell_s": dwells, "gains": dgains,
                          "results": {k: v for k, v in res.items() if k.startswith("g")}}
    ua = {k[5:]: v for k, v in res.items() if k.startswith("user_")}
    for v in ua.values():  # BPS as a fraction of what a perfect decoder achieves with the same user
        for n in DECS:
            v[n]["oracle_normalised_bps"] = v[n]["bps_mean"] / max(v["Oracle"]["bps_mean"], 1e-9)
    out["user_ablation_click"] = {"gains": gains, "results": ua}
    out["gain_curves_click"] = {n: cl["modes"]["click"]["decoders"][n]["cal_bps_by_gain"] for n in DECS if n != "Oracle"}
    out["hold_still_wander"] = hold_still_wander(decs, enc)
    out["open_loop_distribution_shift"] = open_loop_shift(decs, enc, user)
    save_json("diagnostics.json", out)
    fig_dwell(out["dwell_sweep"]["results"], grids, dwells)
    fig_trajectories(decs, enc, user, gains)

    print("dwell BPS (ReFIT-KF / KF / Oracle):")
    for g in grids:
        print(f"  cell {240 / g:5.1f} mm: " + "  ".join(
            f"d{dw}: {res[f'g{g}_d{dw}']['ReFIT-KF']['bps_mean']:.2f}/{res[f'g{g}_d{dw}']['KF']['bps_mean']:.2f}/"
            f"{res[f'g{g}_d{dw}']['Oracle']['bps_mean']:.2f}" for dw in dwells))
    print("user ablation (click BPS):")
    for uk in users:
        print(f"  {uk[:24]:24s} " + "  ".join(f"{n} {res['user_' + uk][n]['bps_mean']:.2f}"
                                              f"[{res['user_' + uk][n]['bps_ci95'][0]:.2f},{res['user_' + uk][n]['bps_ci95'][1]:.2f}]"
                                              for n in DECS))
    print("open-loop R^2 vs command (reaches -> webgrid):")
    for n, v in out["open_loop_distribution_shift"].items():
        print(f"  {n:9s} {v['mc_maze_reaches']['r2']:.2f} -> {v['webgrid_commands']['r2']:.2f}   speed ratio "
              f"{v['mc_maze_reaches']['speed_ratio_when_moving']:.2f} -> {v['webgrid_commands']['speed_ratio_when_moving']:.2f}"
              f"   rest frac {v['mc_maze_reaches']['frac_bins_near_rest']:.2f} -> {v['webgrid_commands']['frac_bins_near_rest']:.2f}")


if __name__ == "__main__":
    main()
