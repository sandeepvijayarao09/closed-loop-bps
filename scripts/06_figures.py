"""Figures and the README results table, from results/*.json only.

Usage: python scripts/06_figures.py   -> figures/*.png, results/summary.md
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbps.io import load_json  # noqa: E402

FIG = ROOT / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILY = {"Kalman": "#2a78d6", "Wiener": "#eb6834", "GRU": "#1baf7a"}  # categorical slots 1-3
# Per-decoder colours for line charts; the same decoder keeps its colour in every figure.
DEC_COLOR = {"KF": "#2a78d6", "Wiener": "#eb6834", "GRU-64": "#1baf7a", "ReFIT-GRU-64": "#eda100",
             "ReFIT-KF": "#6250d6"}
ORDER = ["KF", "ReFIT-KF", "Wiener", "GRU-16", "GRU-32", "GRU-64", "GRU-128", "GRU-256", "GRU-64-cont",
         "ReFIT-GRU-64"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})


LABEL_OFFSETS = {"KF": (7, -11), "ReFIT-KF": (7, 4), "Wiener": (-12, 9), "GRU-16": (6, -12), "GRU-256": (-20, 8),
                 "ReFIT-GRU-64": (7, 4)}


def family(n):
    return "Kalman" if "KF" in n else "Wiener" if n == "Wiener" else "GRU"


def collect():
    off = load_json("offline.json")["decoders"]
    cl = load_json("closed_loop.json")
    lat = load_json("latency.json")["decoders"]
    for n, v in cl["refit"]["real_data_r2"].items():
        off[n] = v
    rows = []
    for n in ORDER:
        if n not in cl["modes"]["click"]["decoders"]:
            continue
        c, d = cl["modes"]["click"]["decoders"][n], cl["modes"]["dwell"]["decoders"][n]
        rows.append({"name": n, "params": off[n]["n_params"], "r2": off[n]["test_r2"], "r2_ci": off[n]["test_r2_ci95"],
                     "bps": c["bps_mean"], "bps_ci": c["bps_ci95"], "gain": c["gain"],
                     "bps_untuned": c["untuned_gain1"]["bps_mean"], "dwell": d["bps_mean"], "dwell_ci": d["bps_ci95"],
                     "p50": lat[n]["cpp"]["p50_us"], "p99": lat[n]["cpp"]["p99_us"],
                     "py_p50": lat[n]["python_numpy"]["p50_us"], "diff_kf": c["paired_diff_vs_KF"]})
    return rows, cl


def errbar(ax, x, y, ci, color, marker):
    ax.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt=marker, color=color, ms=8, elinewidth=1.2, capsize=0,
                mec=SURF, mew=1.5, zorder=3)


def fig_size(rows, cl):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    titles = ["Offline velocity R² (real held-out trials)", "Closed-loop BPS (simulated Webgrid, click)",
              "C++ per-step latency, p99 (µs)"]
    for r in rows:
        col = FAMILY[family(r["name"])]
        mk = "s" if r["name"].startswith("ReFIT") else "D" if r["name"].endswith("cont") else "o"
        errbar(axes[0], r["params"], r["r2"], r["r2_ci"], col, mk)
        errbar(axes[1], r["params"], r["bps"], r["bps_ci"], col, mk)
        axes[2].plot(r["params"], r["p99"], mk, color=col, ms=8, mec=SURF, mew=1.5, zorder=3)
        for ax, y in ((axes[0], r["r2"]), (axes[1], r["bps"])):
            if r["name"] in LABEL_OFFSETS:
                ax.annotate(r["name"], (r["params"], y), xytext=LABEL_OFFSETS[r["name"]], textcoords="offset points",
                            fontsize=8, color=INK2)
    orc = cl["modes"]["click"]["decoders"]["Oracle"]["bps_mean"]
    axes[1].axhline(orc, color=INK2, ls="--", lw=1)
    axes[1].text(600, orc * 0.93, f"oracle decoder (true intent): {orc:.1f}", fontsize=8, color=INK2, va="top")
    axes[2].axhline(10000, color=INK2, ls="--", lw=1)
    axes[2].text(600, 10000 * 0.8, "10 ms budget", fontsize=8, color=INK2, va="top")
    axes[2].set_yscale("log")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("decoder parameters")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=f) for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], marker="s", ls="", color=INK2, ms=7, label="ReFIT (recalibrated in sim)"),
                plt.Line2D([], [], marker="D", ls="", color=INK2, ms=7, label="continuous-crop training")]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.savefig(FIG / "bps_vs_size_vs_latency.png", dpi=160)
    plt.close(fig)


def fig_r2_bps(rows):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for r in rows:
        col = FAMILY[family(r["name"])]
        ax.errorbar(r["r2"], r["bps"], xerr=[[r["r2"] - r["r2_ci"][0]], [r["r2_ci"][1] - r["r2"]]],
                    yerr=[[r["bps"] - r["bps_ci"][0]], [r["bps_ci"][1] - r["bps"]]], fmt="o", color=col, ms=8,
                    mec=SURF, mew=1.5, elinewidth=1.2)
        if r["name"] in ("KF", "ReFIT-KF", "Wiener", "GRU-16", "ReFIT-GRU-64"):
            ax.annotate(r["name"], (r["r2"], r["bps"]), xytext=(6, 3), textcoords="offset points", fontsize=8,
                        color=INK2)
    ax.annotate("GRU-32…256, GRU-64-cont", (0.86, 0.06), xytext=(-140, -4), textcoords="offset points", fontsize=8,
                color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlabel("offline velocity R² (real test trials)")
    ax.set_ylabel("closed-loop achieved bitrate (BPS)")
    ax.set_title("Offline R² does not rank closed-loop BPS", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "r2_vs_bps.png", dpi=160)
    plt.close(fig)


def fig_probe():
    p = load_json("probe_sustained.json")
    t = (np.arange(len(p["command_mm_s"])) - p["onset_bin"]) * 20
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(t, p["command_mm_s"], color=INK2, lw=1.5, ls="--", label="user command")
    for n in ["KF", "Wiener", "GRU-64", "ReFIT-GRU-64"]:
        ax.plot(t, p["decoders"][n]["mean_vx"], color=DEC_COLOR[n], lw=2, label=n)
    ax.set_xlabel("ms from command onset (command held for 2 s)")
    ax.set_ylabel("decoded vx (mm/s), mean of 20 runs")
    ax.set_title("Response to a held 500 mm/s command (simulated spikes)", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "sustained_command_probe.png", dpi=160)
    plt.close(fig)


def fig_drift(cl):
    d = cl["modes"]["click"]["drift"]
    lv = d["levels_log_rate_std_per_10min"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for n in ["ReFIT-KF", "KF", "Wiener", "GRU-64"]:
        col = DEC_COLOR[n]
        m = [d["decoders"][n][str(x)]["bps_mean"] for x in lv]
        lo = [d["decoders"][n][str(x)]["bps_ci95"][0] for x in lv]
        hi = [d["decoders"][n][str(x)]["bps_ci95"][1] for x in lv]
        ax.plot(lv, m, "-o", color=col, lw=2, ms=6, mec=SURF, label=n)
        ax.fill_between(lv, lo, hi, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("per-neuron log-rate drift (std after 10 min)")
    ax.set_ylabel("BPS over a 10-min session")
    ax.set_title("Within-session drift, no recalibration", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "drift.png", dpi=160)
    plt.close(fig)


def table(rows, cl):
    orc = cl["modes"]["click"]["decoders"]["Oracle"]
    lines = ["| Decoder | Params | Offline R² (test, 95% CI) | Sim BPS, click (95% CI) | gain | Sim BPS, dwell (6×6, 1 s) | "
             "C++ p50 / p99 (µs) |", "|---|---:|---|---|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['params']:,} | {r['r2']:.3f} [{r['r2_ci'][0]:.3f}, {r['r2_ci'][1]:.3f}] | "
                     f"{r['bps']:.2f} [{r['bps_ci'][0]:.2f}, {r['bps_ci'][1]:.2f}] | {r['gain']} | "
                     f"{r['dwell']:.2f} [{r['dwell_ci'][0]:.2f}, {r['dwell_ci'][1]:.2f}] | "
                     f"{r['p50']:.1f} / {r['p99']:.1f} |")
    lines.append(f"| Oracle (decoded = true intent) | 0 | n/a | {orc['bps_mean']:.2f} "
                 f"[{orc['bps_ci95'][0]:.2f}, {orc['bps_ci95'][1]:.2f}] | {orc['gain']} | "
                 f"{cl['modes']['dwell']['decoders']['Oracle']['bps_mean']:.2f} | n/a |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    rows, cl = collect()
    fig_size(rows, cl)
    fig_r2_bps(rows)
    fig_probe()
    fig_drift(cl)
    table(rows, cl)


if __name__ == "__main__":
    main()
