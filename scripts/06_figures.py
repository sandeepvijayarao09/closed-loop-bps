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
             "ReFIT-KF": "#6250d6", "KF-lag": "#e87ba4"}
ORDER = ["KF", "ReFIT-KF", "KF-lag", "Wiener", "Wiener-lag0", "GRU-16", "GRU-32", "GRU-64", "GRU-128", "GRU-256",
         "GRU-64-cont", "GRU-64-lag0", "ReFIT-GRU-64"]
# Variants share a parameter count with their base model; spread them horizontally (multiplicative,
# log x-axis) so every marker and label is visible and unambiguous.
JITTER = {"ReFIT-KF": 1.45, "Wiener-lag0": 1.45, "GRU-64-cont": 0.72, "GRU-64-lag0": 1.4, "ReFIT-GRU-64": 1.95}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})




# Label placement (points offset from the marker) for the crowded panels.
R2_LABELS = {"KF": (6, -3), "ReFIT-KF": (6, -3), "KF-lag": (6, -3), "Wiener": (6, -3), "GRU-16": (-14, 8),
             "GRU-256": (-12, 8), "ReFIT-GRU-64": (6, -3)}
BPS_OFFSETS = {"GRU-16": (-8, 7), "GRU-32": (-26, -13), "GRU-128": (6, -9), "GRU-256": (6, -3), "Wiener-lag0": (6, -3)}
BPS_TEXT = {"GRU-64": (2.2e4, 1.25), "GRU-64-lag0": (2.0e5, 1.7), "ReFIT-GRU-64": (1.4e5, 2.15),
            "GRU-64-cont": (5.5e4, -0.35)}


def marker(n):
    return "s" if n.startswith("ReFIT") else "D" if n.endswith("cont") else "^" if n.endswith("lag0") else "o"


def family(n):
    return "Kalman" if "KF" in n else "Wiener" if n.startswith("Wiener") else "GRU"


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
                     "r2_nlb": off[n].get("test_r2_nlb_window"),
                     "batched": lat[n]["cpp"]["batched_mean_us"], "p999": lat[n]["cpp"]["call_p999_us"],
                     "py_p50": lat[n]["python_numpy"]["p50_us"], "diff_kf": c["paired_diff_vs_KF"]})
    return rows, cl


def errbar(ax, x, y, ci, color, marker):
    ax.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt=marker, color=color, ms=8, elinewidth=1.2, capsize=0,
                mec=SURF, mew=1.5, zorder=3)


def fig_size(rows, cl):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
    titles = ["Offline velocity R² (real held-out trials)", "Closed-loop BPS (simulated Webgrid, click)",
              "C++ compute per 20 ms step (µs, batched)"]
    for r in rows:
        col, mk = FAMILY[family(r["name"])], marker(r["name"])
        x = r["params"] * JITTER.get(r["name"], 1.0)
        if not r["name"].endswith("lag0"):  # lag-0 R² is against a different target; see table
            errbar(axes[0], x, r["r2"], r["r2_ci"], col, mk)
        errbar(axes[1], x, r["bps"], r["bps_ci"], col, mk)
        axes[2].plot(x, r["batched"], mk, color=col, ms=8, mec=SURF, mew=1.5, zorder=3)
        if r["name"] in R2_LABELS:
            axes[0].annotate(r["name"], (x, r["r2"]), xytext=R2_LABELS[r["name"]], textcoords="offset points",
                             fontsize=7, color=INK2)
        if r["name"] in BPS_TEXT:  # crowded: label placed in data coordinates with a leader line
            axes[1].annotate(r["name"], (x, r["bps"]), xytext=BPS_TEXT[r["name"]], textcoords="data", fontsize=7,
                             color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5})
        else:
            axes[1].annotate(r["name"], (x, r["bps"]), xytext=BPS_OFFSETS.get(r["name"], (6, -3)),
                             textcoords="offset points", fontsize=7, color=INK2)
    axes[0].annotate("GRU-32…128, GRU-64-cont", (3.5e4, 0.85), xytext=(-40, -38), textcoords="offset points",
                     fontsize=7, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    orc = cl["modes"]["click"]["decoders"]["Oracle"]["bps_mean"]
    axes[1].axhline(orc, color=INK2, ls="--", lw=1)
    axes[1].text(600, orc * 0.93, f"oracle decoder (true intent): {orc:.1f}", fontsize=8, color=INK2, va="top")
    axes[2].axhline(10000, color=INK2, ls="--", lw=1)
    axes[2].text(600, 10000 * 0.8, "10 ms budget", fontsize=8, color=INK2, va="top")
    axes[2].set_yscale("log")
    axes[1].set_ylim(-0.55, None)
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlim(350, 1.3e6)
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("decoder parameters (variants offset sideways)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=f) for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], marker="s", ls="", color=INK2, ms=7, label="ReFIT (recalibrated in sim)"),
                plt.Line2D([], [], marker="D", ls="", color=INK2, ms=7, label="continuous-crop training"),
                plt.Line2D([], [], marker="^", ls="", color=INK2, ms=7, label="decode lag 0")]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.savefig(FIG / "bps_vs_size_vs_latency.png", dpi=160)
    plt.close(fig)


def fig_r2_bps(rows):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for r in rows:
        if r["name"].endswith("lag0"):  # different R² target
            continue
        col = FAMILY[family(r["name"])]
        ax.errorbar(r["r2"], r["bps"], xerr=[[r["r2"] - r["r2_ci"][0]], [r["r2_ci"][1] - r["r2"]]],
                    yerr=[[r["bps"] - r["bps_ci"][0]], [r["bps_ci"][1] - r["bps"]]], fmt=marker(r["name"]),
                    color=col, ms=7, mec=SURF, mew=1.5, elinewidth=1.2)
        if not r["name"].startswith("GRU-") or r["name"] == "GRU-16":
            ax.annotate(r["name"], (r["r2"], r["bps"]), xytext=(6, 3), textcoords="offset points", fontsize=8,
                        color=INK2)
    gru = [r for r in rows if r["name"].startswith("GRU-") and r["name"] not in ("GRU-16",)
           and not r["name"].endswith("lag0")]
    if gru:
        x, y = np.mean([r["r2"] for r in gru]), np.mean([r["bps"] for r in gru])
        ax.annotate("GRU-32…256, GRU-64-cont", (x, y), xytext=(-150, -18), textcoords="offset points", fontsize=8,
                    color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlabel("offline velocity R² (real test trials)")
    ax.set_ylabel("closed-loop achieved bitrate (BPS)")
    ax.set_title("Offline R² does not rank closed-loop BPS (MC_Maze sim)", fontsize=10, loc="left", color=INK)
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
    for n in ["KF-lag", "KF", "ReFIT-KF", "Wiener", "GRU-64"]:
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
    lines = ["| Decoder | Params | R² test [−500, +800] ms (95% CI) | R² test, NLB [−250, +450] ms | "
             "Sim BPS, click (95% CI) | gain | Sim BPS, dwell 6×6 / 1 s (95% CI) | C++ µs/step (batched) | "
             "C++ per-call p99.9 (µs) |", "|---|---:|---|---|---|---:|---|---|---|"]
    for r in rows:
        name = r["name"] + (" †" if r["name"].endswith("lag0") else "")
        lines.append(f"| {name} | {r['params']:,} | {r['r2']:.3f} [{r['r2_ci'][0]:.3f}, {r['r2_ci'][1]:.3f}] | "
                     f"{r['r2_nlb']:.3f} | " if r["r2_nlb"] is not None else
                     f"| {name} | {r['params']:,} | {r['r2']:.3f} [{r['r2_ci'][0]:.3f}, {r['r2_ci'][1]:.3f}] | n/a | ")
        lines[-1] += (f"{r['bps']:.2f} [{r['bps_ci'][0]:.2f}, {r['bps_ci'][1]:.2f}] | {r['gain']} | "
                      f"{r['dwell']:.2f} [{r['dwell_ci'][0]:.2f}, {r['dwell_ci'][1]:.2f}] | "
                      f"{r['batched']:.2f} | {r['p999']:.1f} |")
    lines.append(f"| Oracle (decoded = true intent) | 0 | n/a | n/a | {orc['bps_mean']:.2f} "
                 f"[{orc['bps_ci95'][0]:.2f}, {orc['bps_ci95'][1]:.2f}] | {orc['gain']} | "
                 f"{cl['modes']['dwell']['decoders']['Oracle']['bps_mean']:.2f} | n/a | n/a |")
    lines.append("")
    lines.append("† decode lag 0: R² is against velocity in the same bin as the spikes (an easier target than the "
                 "100 ms-ahead velocity in the other rows); in closed loop its output trails intent by 100 ms more.")
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
