#!/usr/bin/env bash
# Full pipeline: data -> offline decoders -> simulator fit -> closed loop -> C++ latency -> figures ->
# checks and controls -> MC_RTT replication. ~60 min on an M-series MacBook (CPU only).
# Each step writes JSON under results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY scripts/01_prepare_data.py        # MC_Maze download (691 MB, sha256-checked), binning, lag selection
$PY scripts/02_train_offline.py       # KF, KF-lag, Wiener, GRUs, ablations; offline R²
$PY scripts/03_fit_encoder.py         # encoding model, user policy, simulator fidelity
$PY scripts/04_closed_loop.py         # ReFIT, closed-loop BPS (click, dwell), drift, probe
$PY scripts/05_export_bench.py        # C++ export, parity check, latency (incl. 60 s paced loops)
$PY scripts/07_continuous_check.py    # continuous vs per-trial decoding on real data
$PY scripts/08_diagnostics.py         # dwell feasibility, user-model ablation, open-loop shift
$PY scripts/09_refit_controls.py      # ReFIT label masking / oracle labels / both users
$PY scripts/10_sim_trained.py         # confound control: decoders trained on simulated spikes
$PY scripts/11_mc_rtt.py              # replication on MC_RTT (50 MB download)
$PY scripts/06_figures.py             # figures + results table from results/*.json
