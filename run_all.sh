#!/usr/bin/env bash
# Full pipeline: data -> offline decoders -> simulator fit -> closed loop -> C++ latency -> figures.
# ~40 min on an M-series MacBook (CPU only). Each step writes JSON under results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY scripts/01_prepare_data.py
$PY scripts/02_train_offline.py
$PY scripts/03_fit_encoder.py
$PY scripts/04_closed_loop.py
$PY scripts/05_export_bench.py
$PY scripts/06_figures.py
$PY scripts/07_continuous_check.py
$PY scripts/08_diagnostics.py
