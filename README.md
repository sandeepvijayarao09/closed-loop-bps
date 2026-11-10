# closed-loop-bps

[![ci](https://github.com/sandeepvijayarao09/closed-loop-bps/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sandeepvijayarao09/closed-loop-bps/actions/workflows/ci.yml)

A reproducible harness for asking: **does offline decoding accuracy predict closed-loop cursor control?**

The harness:
- trains intracortical velocity decoders on real monkey motor-cortex data (NLB MC_Maze, plus MC_RTT as a replication);
- runs them in a simulated Webgrid task, where a simulated user steers by watching the decoded cursor;
- scores them in **bits per second** (the achieved bitrate BrainGate and Neuralink report);
- runs every decoder in a dependency-free C++ loop on 20 ms bins.

It is not a new finding that offline accuracy can mislead (see [Related work](#related-work-and-what-this-repo-adds)).
The contribution is the harness and the failure-mode diagnostics: controls that separate *simulator mismatch* from
*distribution shift*, ablations of the simulated user and of ReFIT labels, and a measured simulator fidelity.

## In 30 seconds

- **What it is:** a reproducible harness that trains intracortical cursor decoders on real monkey motor-cortex data
  (NLB MC_Maze, with MC_RTT as a replication), runs them closed loop in a simulated Webgrid task with a simulated
  user, scores them in bits per second (BPS), and runs each one in a dependency-free C++ loop on 20 ms bins.
- **Why it matters:** BCI decoders are usually chosen by offline accuracy, but users care about closed-loop control.
  This harness measures where the two disagree, and why.
- **Key result:** offline R² does not rank closed-loop BPS here. GRUs have the best offline R² (0.845–0.881) but
  score 0.47–0.79 BPS; a lag-augmented Kalman filter (R² 0.553) scores best, **3.07 BPS** [2.97, 3.16], against a
  9.24 BPS simulator ceiling (6.86 with a human-speed user). Two measured causes: the simulator favours linear
  decoders (GRU-64 trained on simulated spikes recovers from 0.75 to 2.15 BPS), and MC_Maze's isolated reaches differ
  from continuous control (on MC_RTT, GRU-16 trails the KF by only 0.22 BPS). Simulated BPS is **not comparable** to
  Neuralink's or BrainGate's human numbers; read every ranking as a statement about this simulator.
- **What I did:** built the decoders, the encoding model and simulated user, the Webgrid scoring, the confound
  controls (simulated-spike training, MC_RTT, user-model and ReFIT-label ablations) and the C++ loop with parity tests.
- **Built with:** Python, NumPy, PyTorch (GRU training), C++17, pytest, GitHub Actions; public DANDI data.
- **Try it:** `pytest -q tests` (synthetic data, no download), or `./run_all.sh` for the full pipeline.

Every number below comes from a file in [`results/`](results/): the table and figures are generated from them, and
numbers in the prose are quoted from them.

## Results

MC_Maze, 182 units, 20 ms bins. 1,567 / 335 / 336 train / val / test trials (chronological). BPS: simulated Webgrid,
40 two-minute sessions per decoder, the same seeds for every decoder, mean with 95% bootstrap CI. Click: 12×12 grid
(N = 144, 7.16 bits per net correct selection). Dwell: 6×6 grid, 40 mm targets, 1 s dwell. Latency: C++ on Apple M5
Pro, single thread. "Batched" is the mean over 1,000-step blocks; per-call p99.9 is the tail when each step is timed
on its own.

| Decoder | Params | R² test [−500, +800] ms (95% CI) | R² test, NLB [−250, +450] ms | Sim BPS, click (95% CI) | gain | Sim BPS, dwell 6×6 / 1 s (95% CI) | C++ µs/step (batched) | C++ per-call p99.9 (µs) |
|---|---:|---|---|---|---:|---|---|---|
| KF | 550 | 0.216 [0.193, 0.238] | 0.306 | 2.46 [2.39, 2.53] | 0.25 | 1.77 [1.74, 1.81] | 0.11 | 0.3 |
| ReFIT-KF | 550 | 0.146 [0.133, 0.157] | n/a | 1.83 [1.75, 1.91] | 0.7 | 1.46 [1.42, 1.50] | 0.11 | 0.3 |
| KF-lag | 2,510 | 0.553 [0.532, 0.571] | 0.630 | 3.07 [2.97, 3.16] | 0.35 | 2.34 [2.32, 2.37] | 0.56 | 0.9 |
| Wiener | 9,284 | 0.654 [0.638, 0.668] | 0.713 | 1.65 [1.56, 1.75] | 0.5 | 1.56 [1.51, 1.62] | 4.64 | 11.9 |
| Wiener-lag0 † | 9,284 | 0.720 [0.707, 0.731] | 0.772 | 2.32 [2.23, 2.41] | 0.35 | 1.91 [1.87, 1.95] | 4.79 | 24.9 |
| GRU-16 | 9,998 | 0.845 [0.832, 0.856] | 0.851 | 0.79 [0.72, 0.86] | 0.5 | 0.02 [0.00, 0.04] | 2.31 | 3.1 |
| GRU-32 | 21,166 | 0.847 [0.830, 0.862] | 0.856 | 0.47 [0.40, 0.54] | 2.0 | 0.00 [0.00, 0.01] | 4.98 | 27.9 |
| GRU-64 | 48,110 | 0.853 [0.834, 0.869] | 0.861 | 0.75 [0.66, 0.84] | 1.4 | 0.07 [0.03, 0.10] | 11.06 | 24.8 |
| GRU-128 | 120,430 | 0.867 [0.854, 0.879] | 0.874 | 0.65 [0.58, 0.72] | 2.0 | 0.08 [0.04, 0.13] | 27.87 | 55.5 |
| GRU-256 | 338,798 | 0.881 [0.865, 0.893] | 0.886 | 0.59 [0.51, 0.67] | 1.0 | 0.00 [0.00, 0.01] | 93.27 | 193.5 |
| GRU-64-cont | 48,110 | 0.857 [0.843, 0.869] | 0.863 | 0.17 [0.13, 0.23] | 2.8 | 0.00 [0.00, 0.00] | 10.69 | 34.4 |
| GRU-64-lag0 † | 48,110 | 0.892 [0.883, 0.900] | 0.897 | 1.12 [1.04, 1.20] | 0.7 | 1.27 [1.18, 1.35] | 10.83 | 45.5 |
| ReFIT-GRU-64 | 48,110 | 0.222 [0.204, 0.238] | n/a | 1.28 [1.20, 1.37] | 0.7 | 0.94 [0.87, 1.00] | 10.77 | 19.0 |
| Oracle (decoded = true intent) | 0 | n/a | n/a | 9.24 [9.20, 9.28] | 1.0 | 3.58 | n/a | n/a |

† decode lag 0: R² is against velocity in the same bin as the spikes (an easier target than the 100 ms-ahead velocity in the other rows); in closed loop its output trails intent by 100 ms more.

*`gain`* is the click-mode output gain, tuned per decoder on calibration seeds over 0.09–4.0. Sources:
`results/offline.json`, `closed_loop.json`, `latency.json`, generated into `results/summary.md` by
`scripts/06_figures.py`. ReFIT rows: real-data R² of decoders recalibrated on simulated spikes.

![BPS vs model size vs latency](figures/bps_vs_size_vs_latency.png)

### 1. Offline R² does not rank closed-loop BPS

Offline, the ranking is GRU (0.845–0.881) > Wiener (0.654) > KF-lag (0.553) > KF (0.216). In closed loop it is
KF-lag (3.07) > KF (2.46) > Wiener (1.65) > GRUs (0.47–0.79). Paired differences vs the KF:
- KF-lag +0.61 [0.50, 0.72];
- Wiener −0.81 [−0.92, −0.69];
- every trial-trained GRU between −1.67 and −1.99, all CIs excluding 0.

The largest GRU has the best R² and among the worst BPS.

![R² vs BPS](figures/r2_vs_bps.png)

The KF baseline was underfit in an earlier version (R² 0.216). A velocity-only KF can't express that M1 units encode
velocity over 0–160 ms of lead. **KF-lag** puts velocity at leads 0, 40, 80, 100, 120 and 160 ms in the state
(`cbps/decoders.py: KalmanVelocity.fit_lagged`). That brings it to 0.553 (NLB window: 0.630 vs Wiener 0.713), and it
turned out to be the best closed-loop decoder. Other variants tried on validation, all well below KF-lag:
- causal 60–100 ms bins: ≤ 0.273;
- speed + position in the state: ≤ 0.335.

### 2. Why the GRUs lose: simulator mismatch *and* distribution shift

Two controls separate the explanations.

**Control A: train every decoder on simulated spikes** (`10_sim_trained.py`). Spikes are generated by the encoding
model from the real MC_Maze training velocities, then decoders are trained with true labels and the same
hyperparameters, and run closed loop.

| decoder | BPS, trained on real spikes | BPS, trained on simulated spikes | paired vs KF (sim-trained) |
|---|---|---|---|
| KF | 2.46 [2.39, 2.53] | 2.39 [2.27, 2.50] | 0 |
| KF-lag | 3.07 [2.97, 3.16] | 3.08 [2.97, 3.19] | +0.69 [0.50, 0.86] |
| Wiener | 1.65 [1.56, 1.75] | 2.56 [2.41, 2.70] | +0.17 [−0.04, 0.38] |
| GRU-16 | 0.79 [0.72, 0.86] | 1.45 [1.31, 1.59] | −0.94 [−1.12, −0.76] |
| GRU-64 | 0.75 [0.66, 0.84] | 2.15 [1.99, 2.29] | −0.24 [−0.41, −0.08] |

When decoders are trained on the simulator's own neurons, the Wiener filter and GRU-64 gain about 0.9–1.4 BPS, while
the KFs don't move. **Most of the GRUs' original deficit is simulator mismatch.** The real-trained GRUs rely on
population structure that the velocity-GLM simulator doesn't produce. A residual gap remains, largest for GRU-16.

**Control B: replicate on MC_RTT** (`11_mc_rtt.py`). MC_RTT is continuous random-target reaching (a new target
roughly every second), much closer to Webgrid than MC_Maze's hold → go → reach trials. Everything is refit on MC_RTT,
including the user's reach policy (v_max 400 mm/s, fit to 370 MC_RTT reaches), with a 140 mm, 8×8 grid:

| decoder | MC_RTT test R² | closed-loop BPS | paired vs KF |
|---|---|---|---|
| KF | 0.094 [−0.062, 0.203] | 2.61 [2.53, 2.68] | 0 |
| KF-lag | 0.425 [0.322, 0.495] | 3.18 [3.06, 3.30] | +0.58 [0.44, 0.72] |
| Wiener | 0.493 [0.420, 0.545] | 2.80 [2.70, 2.90] | +0.20 [0.10, 0.29] |
| GRU-16 | 0.509 [0.422, 0.569] | 2.38 [2.24, 2.53] | −0.22 [−0.38, −0.07] |
| GRU-64 | 0.550 [0.433, 0.633] | 1.95 [1.82, 2.07] | −0.66 [−0.78, −0.54] |
| Oracle | n/a | 7.30 [7.25, 7.35] | |

Trained on continuous reaching, GRU-16 comes within 0.22 BPS of the KF; on MC_Maze it trailed by 1.67. That is
consistent with **distribution shift** being a real second cause. GRUs still don't lead, and on MC_RTT too their
simulated-spike R² (0.37) is below their real R² (0.51–0.55): the simulator bias persists. MC_RTT is small (10.8 min,
225 training windows), so its offline CIs are wide.

**Supporting evidence for distribution shift** (`diagnostics.json → open_loop_distribution_shift`). These are open
loop, with no user reacting to the cursor. On MC_Maze-style reaches, real-trained GRU-16 delivers 60% of the
commanded speed. On Webgrid command streams (35% of bins near rest, vs 64% in MC_Maze windows) it delivers 24%.
KF, KF-lag and Wiener deliver 53–64% on both.

**The held-command probe is descriptive, not adjudicating.** Under a held 500 mm/s command, the KF sustains
366 mm/s and GRU-64 decays to 21 mm/s (`probe_sustained.json`). But the probe's spikes come from the encoding model,
whose velocity tuning is exactly the KF's generative assumption, so it is circular as evidence. It shows *what*
happens, not *why*.

The obvious timing explanation is ruled out:
- *Not trial timing:* GRU-64 trained on random 4 s crops of the continuous session with channel augmentation
  (GRU-64-cont) scores 0.857 R² and 0.17 BPS.
- *Not resets:* no decoder loses more than 0.014 R² when the 17-minute real test block is decoded continuously
  instead of per trial (`continuous_check.json`).

![trajectories](figures/trajectories.png)

For the same seed and 12 s, the oracle, KF and GRU-64 paths are shown. The KF's mean time to target is 2.5 s and
GRU-64's is 6.6 s.

### 3. Decode lag: a measured trade-off

All main decoders predict velocity 100 ms after the spikes (lag 5). That lag was chosen on validation as the argmax of
held-out encoding-model likelihood. A Wiener filter's *offline* R² is instead best at lag 0 (val 0.707 vs 0.638;
`lag_selection.json`). Lag 0 asks for velocity that is mostly already under way, which is easier offline but adds
100 ms of latency in closed loop. Both are reported:
- Wiener-lag0: R² 0.720 vs 0.654, and **2.32 vs 1.65 BPS**;
- GRU-64-lag0: R² 0.892 vs 0.853, and 1.12 vs 0.75 BPS.

In this simulator the accuracy gained outweighs the latency added. Lag 5 is kept for the main table because the
simulator is built on it, but it is not the closed-loop optimum for history decoders.

### 4. The simulated user's correction model changes the ranking

The user estimates the current cursor position from the delayed view plus in-flight commands (an efference-copy
forward model). Without one, even a perfect decoder oscillates under ~160 ms of loop delay. The default user learns
the cursor's gain online. An earlier version fixed that gain at 1 (the user assumed the cursor moves exactly as hard
as they push), which unfairly penalizes low-gain decoders. Click BPS, 8 one-minute seeds, 95% CI, and fraction of the
oracle with the same user (`diagnostics.json → user_ablation_click`):

| user model | Oracle | KF | KF-lag | ReFIT-KF | Wiener | GRU-16 | GRU-64 |
|---|---|---|---|---|---|---|---|
| adaptive forward model (default) | 9.17 | 2.68 [2.54, 2.82] | 2.71 [2.28, 3.01] | 2.00 [1.61, 2.25] | 1.67 [1.40, 1.95] | 0.81 [0.64, 0.98] | 0.79 [0.61, 0.97] |
| fixed-gain forward model | 9.19 | 1.16 [1.01, 1.33] | 1.00 [0.79, 1.21] | 0.63 [0.42, 0.84] | 0.37 [0.19, 0.54] | 0.24 [0.09, 0.40] | 0.31 [0.16, 0.45] |
| no forward model | 0.00 | 2.40 [2.24, 2.58] | 4.39 [4.13, 4.61] | 0.25 [0.10, 0.45] | 2.48 [2.13, 2.85] | 1.07 [0.87, 1.28] | 0.75 [0.48, 1.00] |
| half-speed user | 8.49 | 2.12 [1.95, 2.28] | 2.04 [1.78, 2.28] | 2.00 [1.76, 2.21] | 1.34 [1.06, 1.61] | 0.55 [0.42, 0.69] | 0.51 [0.39, 0.63] |
| human-speed user (350 mm/s, 300 ms reaction) | 6.86 | 1.46 [1.40, 1.54] | 1.54 [1.42, 1.70] | 1.72 [1.49, 1.95] | 1.06 [0.88, 1.30] | 0.49 [0.31, 0.66] | 0.51 [0.43, 0.58] |
| *fraction of oracle, human-speed user* | 1 | 0.21 | 0.22 | 0.25 | 0.15 | 0.07 | 0.07 |

Orderings among KF, KF-lag, ReFIT-KF and Wiener flip from row to row. The trial-trained GRUs are last in every row.

### 5. ReFIT: diagnosed, not just caveated

`09_refit_controls.py` refits on the same 20 minutes of closed-loop KF data with three label variants, under both
user models, 20 sessions each:
- **rotated, masked:** labels in the bins right after each target change are masked (13.8% of bins), because they
  point at the new target before the user has reacted;
- **rotated, unmasked:** the original, buggy version;
- **oracle:** the true intent c_{t−3}.

| variant | adaptive user: BPS (paired vs KF) | fixed-gain user: BPS (paired vs KF) | delivered speed / intent, open loop |
|---|---|---|---|
| KF | 2.43 | 1.63 | 0.55 |
| ReFIT-KF, rotated, masked | 1.88 (−0.55 [−0.71, −0.41]) | 2.49 (+0.85 [0.66, 1.05]) | 0.25 / 0.43 |
| ReFIT-KF, rotated, unmasked | 2.02 (−0.41 [−0.53, −0.31]) | 2.46 (+0.83 [0.65, 1.00]) | 0.26 / 0.44 |
| ReFIT-KF, oracle labels | **2.74 (+0.31 [0.23, 0.40])** | 2.52 (+0.89 [0.71, 1.07]) | 0.32 / 0.45 |
| ReFIT-GRU-64, rotated, masked | 1.32 (−1.12 [−1.26, −0.97]) | 1.32 (−0.31 [−0.41, −0.21]) | 0.38 / 0.32 |
| ReFIT-GRU-64, oracle labels | 1.83 (−0.61 [−0.70, −0.51]) | 1.81 (+0.17 [0.02, 0.33]) | 0.94 / 0.70 |

(Speed column: adaptive / fixed-gain calibration.)

- **Masking the target-switch bins made no measurable difference.** The bug was real but small.
- **Under the adaptive user, the rotated-label heuristic is what makes ReFIT-KF lose.** Rotated labels inherit the
  KF's shrunken decoded speed. Calibrated in that regime, ReFIT-KF delivers only 25% of the intended speed, needs a
  higher gain, and that gain amplifies its jitter.
- **With true-intent labels, ReFIT-KF beats the KF** (+0.31). Under the fixed-gain user every ReFIT-KF variant beats
  the KF by about +0.85.
- So ReFIT's benefit here depends on both label quality and the user model. ReFIT-GRU-64's real-data R² collapses
  (0.002 with oracle labels), because it fits the simulator's neurons, not the monkey's.

### 6. Dwell selection

On a 12×12 grid with 0.5 s dwell, every real decoder scores about 0 BPS while the oracle scores 6.7. The errors are
0.5 s dwells on non-target cells (KF: 24.9 per minute vs 19.1 correct; re-selecting the previous target: 1.3 per
minute), because cursor wander with zero intended movement is 17.5–25 mm per 0.5 s (median, across decoders) on 20 mm targets.

The main table therefore uses 40 mm targets with 1 s dwell. There, KF-lag scores 2.34 BPS and KF 1.77, and the GRUs
stay near 0 because a stalling cursor dwells on wrong cells. The sweep uses the same tuned gains as the main table.

![dwell feasibility](figures/dwell_feasibility.png)

### 7. Drift

10-minute sessions, per-neuron log-rate random walk (std 1.0 after 10 min), no recalibration:

| decoder | no drift | drift 1.0 |
|---|---|---|
| KF-lag | 3.06 [2.97, 3.16] | 1.62 [1.31, 1.92] |
| KF | 2.48 [2.44, 2.52] | 1.95 [1.73, 2.17] |
| ReFIT-KF | 1.81 [1.73, 1.87] | 1.20 [1.02, 1.37] |
| Wiener | 1.60 [1.51, 1.68] | 0.80 [0.53, 1.13] |
| GRU-64 | 0.67 [0.58, 0.74] | 0.43 [0.34, 0.53] |

The richer KF-lag degrades fastest. Under strong drift the simple KF overtakes it.

![drift](figures/drift.png)

### 8. Latency

Compute is negligible:
- batched C++ step time: 0.11 µs (KF), 0.56 µs (KF-lag), 4.6 µs (Wiener), 2.3–93 µs (GRU-16 … GRU-256);
- per-call p99.9 ≤ 194 µs for every decoder;
- every decoder reproduces the Python outputs to ≤ 2.8e-6 relative error.

Timing each step on its own was misleading for the KF: the 41 ns clock quantises a 110 ns step, which is why batched
timing is reported.

Scheduling dominates, and it depends entirely on how the thread waits. Paced 20 ms loops, 60 s (3,000 ticks) each
(`latency.json → paced`):

| wait strategy | KF wake + compute, p99 | GRU-256 wake + compute, p99 | missed 20 ms deadlines |
|---|---|---|---|
| `sleep_until`, default thread | 4.9 ms | 9.8 ms | 0 / 0 |
| Mach real-time thread + `mach_wait_until` | **15 µs** | 2.4 ms | 0 / 0 |
| busy-wait, default thread | 5 µs | 114 µs | 0 / 0 |
| busy-wait on a real-time thread | 0.8 ms (max 84 ms) | 0.4 ms | **7** / 0 |

The 4–10 ms wake-up lateness of a plain sleep is macOS timer coalescing for an ordinary background process
(USER_INTERACTIVE QoS didn't change it), not a fundamental limit. The platform's real-time thread policy removes it.
Don't busy-wait on a time-constraint thread: the kernel throttles a thread that overruns its declared 1 ms
computation budget, which is where the 84 ms stall came from.

## Simulator fidelity (measured)

The encoding model reproduces first-order statistics: firing rate 3.24 vs 3.38 Hz real, 20 ms Fano factor 1.04 vs
1.01, and 0.134 bits/spike of held-out encoding information. Every real-trained decoder is also scored on spikes
generated from the **same real test velocities** (`encoder_fidelity.json`):

| | real test R² | simulated-spike R² |
|---|---|---|
| KF | 0.216 | 0.284 |
| KF-lag | 0.553 | 0.591 |
| Wiener | 0.654 | 0.504 |
| GRU-16 … GRU-256 | 0.845 – 0.881 | 0.497 – 0.553 |
| Wiener-lag0 / GRU-64-lag0 | 0.720 / 0.892 | 0.625 / 0.734 |

The simulator is easier than reality for the Kalman filters and harder for history and recurrent decoders. Control A
(section 2) quantifies how much closed-loop BPS that costs them.

## Related work, and what this repo adds

That offline accuracy can mislead about closed-loop control is **not new**, and this repo doesn't claim it is:

- **Offline vs closed-loop disagreement.** Chase, Schwartz & Kass (2009, *Neural Networks*) showed that open-loop
  simulation misranks decoders, because closed-loop users compensate for bias. Koyama et al. (2010, *J Comput
  Neurosci*) compared decoders offline and online, with the same conclusion.
- **Closed-loop simulators with a model of the user.** Cunningham et al. (2011, *J Neurophysiol*) built an online
  prosthetic simulator: real people control a cursor through synthetic neural activity. Willett et al. (2017,
  *J Neural Eng*; 2019, *Sci Rep*) modelled the user as a feedback controller and used that model to design and tune
  decoders. The simulated user here is a much cruder member of that family.
- **ReFIT.** Gilja et al. (2012, *Nat Neurosci*) introduced intention-rotated retraining and the velocity-state
  Kalman filter used here. Wu et al. (2006, *Neural Comput*) introduced the Kalman decoder.
- **RNN decoders in closed loop.** Sussillo et al. (2012, *J Neural Eng*) ran RNN decoders closed loop. Sussillo et
  al. (2016, *Nat Commun*) showed they need perturbation augmentation and broad training data to stay robust, the same
  failure family as the distribution shift in section 2.
- **The metric and data.** Achieved bitrate on a grid task: Nuyujukian et al. (2015, *IEEE TBME*) and Pandarinath et
  al. (2017, *eLife*). MC_Maze and MC_RTT: Pei et al. (2021, Neural Latents Benchmark).

**What this repo adds is engineering and diagnostics, not a new scientific claim:**
- one causal decoder code path from offline R² through closed-loop BPS to a C++ loop, with parity tests;
- a simulator whose fidelity is measured against the real neurons, and reported where it fails;
- controls that separate simulator mismatch from distribution shift (decoders trained on simulated spikes; MC_RTT);
- user-model and ReFIT-label ablations with CIs;
- every number regenerated from committed JSON.

## What the simulator can't capture

- **Real population dynamics.** Neurons are velocity-tuned Poisson units (a multi-lag GLM) with low-rank AR(1)
  noise. There are no rotational or preparatory dynamics, no position tuning, no condition-specific structure. This
  is the main bias (fidelity table, section 2).
- **Learning and adaptation.** The user adapts to one thing, the cursor's gain. Bias, lag and direction errors are
  not learned: no re-aiming, no co-adaptation. Section 4 shows the ranking is sensitive to exactly this.
- **A human motor system.** The MC_Maze user is fit to monkey hand reaches (v_max 1000 mm/s), with fixed 100 ms
  visual and 200 ms reaction delays. The human-speed row in section 4 is a stand-in, not a model of a person using a
  BCI.
- **Intended vs attempted movement.** MC_Maze and MC_RTT are able-bodied reaching. Click decoding is a fixed
  miss/false-click process, not decoded from neurons.
- **Nonstationarity beyond a random walk.** There is no unit loss or gain, no day-to-day change, no correlated array
  shifts.
- **The real Webgrid.** Grid size, workspace and click parameters are plausible choices, not Neuralink's or
  BrainGate's settings. **Absolute BPS here is not comparable to published human numbers.**
- **Real-time I/O.** Latency covers decoder compute on prepared bin counts, not spike detection, binning, radio or
  display.

## Next steps

1. **Close the fidelity gap:** replace the GLM with a latent dynamical model (LFADS/SLDS-style) fit to MC_Maze, so
   simulated-spike R² matches real R² for every decoder. Then re-run Control A.
2. **Train RNNs for closed loop:** continuous-reaching data (MC_RTT-style), ReFIT-style labels with better speed
   handling (section 5), and perturbation augmentation.
3. **A better user model:** adaptation to bias and lag, not just gain, following Willett et al.'s feedback-control
   user.
4. **Lag as a closed-loop hyperparameter** (section 3), tuned on calibration sessions like the gain.
5. **Cross-session evaluation:** connect to the sibling FALCON stability project for multi-day decoders.

## Pipeline

```
MC_Maze (DANDI 000128, Jenkins, 182 units, 2,295 reaches)          MC_RTT (DANDI 000129, Indy, 130 units, 10.8 min)
  │  20 ms bins · chronological 70/15/15 split · decode lag 100 ms (chosen on val)
  ├─ offline decoders ─ KF · KF-lag · Wiener (≤40-bin history) · GRU-{16…256} · ablations (cont, lag 0)
  │                     R² on held-out test trials (scored window and NLB window), bootstrap CIs
  ├─ encoding model ── Poisson GLM, multi-lag velocity tuning (0–160 ms leads)
  │                     + shared AR(1) latent noise (factor analysis) + private noise + optional drift
  ├─ simulated user ── reach policy fit to real reaches, 100 ms visual delay, 200 ms reaction time,
  │                     forward model with online-learned cursor gain, motor noise
  ├─ Webgrid ───────── click on 12×12 (primary) or 1 s dwell on 6×6; B = log2(N−1)·max(correct−incorrect, 0)/t
  ├─ ReFIT ─────────── KF closed loop → intention-rotated labels (switch bins masked) → refit KF / GRU
  ├─ controls ──────── sim-trained decoders · MC_RTT replication · oracle-label ReFIT · user-model ablation
  └─ rt/ (C++17) ───── float32 KF / KF-lag / Wiener / GRU step(), parity with Python, batched + paced latency
```

### Design choices that matter for the numbers

- **Chronological split** (first 70% of trials train, next 15% val, last 15% test), not NLB's interleaved split.
  Within-session drift can't leak between train and test.
- **Scoring window [−500, +800] ms around movement onset**, including the still period, which closed loop punishes.
  The NLB [−250, +450] ms window is reported too.
- **One causal code path.** `predict()` loops the same `step()` the simulator and the C++ port use; a test checks
  causality.
- **All selection on validation or calibration seeds.** That covers lag, Wiener history and ridge, GRU early stopping
  and per-decoder gains. Calibration seeds are disjoint from evaluation seeds.
- **Common random numbers.** Each evaluation seed fixes the target sequence and every noise stream, and all decoders
  see the same seeds, so paired differences are tight.

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock
```

```bash
./run_all.sh
```

`run_all.sh` runs scripts `01`–`11` (about 60 minutes on an M-series MacBook, CPU only). It downloads MC_Maze
(691 MB) and MC_RTT (50 MB), both sha256-checked. `requirements.lock` pins exact versions; `requirements.txt` lists
the loose ones. `artifacts/` (the trained decoders and encoders) is not in the repo. It will be attached to
the v1.0.0 GitHub Release as `closed-loop-bps-artifacts-v1.0.0.tar.gz`, built by `scripts/make_release_asset.sh v1.0.0`.
Once it is published, unpack it at the repo root to restore `artifacts/` and re-run the closed-loop steps without
retraining. Until then, `./run_all.sh` regenerates `artifacts/`.

```bash
.venv/bin/python -m pytest -q tests
```

```bash
make -C rt && rt/build/bench rt/models/kf.bin rt/models/kf_ref.bin --steps 100000 --paced-seconds 60 --rt
```

The 25 tests run on synthetic data, so CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), pinned to the
lockfile) needs no download. They cover:
- metrics: the bitrate formula;
- decoders: KF steady state vs time-varying, KF-lag, streaming vs batch, causality, numpy GRU vs torch;
- simulator: GLM recovery, sampler mean rate, drift scale, Webgrid sanity, reproducibility, ReFIT labels and switch
  mask;
- C++: parity with Python for all decoder types, and rejection of truncated or padded weight files.

`make -C rt NATIVE=1` builds with `-mcpu=native` / `-march=native`.

## Layout

| Path | What |
|---|---|
| `cbps/data.py`, `cbps/data_rtt.py` | NWB → 20 ms bins, trial windows / continuous windows, chronological splits |
| `cbps/decoders.py` | KF (velocity or lag-augmented state, steady-state), Wiener, GRU (numpy inference) |
| `cbps/train_gru.py` | GRU training (trial windows, or continuous-stream crops + augmentation) |
| `cbps/encoding.py` | Generative spike model used by the simulator |
| `cbps/webgrid.py` | Task, simulated user (adaptive forward model), session runner, oracle, reach-policy fit |
| `cbps/refit.py` | ReFIT relabelling (masked / oracle), ReFIT-KF and ReFIT-GRU |
| `cbps/experiment.py` | Parallel sessions, gain tuning, paired bootstrap summaries |
| `cbps/export.py`, `rt/` | Binary weight format, C++ decoders, latency benchmark |
| `scripts/01…11` | Pipeline steps, each writing `results/*.json`. Figures: `06` (headline, R² vs BPS, probe, drift), `08` (`trajectories.png`, `dwell_feasibility.png`) |
