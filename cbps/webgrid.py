"""Closed-loop Webgrid simulator.

Each 20 ms tick:
  1. The simulated user sees the cursor `vis_delay` bins late (and a new target
     `react_delay` bins late) and issues a velocity command toward the target
     with a policy fit to real MC_Maze straight reaches (scripts/03_fit_encoder.py).
  2. The encoding model turns the command history into spike counts (multi-lag
     tuning, shared latent noise, private noise, optional drift).
  3. The decoder sees only those counts and outputs a velocity; the cursor moves
     by gain * v_hat * dt.
  4. Selection: dwell (stay in one cell for `dwell_s`) or click (user tries to
     click once they think they are on target and slow; the click decoder has
     latency, misses and false positives).

Because the user only ever reacts to where the decoded cursor went, decoder
errors compound: a lag forces corrections, bias must be fought continuously,
jitter breaks dwell. Offline R^2 measures none of that directly.

Throughput is the Webgrid / BrainGate achieved bitrate,
  B = log2(N - 1) * max(correct - incorrect, 0) / t,   N = grid cells.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import BIN_S
from .data import DECODE_LAG
from .decoders import Decoder
from .encoding import CMD_LEAD, EncodingModel
from .metrics import achieved_bitrate


@dataclass
class UserModel:
    """Command = vmax * tanh(d / (vmax * tau)) toward the target, low-passed with tau_m.

    Defaults are the least-squares fit to mean speed profiles of real straight
    reaches (train split); results/user_model.json holds the fit.
    """
    vmax: float = 1000.0    # mm/s
    tau: float = 0.08       # s; sets deceleration distance
    tau_m: float = 0.08     # s; motor smoothing
    vis_delay: int = 5      # bins (100 ms) cursor feedback delay
    react_delay: int = 10   # bins (200 ms) to react to a new target
    dir_noise: float = 0.15  # std of multiplicative 2-D command noise
    click_speed: float = 80.0  # mm/s perceived speed under which the user tries to click
    # Internal forward model (efference copy): the user estimates where the
    # cursor is now as the delayed view plus the commands still "in flight",
    # assuming the cursor follows intent. Without it, the reach policy (fit to
    # hand reaches that need no visual feedback) oscillates under 160 ms of
    # loop delay even with a perfect decoder. Decoder errors still compound,
    # because the estimate is re-anchored on the true (delayed) cursor.
    forward_model: bool = True
    # "adaptive": the forward model's cursor gain k is learned online (recursive
    # least squares of seen cursor displacement on the command that produced it,
    # ~1 s memory), as people adapt to a cursor's gain within seconds.
    # "fixed": k = 1, i.e. the user assumes the cursor moves exactly as intended.
    # With k = 1 a decoder that under-delivers speed (or whose gain decays during
    # a sustained push) makes the user believe the cursor has arrived and stop
    # pushing; results/diagnostics.json shows that this alone reorders decoders.
    forward_gain: str = "adaptive"
    gain_memory: float = 0.98   # RLS forgetting factor per bin


@dataclass
class TaskConfig:
    grid: int = 12            # grid x grid targets
    width_mm: float = 240.0   # screen side, in hand-workspace mm (MC_Maze reaches ~130 mm)
    mode: str = "dwell"       # "dwell" | "click"
    dwell_s: float = 0.5
    click_latency: int = 2    # bins from intent to registered click
    click_miss: float = 0.05  # P(click attempt is missed)
    false_click_hz: float = 0.02  # spurious clicks per second (select whatever is under the cursor)
    click_refractory: int = 15  # bins between click attempts
    seconds: float = 120.0
    noise_scale: float = 1.0
    drift_per_10min: float = 0.0  # std of per-neuron log-rate random walk after 10 min
    gain: float = 1.0          # decoder output gain (tuned per decoder on calibration seeds)

    @property
    def n_targets(self):
        return self.grid * self.grid

    @property
    def cell(self):
        return self.width_mm / self.grid


@dataclass
class SessionResult:
    correct: int
    incorrect: int
    seconds: float
    bps: float
    mean_time_to_target_s: float
    path_efficiency: float

    def as_dict(self):
        return asdict(self)


class Oracle(Decoder):
    name = "Oracle"
    is_oracle = True
    n_params = 0

    def step(self, y):  # never called; run_session short-circuits
        raise RuntimeError


def cell_of(p, cfg: TaskConfig):
    ij = np.floor((p + cfg.width_mm / 2) / cfg.cell).astype(int)
    return tuple(np.clip(ij, 0, cfg.grid - 1))


def center_of(ij, cfg: TaskConfig):
    return (np.array(ij) + 0.5) * cfg.cell - cfg.width_mm / 2


def run_session(decoder: Decoder, enc: EncodingModel, cfg: TaskConfig, user: UserModel, seed: int,
                record: bool = False):
    """One closed-loop session. Returns (SessionResult, log or None).

    Seeds are shared across decoders (common random numbers): the same seed
    gives the same target sequence and the same noise streams, so decoder
    differences are not swamped by session-to-session luck.
    """
    rng_task = np.random.default_rng([seed, 1])
    rng_user = np.random.default_rng([seed, 2])
    rng_click = np.random.default_rng([seed, 3])
    sampler = enc.sampler(np.random.default_rng([seed, 4]), cfg.noise_scale, cfg.drift_per_10min)
    decoder.reset()
    n_steps = int(round(cfg.seconds / BIN_S))
    hist = max(user.vis_delay, user.react_delay) + 1

    def new_target(exclude):
        while True:
            ij = (int(rng_task.integers(cfg.grid)), int(rng_task.integers(cfg.grid)))
            if ij != exclude:
                return ij

    p = np.zeros(2)
    c = np.zeros(2)
    target = new_target(cell_of(p, cfg))
    in_flight = user.vis_delay + (CMD_LEAD - DECODE_LAG)
    k_hat = 1.0
    k_num = k_den = 400.0  # prior worth ~0.5 s of 200 mm/s movement at k = 1
    c_hist = np.zeros((max(in_flight, 1), 2))  # c_hist[k] = user command k+1 bins ago
    p_hist = np.zeros((hist, 2))            # p_hist[k] = cursor k bins ago
    v_hist = np.zeros((hist, 2))
    t_hist = [target] * hist                 # target k bins ago
    correct = incorrect = 0
    dwell_cell, dwell_t = None, 0.0
    pending_click, refractory = -1, 0
    t_target_start, times, start_pos, path_len, effs = 0, [], p.copy(), 0.0, []
    log = {"p": [], "target": [], "cmd": [], "vhat": [], "y": [], "k_hat": [], "events": []} if record else None

    def select(ij, t):
        nonlocal correct, incorrect, target, t_target_start, start_pos, path_len, dwell_cell, dwell_t
        hit = ij == target
        if record:
            log["events"].append((t, hit, ij))
        if hit:
            correct += 1
            times.append((t - t_target_start) * BIN_S)
            straight = np.linalg.norm(center_of(target, cfg) - start_pos)
            if path_len > 0:
                effs.append(min(1.0, straight / path_len))
            target = new_target(target)
            t_target_start, start_pos, path_len = t, p.copy(), 0.0
        else:
            incorrect += 1
        dwell_cell, dwell_t = None, 0.0

    for t in range(n_steps):
        # --- user: act on delayed percepts ---
        seen_p = p_hist[user.vis_delay - 1] if user.vis_delay > 0 else p
        seen_target = t_hist[user.react_delay - 1] if user.react_delay > 0 else target
        if user.forward_model and user.forward_gain == "adaptive" and user.vis_delay > 0:
            # The cursor step seen last was driven by the command issued in_flight bins ago.
            u = BIN_S * c_hist[in_flight - 1]
            dp = p_hist[user.vis_delay - 1] - p_hist[user.vis_delay]
            k_num = user.gain_memory * k_num + dp @ u
            k_den = user.gain_memory * k_den + u @ u
            k_hat = float(np.clip(k_num / k_den, 0.05, 3.0))
        # Commands not yet visible in the delayed cursor: the last in_flight - 1.
        est_p = seen_p + k_hat * BIN_S * c_hist[:in_flight - 1].sum(0) if user.forward_model else seen_p
        e = center_of(seen_target, cfg) - est_p
        d = np.linalg.norm(e)
        cmd = user.vmax * np.tanh(d / (user.vmax * user.tau)) * e / max(d, 1e-9)
        cmd = cmd * (1 + user.dir_noise * rng_user.standard_normal()) \
            + user.dir_noise * np.linalg.norm(cmd) * rng_user.standard_normal(2) * 0.5
        c = c + (cmd - c) * BIN_S / user.tau_m
        c_hist = np.roll(c_hist, 1, 0)
        c_hist[0] = c

        # --- brain -> decoder -> cursor ---
        y = sampler.step(c)
        if getattr(decoder, "is_oracle", False):
            # Ceiling: decoded velocity == the user's command, with the same
            # 60 ms neural-to-decoder latency real decoders have (see encoding.py).
            vhat = cfg.gain * sampler.cmd[CMD_LEAD - DECODE_LAG].copy()
        else:
            vhat = cfg.gain * decoder.step(y)
        p_new = np.clip(p + vhat * BIN_S, -cfg.width_mm / 2 + 1e-6, cfg.width_mm / 2 - 1e-6)
        path_len += np.linalg.norm(p_new - p)
        p = p_new
        p_hist = np.roll(p_hist, 1, 0)
        p_hist[0] = p
        v_hist = np.roll(v_hist, 1, 0)
        v_hist[0] = vhat
        t_hist = [target] + t_hist[:-1]

        # --- selection ---
        here = cell_of(p, cfg)
        if cfg.mode == "dwell":
            if here == dwell_cell:
                dwell_t += BIN_S
            else:
                dwell_cell, dwell_t = here, 0.0
            if dwell_t >= cfg.dwell_s:
                select(here, t)
        else:
            refractory = max(refractory - 1, 0)
            seen_v = v_hist[user.vis_delay - 1] if user.vis_delay > 0 else vhat
            on_target = cell_of(seen_p, cfg) == seen_target == target
            if pending_click < 0 and refractory == 0 and on_target and np.linalg.norm(seen_v) < user.click_speed:
                pending_click, refractory = cfg.click_latency, cfg.click_refractory
            if pending_click == 0:
                if rng_click.random() >= cfg.click_miss:
                    select(here, t)
                pending_click = -1
            elif pending_click > 0:
                pending_click -= 1
            if rng_click.random() < cfg.false_click_hz * BIN_S:
                select(here, t)
        if record:
            log["p"].append(p.copy())
            log["target"].append(center_of(target, cfg))
            log["cmd"].append(c.copy())
            log["vhat"].append(vhat.copy())
            log["y"].append(y)
            log["k_hat"].append(k_hat)

    res = SessionResult(correct, incorrect, cfg.seconds,
                        achieved_bitrate(cfg.n_targets, correct, incorrect, cfg.seconds),
                        float(np.mean(times)) if times else float("nan"),
                        float(np.mean(effs)) if effs else float("nan"))
    if record:
        log = {k: (np.array(v) if k != "events" else v) for k, v in log.items()}
    return res, log
