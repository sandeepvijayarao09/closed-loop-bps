"""Leakage/reset check: decode the real test block as one continuous stream vs per-trial resets.

A decoder that only works when reset at trial start (or that drifts over minutes)
would lose R^2 here. Writes results/continuous_check.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbps import data  # noqa: E402
from cbps.io import load_pickle, save_json  # noqa: E402
from cbps.metrics import r2  # noqa: E402


def main():
    decs = load_pickle("decoders_cl.pkl")
    d = np.load(data.PROCESSED)
    on, L, E = d["onset_bin"], data.DECODE_LAG, data.EVAL_START
    _, _, te = data.split(data.load(L))
    ti = te.trial_idx
    s0 = int(on[ti[0]]) - data.PRE_BINS
    s1 = int(on[ti[-1]]) + data.POST_BINS
    X = d["spikes"][s0:s1].astype(np.float32)
    idx = np.concatenate([np.arange(on[i] - data.PRE_BINS + E, on[i] + data.POST_BINS) for i in ti]) - s0
    Vt = d["vel"][idx + s0 + L]
    out = {"stream_minutes": len(X) * 0.02 / 60, "decoders": {}}
    for n, dec in decs.items():
        per_trial = r2(te.V[:, E:], dec.predict(te.X)[:, E:])
        cont = r2(Vt, dec.predict(X[None])[0][idx])
        out["decoders"][n] = {"per_trial_reset_r2": per_trial, "continuous_r2": cont, "delta": cont - per_trial}
        print(f"{n:14s} reset {per_trial:.3f}  continuous {cont:.3f}")
    save_json("continuous_check.json", out)


if __name__ == "__main__":
    main()
