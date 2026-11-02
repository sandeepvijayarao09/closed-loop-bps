"""Export decoders to rt/models, build the C++ loop, verify vs Python, benchmark latency.

Writes results/latency.json. Usage: python scripts/05_export_bench.py [--steps 100000] [--paced-seconds 10]
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbps import data  # noqa: E402
from cbps.export import export, export_reference  # noqa: E402
from cbps.io import load_pickle, save_json  # noqa: E402

RT = ROOT / "rt"
MODELS = RT / "models"


def python_step_latency(dec, X, n=5000):
    dec = dec.clone()
    dec.reset()
    for x in X[:200]:
        dec.step(x)
    ts = np.empty(n)
    for i in range(n):
        x = X[i % len(X)]
        a = time.perf_counter_ns()
        dec.step(x)
        ts[i] = (time.perf_counter_ns() - a) / 1e3
    return {"p50_us": float(np.percentile(ts, 50)), "p99_us": float(np.percentile(ts, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--paced-seconds", type=float, default=10.0)
    args = ap.parse_args()
    decs = load_pickle("decoders_cl.pkl")
    decs.pop("Oracle", None)
    _, _, te = data.split(data.load(data.DECODE_LAG))
    X = te.X[:30].reshape(-1, te.X.shape[-1])  # 30 real test trials as one continuous stream
    MODELS.mkdir(exist_ok=True)
    subprocess.run(["make", "-C", str(RT)], check=True, capture_output=True)
    out = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                             text=True).stdout.strip() if sys.platform == "darwin" else ""},
           "bin_ms": 20, "decoders": {}}
    for name, dec in decs.items():
        fn = name.replace("-", "_").lower()
        export(dec, MODELS / f"{fn}.bin")
        export_reference(dec, X, MODELS / f"{fn}_ref.bin")
        js = MODELS / f"{fn}_latency.json"
        r = subprocess.run([str(RT / "build" / "bench"), str(MODELS / f"{fn}.bin"), str(MODELS / f"{fn}_ref.bin"),
                            "--steps", str(args.steps), "--paced-seconds", str(args.paced_seconds), "--json", str(js), "--label", name],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        cpp = json.loads(js.read_text())
        out["decoders"][name] = {"cpp": cpp, "python_numpy": python_step_latency(dec, X)}
        if r.returncode != 0:
            raise SystemExit(f"{name}: C++ output does not match Python")
    save_json("latency.json", out)


if __name__ == "__main__":
    main()
