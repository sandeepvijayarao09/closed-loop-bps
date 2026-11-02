"""Small helpers for result files and decoder artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    return p


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def save_pickle(name: str, obj) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    p = ARTIFACTS / name
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def load_pickle(name: str):
    with open(ARTIFACTS / name, "rb") as f:
        return pickle.load(f)
