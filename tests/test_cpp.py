"""C++ decoders must reproduce the numpy decoders bit-for-bit up to float32 rounding."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cbps.decoders import GRUDecoder, KalmanVelocity, Wiener
from cbps.export import export, export_reference
from cbps.train_gru import GRUNet

RT = Path(__file__).resolve().parents[1] / "rt"
pytestmark = pytest.mark.skipif(shutil.which("make") is None or
                                (shutil.which("c++") is None and shutil.which("clang++") is None),
                                reason="no C++ toolchain")


@pytest.fixture(scope="module")
def bench():
    subprocess.run(["make", "-C", str(RT)], check=True, capture_output=True)
    return RT / "build" / "bench"


def _gru(U, H):
    torch.manual_seed(0)
    sd = {k: v.detach().double().numpy() for k, v in GRUNet(U, H).state_dict().items()}
    return GRUDecoder(sd["gru.weight_ih_l0"], sd["gru.weight_hh_l0"], sd["gru.bias_ih_l0"], sd["gru.bias_hh_l0"],
                      sd["out.weight"], sd["out.bias"], np.full(U, 0.2), np.full(U, 0.5), np.array([200.0, 150.0]))


def test_cpp_matches_python(bench, toy_data, tmp_path):
    X, V, _ = toy_data
    decs = {"kf": KalmanVelocity.fit(X[:40], V[:40]), "wiener": Wiener.fit(X[:40], V[:40], 7, 1e-3),
            "gru": _gru(X.shape[-1], 16)}
    stream = X[40:45].reshape(-1, X.shape[-1])
    for name, d in decs.items():
        export(d, tmp_path / f"{name}.bin")
        export_reference(d, stream, tmp_path / f"{name}_ref.bin")
        r = subprocess.run([str(bench), str(tmp_path / f"{name}.bin"), str(tmp_path / f"{name}_ref.bin"),
                            "--steps", "2000"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PASS" in r.stdout
