import numpy as np
import pytest

from cbps.metrics import achieved_bitrate, bootstrap_ci, r2


def test_bitrate_formula_uses_n_minus_1():
    # 144 targets, 30 correct, 0 wrong, 60 s -> log2(143) * 30 / 60
    assert achieved_bitrate(144, 30, 0, 60.0) == pytest.approx(np.log2(143) * 0.5)


def test_bitrate_wrong_selections_cancel_and_floor_at_zero():
    assert achieved_bitrate(144, 30, 10, 60.0) == pytest.approx(np.log2(143) * 20 / 60)
    assert achieved_bitrate(144, 5, 9, 60.0) == 0.0


def test_bitrate_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        achieved_bitrate(1, 1, 0, 1.0)
    with pytest.raises(ValueError):
        achieved_bitrate(4, 1, 0, 0.0)


def test_r2_perfect_and_mean_prediction():
    y = np.random.default_rng(0).standard_normal((100, 2))
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.broadcast_to(y.mean(0), y.shape)) == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci(np.arange(20.0))
    assert lo < m < hi
