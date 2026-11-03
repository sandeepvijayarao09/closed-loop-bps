import numpy as np
import pytest

from cbps.decoders import Decoder, KalmanVelocity
from cbps.webgrid import Oracle, TaskConfig, UserModel, cell_of, center_of, run_session


class Zero(Decoder):
    name = "zero"
    n_params = 0

    def step(self, y):
        return np.zeros(2)


def test_cell_center_roundtrip():
    cfg = TaskConfig(grid=12, width_mm=240)
    for ij in [(0, 0), (5, 7), (11, 11)]:
        assert cell_of(center_of(ij, cfg), cfg) == ij


@pytest.mark.parametrize("mode", ["dwell", "click"])
def test_oracle_scores_and_frozen_cursor_does_not(toy_encoder, mode):
    cfg = TaskConfig(mode=mode, seconds=30, false_click_hz=0.0)
    good, _ = run_session(Oracle(), toy_encoder, cfg, UserModel(), seed=0)
    assert good.correct > 5 and good.bps > 1.0
    bad, _ = run_session(Zero(), toy_encoder, cfg, UserModel(), seed=0)
    assert bad.correct == 0 and bad.bps == 0.0


def test_sessions_are_reproducible(toy_encoder, toy_data):
    X, V, _ = toy_data
    kf = KalmanVelocity.fit(X[:40], V[:40])
    cfg = TaskConfig(seconds=10)
    a, _ = run_session(kf.clone(), toy_encoder, cfg, UserModel(), seed=3)
    b, _ = run_session(kf.clone(), toy_encoder, cfg, UserModel(), seed=3)
    assert a == b


def test_refit_labels_point_at_target_and_zero_on_target():
    from cbps.refit import refit_labels
    cfg = TaskConfig()
    p = np.array([[0.0, 0.0], [100.0, 100.0]])
    tgt = np.array([[50.0, 0.0], [100.0, 100.0]])
    v = np.array([[0.0, 30.0], [10.0, 0.0]])
    lab = refit_labels(p, tgt, v, cfg)
    assert np.allclose(lab[0], [30.0, 0.0])
    assert np.allclose(lab[1], 0.0)


def test_switch_mask_blanks_bins_after_target_change():
    from cbps.refit import switch_mask
    tgt = np.zeros((30, 2))
    tgt[10:] = [5.0, 5.0]
    tgt[25:] = [1.0, 1.0]
    m = switch_mask(tgt, 4)
    assert m[:10].all() and not m[10:14].any() and m[14:25].all() and not m[25:29].any() and m[29]
