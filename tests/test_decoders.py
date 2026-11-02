import numpy as np
import pytest
import torch

from cbps.decoders import GRUDecoder, KalmanVelocity, Wiener
from cbps.metrics import r2
from cbps.train_gru import GRUNet


def _stream(dec, X):
    dec.reset()
    return np.stack([dec.step(x) for x in X])


def test_kf_learns_and_streaming_matches_batch(toy_data):
    X, V, _ = toy_data
    kf = KalmanVelocity.fit(X[:40], V[:40])
    p = kf.predict(X[40:])
    assert r2(V[40:, 20:], p[:, 20:]) > 0.3
    assert np.allclose(_stream(kf, X[40]), p[0])


def test_steady_state_kf_matches_time_varying_after_burn_in(toy_data):
    X, V, _ = toy_data
    kf = KalmanVelocity.fit(X[:40], V[:40])
    A, W, H, Q = kf.A, kf.W, kf.H, kf.Q
    x, P, out = np.zeros(2), W.copy(), []
    for y in X[40] - kf.mu:
        x, P = A @ x, A @ P @ A.T + W
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Q)
        x, P = x + K @ (y - H @ x), (np.eye(2) - K @ H) @ P
        out.append(x)
    ss = _stream(kf, X[40])
    assert np.allclose(np.array(out)[50:], ss[50:], rtol=1e-3, atol=1e-2)


def test_wiener_streaming_matches_batch(toy_data):
    X, V, _ = toy_data
    w = Wiener.fit(X[:40], V[:40], n_hist=6, alpha=1e-3)
    assert np.allclose(_stream(w, X[41]), w.predict(X[41:42])[0])


def test_numpy_gru_matches_torch():
    torch.manual_seed(0)
    net = GRUNet(12, 8).double()
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    mu, s, vsd = np.zeros(12), np.ones(12), np.ones(2)
    dec = GRUDecoder(sd["gru.weight_ih_l0"], sd["gru.weight_hh_l0"], sd["gru.bias_ih_l0"], sd["gru.bias_hh_l0"],
                     sd["out.weight"], sd["out.bias"], mu, s, vsd)
    X = np.random.default_rng(0).poisson(1.0, (3, 30, 12)).astype(float)
    with torch.no_grad():
        ref = net(torch.tensor(X)).numpy()
    assert np.allclose(dec.predict(X), ref, atol=1e-10)
    assert np.allclose(_stream(dec, X[1]), ref[1], atol=1e-10)


def test_decoders_are_causal(toy_data):
    """Changing future bins must not change past outputs."""
    X, V, _ = toy_data
    for dec in (KalmanVelocity.fit(X[:40], V[:40]), Wiener.fit(X[:40], V[:40], 5, 1e-3)):
        a = X[45:46].copy()
        b = a.copy()
        b[0, 70:] += 5
        pa, pb = dec.predict(a), dec.predict(b)
        assert np.allclose(pa[0, :70], pb[0, :70])
        assert not np.allclose(pa[0, 70:], pb[0, 70:])


@pytest.mark.parametrize("hidden", [4])
def test_gru_training_smoke(toy_data, hidden):
    from cbps.train_gru import train_gru
    X, V, _ = toy_data
    dec, info = train_gru(X[:40], V[:40], X[40:], V[40:], hidden=hidden, epochs=3, patience=3)
    assert dec.predict(X[:2]).shape == (2, X.shape[1], 2)
    assert info["epochs_run"] <= 3
