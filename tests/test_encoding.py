import numpy as np

from cbps.encoding import ENC_LAGS, fit_encoding_model, fit_poisson_glm


def test_glm_recovers_weights():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((20000, 3)) * 0.5
    Btrue = np.array([[np.log(0.3), 0.5, -0.4, 0.2], [np.log(1.0), -0.2, 0.3, 0.0]])
    lam = np.exp(Btrue[:, 0] + F @ Btrue[:, 1:].T)
    B = fit_poisson_glm(F, rng.poisson(lam).astype(float), l2=0.0)
    assert np.allclose(B, Btrue, atol=0.05)


def test_sampler_keeps_glm_mean_rate(toy_encoder):
    """The -var/2 correction keeps E[rate] at the GLM rate even with latent noise."""
    rng = np.random.default_rng(0)
    s = toy_encoder.sampler(rng, noise_scale=1.0)
    still = np.zeros((len(ENC_LAGS), 2))
    base = toy_encoder.log_rate(still)
    counts = np.mean([s.sample(base) for _ in range(20000)], 0)
    assert np.allclose(counts, np.exp(base), rtol=0.1)


def test_drift_moves_rates(toy_encoder):
    rng = np.random.default_rng(0)
    s = toy_encoder.sampler(rng, drift_per_10min=1.0)
    for _ in range(30000):  # 10 minutes of bins
        s.step(np.zeros(2))
    assert 0.6 < s.d.std() < 1.4


def test_fit_encoding_model_runs(toy_data):
    X, _, Vs = toy_data
    enc = fit_encoding_model(X, Vs, n_factors=2)
    assert enc.B.shape == (X.shape[-1], 1 + 3 * len(ENC_LAGS))
    assert np.all(np.isfinite(enc.C)) and np.all(enc.log_var >= 0)
