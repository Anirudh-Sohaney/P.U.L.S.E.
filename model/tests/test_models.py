"""Model and signal unit tests (plain asserts)."""

import numpy as np

from arkansas_pharma_signal.neural import NeuralSignalBlender
from arkansas_pharma_signal.regression import LogisticRidge, RidgeLinear


def test_ridge_fit_predict():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3.0 * x[:, 0] + 5.0
    m = RidgeLinear(alpha=1e-6).fit(x, y, ["f"])
    pred = m.predict(x)
    assert np.abs(pred - y).max() < 1e-4


def test_ridge_serialization_roundtrip():
    x = np.random.default_rng(1).normal(size=(50, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 1.0
    m = RidgeLinear().fit(x, y, ["a", "b", "c"])
    m2 = RidgeLinear.from_dict(m.to_dict())
    assert np.allclose(m.predict(x), m2.predict(x))


def test_logistic_separable():
    rng = np.random.default_rng(2)
    x0 = rng.normal(-2, 1, size=(30, 2))
    x1 = rng.normal(2, 1, size=(30, 2))
    X = np.vstack([x0, x1])
    y = np.r_[np.zeros(30), np.ones(30)]
    m = LogisticRidge(alpha=1e-4).fit(X, y, ["a", "b"])
    assert m.predict_proba(X)[y == 1].mean() > 0.9
    assert m.predict_proba(X)[y == 0].mean() < 0.1


def test_neural_param_budget():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 6))
    Y = (X @ np.ones(6)).reshape(-1, 1)
    m = NeuralSignalBlender(epochs=200).fit(X, Y, [f"f{i}" for i in range(6)])
    assert m.n_params < 50_000_000
    assert m.n_params > 0
    assert m.predict(X).shape == (80, 1)
    assert np.isfinite(m.predict(X)).all()


if __name__ == "__main__":
    test_ridge_fit_predict()
    test_ridge_serialization_roundtrip()
    test_logistic_separable()
    test_neural_param_budget()
    print("ok")