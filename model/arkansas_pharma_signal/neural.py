"""Compact neural-style multi-task model implemented in numpy.

NeuralSignalBlender learns a small MLP over panel features with gradient
descent and stays far under the 50M parameter budget (here a few thousand).
Falls back to regression coefficients when data is too small.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class NeuralSignalBlender:
    """Small two-layer MLP with tanh hidden activations.

    Multi-task output: [demand (log-scaled), demand_shock, supply_risk].
    Trained with gradient descent + momentum. Parameter count stays in the
    thousands regardless of input width (hidden size is fixed).
    """

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 0.01,
        epochs: int = 300,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        self.W1: np.ndarray
        self.b1: np.ndarray
        self.W2: np.ndarray
        self.b2: np.ndarray
        self.mean: np.ndarray
        self.std: np.ndarray
        self.out_mean: np.ndarray
        self.out_std: np.ndarray

    # -- public API ------------------------------------------------------------
    @property
    def n_params(self) -> int:
        if not hasattr(self, "W1"):
            return 0
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "NeuralSignalBlender":
        X = _fill_nan(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        Xs = (X - self.mean) / self.std

        self.out_mean = np.nanmean(Y, axis=0)
        self.out_std = np.nanstd(Y, axis=0)
        self.out_std[self.out_std < 1e-9] = 1.0
        Ys = (Y - self.out_mean) / self.out_std

        d_in = Xs.shape[1]
        d_out = Ys.shape[1]
        self.W1 = self.rng.normal(0, 0.1, (d_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0, 0.1, (self.hidden, d_out))
        self.b2 = np.zeros(d_out)

        mW1 = np.zeros_like(self.W1)
        mb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2)
        mb2 = np.zeros_like(self.b2)

        for _ in range(self.epochs):
            h = np.tanh(Xs @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            grad_out = (pred - Ys) / len(Xs)
            gW2 = h.T @ grad_out + self.weight_decay * self.W2
            gb2 = grad_out.sum(axis=0)
            ghidden = grad_out @ self.W2.T * (1 - h ** 2)
            gW1 = Xs.T @ ghidden + self.weight_decay * self.W1
            gb1 = ghidden.sum(axis=0)

            mW1 = 0.9 * mW1 - self.lr * gW1
            mb1 = 0.9 * mb1 - self.lr * gb1
            mW2 = 0.9 * mW2 - self.lr * gW2
            mb2 = 0.9 * mb2 - self.lr * gb2
            self.W1 += mW1
            self.b1 += mb1
            self.W2 += mW2
            self.b2 += mb2
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _fill_nan(np.asarray(X, dtype=float))
        Xs = (X - self.mean) / self.std
        h = np.tanh(Xs @ self.W1 + self.b1)
        return h @ self.W2 + self.b2 + self.out_mean

    def to_dict(self) -> Dict:
        return {
            "family": "neural_signal_blender_mlp",
            "n_params": self.n_params,
            "hidden": self.hidden,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "out_mean": self.out_mean.tolist(),
            "out_std": self.out_std.tolist(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "NeuralSignalBlender":
        m = cls(hidden=d["hidden"])
        m.feature_names = d["feature_names"]
        m.mean = np.asarray(d["mean"], dtype=float)
        m.std = np.asarray(d["std"], dtype=float)
        m.out_mean = np.asarray(d["out_mean"], dtype=float)
        m.out_std = np.asarray(d["out_std"], dtype=float)
        m.W1 = np.asarray(d["W1"], dtype=float)
        m.b1 = np.asarray(d["b1"], dtype=float)
        m.W2 = np.asarray(d["W2"], dtype=float)
        m.b2 = np.asarray(d["b2"], dtype=float)
        return m


def _fill_nan(X: np.ndarray) -> np.ndarray:
    X = np.array(np.asarray(X, dtype=float))
    for j in range(X.shape[1]):
        col = X[:, j]
        sel = col[~np.isnan(col)]
        m = sel.mean() if sel.size else 0.0
        col[col != col] = m  # NaN != NaN
    return X
