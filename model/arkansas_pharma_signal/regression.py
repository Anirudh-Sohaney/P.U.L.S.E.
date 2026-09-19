"""Interpretable regression baselines implemented with numpy only.

- RidgeLinear: closed-form ridge regression on standardized features.
- LogisticRidge: ridge-regularized logistic regression via Newton-Raphson.

Models persist coefficients and statistics to JSON for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class RidgeLinear:
    """Standardized ridge regression solved by normal equations."""

    alpha: float = 1.0
    feature_names: List[str] = field(default_factory=list)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    residual_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
            sample_weight: Optional[np.ndarray] = None) -> "RidgeLinear":
        """Fit standardized ridge regression with optional row weights."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_names = list(feature_names)
        if sample_weight is None:
            mask = np.isfinite(y)
            X, y = X[mask], y[mask]
            weights = np.ones(len(y))
        else:
            weights = np.asarray(sample_weight, dtype=float)
            mask = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
            X, y, weights = X[mask], y[mask], weights[mask]

        self.mean = (X * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-9)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        Xs = (X - self.mean) / self.std
        b = float((weights * y).sum() / max(weights.sum(), 1e-9))
        sw = np.sqrt(weights)
        Xt = Xs * sw[:, None]
        yt = (y - b) * sw
        XtX = Xt.T @ Xt + self.alpha * np.eye(Xs.shape[1])
        Xty = Xt.T @ yt
        self.coef = np.linalg.solve(XtX, Xty)
        self.intercept = b  # Xs is weight-center on zero
        self.residual_std = float(np.std(y - self.predict(X)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the continuous response for standardized feature rows."""
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean) / self.std
        return Xs @ self.coef + self.intercept

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Signed feature contributions (feature * scaled coef) per row."""
        X = np.asarray(X, dtype=float)
        return ((X - self.mean) / self.std) * self.coef

    def to_dict(self) -> Dict:
        """Serialize fitted coefficients and preprocessing statistics."""
        return {
            "family": "ridge_linear",
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "residual_std": self.residual_std,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RidgeLinear":
        """Restore a fitted ridge model from ``to_dict`` output."""
        m = cls(alpha=d.get("alpha", 1.0))
        m.feature_names = d["feature_names"]
        m.mean = np.asarray(d["mean"], dtype=float)
        m.std = np.asarray(d["std"], dtype=float)
        m.coef = np.asarray(d["coef"], dtype=float)
        m.intercept = float(d["intercept"])
        m.residual_std = float(d.get("residual_std", 1.0))
        return m


@dataclass
class LogisticRidge:
    """Ridge-regularized logistic regression (Newton-Raphson / IRLS)."""

    alpha: float = 1.0
    max_iter: int = 50
    tol: float = 1e-6
    feature_names: List[str] = field(default_factory=list)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
            sample_weight: Optional[np.ndarray] = None) -> "LogisticRidge":
        """Fit IRLS logistic regression with ridge regularization."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if sample_weight is None:
            weights = np.ones(len(y), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(weights) != len(y):
                raise ValueError("sample_weight must match y length")
        mask = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        X = X[mask]
        y = y[mask]
        weights = weights[mask]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_names = list(feature_names)
        self.mean = (X * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-9)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        Xs = (X - self.mean) / self.std
        Xs = np.column_stack([np.ones(len(Xs)), Xs])
        w = np.zeros(Xs.shape[1])
        for _ in range(self.max_iter):
            p = _logistic(Xs @ w)
            W = weights * p * (1 - p)
            grad = Xs.T @ ((p - y) * weights)
            hess = Xs.T @ (Xs * W[:, None]) + self.alpha * np.eye(Xs.shape[1])
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.intercept = float(w[0])
        self.coef = w[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for feature rows."""
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean) / self.std
        return _logistic(Xs @ self.coef + self.intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions using the fixed 0.5 operating point."""
        return (self.predict_proba(X) >= 0.5).astype(float)

    def to_dict(self) -> Dict:
        """Serialize fitted coefficients and preprocessing statistics."""
        return {
            "family": "logistic_ridge",
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LogisticRidge":
        """Restore a fitted logistic model from ``to_dict`` output."""
        m = cls(alpha=d.get("alpha", 1.0))
        m.feature_names = d["feature_names"]
        m.mean = np.asarray(d["mean"], dtype=float)
        m.std = np.asarray(d["std"], dtype=float)
        m.coef = np.asarray(d["coef"], dtype=float)
        m.intercept = float(d["intercept"])
        return m


def _fill_nan(X: np.ndarray) -> np.ndarray:
    """Replace NaN with per-column mean (computed over observed values)."""
    X = np.array(np.asarray(X, dtype=float))
    for j in range(X.shape[1]):
        col = X[:, j]
        sel = col[~np.isnan(col)]
        m = sel.mean() if sel.size else 0.0
        col[col != col] = m  # NaN != NaN
    return X
