"""Event-focused acceptance metrics for numeric and state signals.

The event score is precision among predictions that announce an event. It is
not a replacement for raw state accuracy or numeric within-5-percent accuracy.
Thresholds must be declared before scoring and fit only on training data.
"""

from __future__ import annotations

import numpy as np


def event_route_passes(
    raw_accuracy: float | None,
    true_positive_precision: float | None,
    *,
    raw_floor: float = 0.65,
    true_positive_floor: float = 0.80,
) -> bool:
    """Return whether the declared event route meets both acceptance floors.

    The event route never replaces raw evaluation: a signal must retain at
    least the 65% raw floor while achieving at least 80% precision among its
    high-risk announcements. ``None`` is not treated as an empty perfect
    event set.
    """
    if raw_accuracy is None or true_positive_precision is None:
        return False
    return bool(float(raw_accuracy) >= raw_floor
                and float(true_positive_precision) >= true_positive_floor)


def score_event_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    kind: str,
    event_threshold: float | None = None,
    event_states: set[int] | None = None,
    relative_tolerance: float = 0.05,
) -> dict:
    """Score event announcements without changing the primary metric.

    For states, ``event_states`` identifies the high-risk states. For numeric
    targets, a prediction at or above ``event_threshold`` announces the event;
    it is a true positive when it is within the declared relative tolerance of
    the observed value. The observed value does not need to cross the
    threshold. Empty predicted-event sets are reported as undefined, not as
    perfect precision.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes")
    if kind == "state":
        if not event_states:
            raise ValueError("event_states is required for state signals")
        actual_event = np.isin(actual.astype(int), sorted(event_states))
        predicted_event = np.isin(predicted.astype(int), sorted(event_states))
    elif kind == "numeric":
        if event_threshold is None:
            raise ValueError("event_threshold is required for numeric signals")
        actual = actual.astype(float)
        predicted = predicted.astype(float)
        actual_event = actual >= event_threshold
        predicted_event = predicted >= event_threshold
    else:
        raise ValueError("kind must be 'state' or 'numeric'")
    predicted_count = int(predicted_event.sum())
    if kind == "numeric":
        correct_event = predicted_event & (
            np.abs(predicted - actual) <= relative_tolerance * np.maximum(np.abs(actual), 1e-12))
    else:
        correct_event = predicted_event & (predicted == actual)
    true_positive_count = int(correct_event.sum())
    return {
        "event_applicable": True,
        "event_definition": (
            {"kind": "state", "event_states": sorted(event_states)}
            if kind == "state" else
            {"kind": "numeric", "event_threshold": float(event_threshold),
             "relative_tolerance": float(relative_tolerance)}),
        "predicted_event_count": predicted_count,
        "true_positive_count": true_positive_count,
        "true_positive_precision": (
            float(true_positive_count / predicted_count) if predicted_count else None),
        "actual_event_count": int(actual_event.sum()),
    }
