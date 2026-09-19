import numpy as np

from arkansas_pharma_signal.event_accuracy import event_route_passes, score_event_predictions


def test_state_event_precision_uses_only_predicted_high_states():
    result = score_event_predictions(
        np.array([0, 3, 4, 2, 4]), np.array([0, 3, 2, 4, 4]),
        kind="state", event_states={3, 4})
    assert result["predicted_event_count"] == 3
    assert result["true_positive_count"] == 2
    assert result["true_positive_precision"] == 2 / 3


def test_numeric_event_precision_requires_threshold_and_five_percent_error():
    result = score_event_predictions(
        np.array([80.0, 90.0, 50.0, 100.0]),
        np.array([82.0, 96.0, 90.0, 69.0]),
        kind="numeric", event_threshold=70.0)
    assert result["predicted_event_count"] == 3
    assert result["true_positive_count"] == 1
    assert result["true_positive_precision"] == 1 / 3


def test_numeric_event_prediction_can_be_correct_below_event_threshold():
    result = score_event_predictions(
        np.array([68.0]), np.array([70.0]), kind="numeric", event_threshold=70.0)
    assert result["predicted_event_count"] == 1
    assert result["actual_event_count"] == 0
    assert result["true_positive_count"] == 1
    assert result["true_positive_precision"] == 1.0


def test_no_predicted_events_are_not_perfect_precision():
    result = score_event_predictions(
        np.array([0, 1]), np.array([0, 1]), kind="state", event_states={3, 4})
    assert result["true_positive_precision"] is None


def test_event_route_retains_raw_floor_and_true_positive_floor():
    assert event_route_passes(0.65, 0.80)
    assert event_route_passes(0.75, 0.80)
    assert not event_route_passes(0.64, 0.99)
    assert not event_route_passes(0.90, 0.79)
    assert not event_route_passes(None, 1.0)
