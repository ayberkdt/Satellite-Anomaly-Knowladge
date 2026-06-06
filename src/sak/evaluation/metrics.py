"""Point-wise and event-wise anomaly detection metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from sak.anomaly.events import DetectedEvent


class IntervalEvent(Protocol):
    """Minimum event interface required by event evaluation."""

    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Calculate binary point-level precision, recall and F1."""

    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("labels and predictions must have the same shape")

    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if truth.any() else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def event_metrics(
    predicted_events: Sequence[DetectedEvent],
    true_events: Sequence[IntervalEvent],
    tolerance: pd.Timedelta = pd.Timedelta(minutes=5),
    observation_duration: pd.Timedelta | None = None,
) -> dict[str, float | int | None | list[dict[str, object]]]:
    """One-to-one interval matching with delay and false-alarm reporting."""

    unmatched_predictions = set(range(len(predicted_events)))
    matches: list[dict[str, object]] = []

    for true_event in sorted(true_events, key=lambda event: event.start):
        candidates: list[tuple[pd.Timedelta, int]] = []
        expanded_start = true_event.start - tolerance
        expanded_end = true_event.end + tolerance
        for prediction_index in unmatched_predictions:
            prediction = predicted_events[prediction_index]
            overlaps = (
                prediction.end_time >= expanded_start
                and prediction.start_time <= expanded_end
            )
            if overlaps:
                distance = abs(prediction.start_time - true_event.start)
                candidates.append((distance, prediction_index))

        if not candidates:
            continue
        _, prediction_index = min(candidates, key=lambda item: item[0])
        prediction = predicted_events[prediction_index]
        unmatched_predictions.remove(prediction_index)
        matches.append(
            {
                "true_event_id": true_event.event_id,
                "predicted_event_id": prediction.event_id,
                "detection_delay_minutes": (
                    prediction.start_time - true_event.start
                ).total_seconds()
                / 60.0,
            }
        )

    match_count = len(matches)
    predicted_count = len(predicted_events)
    true_count = len(true_events)
    precision = match_count / predicted_count if predicted_count else 0.0
    recall = match_count / true_count if true_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [float(item["detection_delay_minutes"]) for item in matches]
    false_alarm_count = predicted_count - match_count

    false_alarms_per_day: float | None = None
    if observation_duration is not None and observation_duration.total_seconds() > 0:
        days = observation_duration.total_seconds() / 86400.0
        false_alarms_per_day = false_alarm_count / days

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_events": match_count,
        "predicted_events": predicted_count,
        "true_events": true_count,
        "false_alarm_events": false_alarm_count,
        "false_alarms_per_day": false_alarms_per_day,
        "median_detection_delay_minutes": float(np.median(delays)) if delays else None,
        "p90_detection_delay_minutes": float(np.percentile(delays, 90.0))
        if delays
        else None,
        "matches": matches,
    }

