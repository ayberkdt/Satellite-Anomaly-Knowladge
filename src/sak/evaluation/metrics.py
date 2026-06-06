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


def _timestamp_attr(event: IntervalEvent, name: str, default: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(getattr(event, name, default))


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
    """One-to-one matching with anomaly delay and critical-region lead time."""

    unmatched_predictions = set(range(len(predicted_events)))
    matches: list[dict[str, object]] = []

    for true_event in sorted(true_events, key=lambda event: event.start):
        candidates: list[tuple[pd.Timedelta, int]] = []
        early_warning_start = _timestamp_attr(
            true_event,
            "early_warning_region_start",
            pd.Timestamp(true_event.start),
        )
        critical_start = _timestamp_attr(
            true_event,
            "failure_region_start",
            pd.Timestamp(true_event.start),
        )
        expanded_start = early_warning_start
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
        anomaly_delay = (
            prediction.start_time - true_event.start
        ).total_seconds() / 60.0
        lead_time = (
            critical_start - prediction.start_time
        ).total_seconds() / 60.0
        detected_before_critical = prediction.start_time < critical_start
        detected_in_precursor = (
            prediction.start_time >= early_warning_start
            and prediction.start_time < true_event.start
        )
        critical_region_covered = (
            prediction.start_time <= true_event.end
            and prediction.end_time >= critical_start
        )
        late_detection = prediction.start_time > critical_start
        matches.append(
            {
                "true_event_id": true_event.event_id,
                "predicted_event_id": prediction.event_id,
                "detection_delay_minutes": anomaly_delay,
                "detection_delay_from_anomaly_start": anomaly_delay,
                "lead_time_to_critical_region_minutes": lead_time,
                "detected_before_critical_region": detected_before_critical,
                "detected_in_precursor_region": detected_in_precursor,
                "critical_region_covered": critical_region_covered,
                "late_detection": late_detection,
            }
        )

    match_count = len(matches)
    predicted_count = len(predicted_events)
    true_count = len(true_events)
    precision = match_count / predicted_count if predicted_count else 0.0
    recall = match_count / true_count if true_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [float(item["detection_delay_minutes"]) for item in matches]
    lead_times = [
        float(item["lead_time_to_critical_region_minutes"]) for item in matches
    ]
    detected_before_critical = sum(
        bool(item["detected_before_critical_region"]) for item in matches
    )
    critical_region_hits = sum(
        bool(item["detected_before_critical_region"])
        or bool(item["critical_region_covered"])
        for item in matches
    )
    precursor_detections = sum(
        bool(item["detected_in_precursor_region"]) for item in matches
    )
    late_detections = sum(bool(item["late_detection"]) for item in matches)
    false_alarm_count = predicted_count - match_count
    missed_critical_count = true_count - critical_region_hits

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
        "median_lead_time_to_critical_minutes": (
            float(np.median(lead_times)) if lead_times else None
        ),
        "p10_lead_time_to_critical_minutes": (
            float(np.percentile(lead_times, 10.0)) if lead_times else None
        ),
        "critical_region_recall": (
            critical_region_hits / true_count if true_count else 0.0
        ),
        "detected_before_critical_rate": (
            detected_before_critical / true_count if true_count else 0.0
        ),
        "precursor_detection_rate": (
            precursor_detections / true_count if true_count else 0.0
        ),
        "missed_critical_count": missed_critical_count,
        "late_detection_rate": late_detections / true_count if true_count else 0.0,
        "p90_detection_delay_minutes": float(np.percentile(delays, 90.0))
        if delays
        else None,
        "matches": matches,
    }
