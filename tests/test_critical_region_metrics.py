from dataclasses import dataclass

import pandas as pd

from sak.anomaly.events import DetectedEvent
from sak.evaluation import event_metrics


@dataclass(frozen=True)
class TrueEvent:
    event_id: str
    early_warning_region_start: pd.Timestamp
    start: pd.Timestamp
    failure_region_start: pd.Timestamp
    end: pd.Timestamp


def _true_event() -> TrueEvent:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    return TrueEvent(
        event_id="SYN-0001",
        early_warning_region_start=base,
        start=base + pd.Timedelta(minutes=10),
        failure_region_start=base + pd.Timedelta(minutes=30),
        end=base + pd.Timedelta(minutes=50),
    )


def _prediction(
    start: pd.Timestamp,
    *,
    duration_minutes: int = 2,
    event_id: str = "SAK-0001",
) -> DetectedEvent:
    return DetectedEvent(
        event_id=event_id,
        start_index=0,
        end_index=duration_minutes,
        peak_index=0,
        start_time=start,
        end_time=start + pd.Timedelta(minutes=duration_minutes),
        peak_time=start,
        peak_score=1.0,
    )


def test_precursor_alarm_counts_as_before_critical_and_precursor() -> None:
    truth = _true_event()
    metrics = event_metrics([_prediction(truth.start - pd.Timedelta(minutes=4))], [truth])

    assert metrics["recall"] == 1.0
    assert metrics["precursor_detection_rate"] == 1.0
    assert metrics["detected_before_critical_rate"] == 1.0
    assert metrics["critical_region_recall"] == 1.0
    assert metrics["median_lead_time_to_critical_minutes"] == 24.0
    assert metrics["p10_lead_time_to_critical_minutes"] == 24.0
    assert metrics["late_detection_rate"] == 0.0


def test_alarm_after_onset_but_before_critical_is_not_precursor() -> None:
    truth = _true_event()
    metrics = event_metrics([_prediction(truth.start + pd.Timedelta(minutes=5))], [truth])

    assert metrics["recall"] == 1.0
    assert metrics["precursor_detection_rate"] == 0.0
    assert metrics["detected_before_critical_rate"] == 1.0
    assert metrics["critical_region_recall"] == 1.0
    assert metrics["median_lead_time_to_critical_minutes"] == 15.0


def test_late_critical_alarm_covers_critical_region_with_negative_lead() -> None:
    truth = _true_event()
    metrics = event_metrics(
        [_prediction(truth.failure_region_start + pd.Timedelta(minutes=3))],
        [truth],
    )

    assert metrics["recall"] == 1.0
    assert metrics["detected_before_critical_rate"] == 0.0
    assert metrics["critical_region_recall"] == 1.0
    assert metrics["late_detection_rate"] == 1.0
    assert metrics["median_lead_time_to_critical_minutes"] == -3.0
    assert metrics["matches"][0]["critical_region_covered"] is True


def test_missed_critical_event_is_counted() -> None:
    metrics = event_metrics([], [_true_event()])

    assert metrics["recall"] == 0.0
    assert metrics["critical_region_recall"] == 0.0
    assert metrics["missed_critical_count"] == 1


def test_false_alarm_only_counts_false_alarm_rate() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    metrics = event_metrics(
        [_prediction(start)],
        [],
        observation_duration=pd.Timedelta(days=2),
    )

    assert metrics["false_alarm_events"] == 1
    assert metrics["false_alarms_per_day"] == 0.5


def test_event_recall_can_diverge_from_critical_region_recall() -> None:
    truth = _true_event()
    metrics = event_metrics(
        [
            _prediction(
                truth.end + pd.Timedelta(minutes=3),
                duration_minutes=1,
            )
        ],
        [truth],
        tolerance=pd.Timedelta(minutes=5),
    )

    assert metrics["recall"] == 1.0
    assert metrics["critical_region_recall"] == 0.0
    assert metrics["missed_critical_count"] == 1
    assert metrics["late_detection_rate"] == 1.0
