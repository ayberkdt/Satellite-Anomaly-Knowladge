"""Constraint-aware threshold and alarm-filter candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Selected candidate and whether operational constraints were satisfied."""

    candidate: dict[str, Any]
    constraints_satisfied: bool
    selection_reason: str


def _metric(candidate: dict[str, Any], name: str, default: float) -> float:
    value = candidate.get(name)
    return float(value) if value is not None else default


def select_threshold_candidate(
    candidates: list[dict[str, Any]],
    *,
    minimum_event_recall: float,
    minimum_critical_recall: float = 0.0,
    minimum_before_critical_rate: float = 0.0,
    maximum_false_alarms_per_day: float,
) -> ThresholdSelectionResult:
    """Select a calibration operating point under event/critical constraints."""

    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    feasible = [
        candidate
        for candidate in candidates
        if _metric(candidate, "event_recall", 0.0) >= minimum_event_recall
        and _metric(candidate, "critical_region_recall", 0.0)
        >= minimum_critical_recall
        and _metric(candidate, "detected_before_critical_rate", 0.0)
        >= minimum_before_critical_rate
        and _metric(candidate, "false_alarms_per_day", float("inf"))
        <= maximum_false_alarms_per_day
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                -_metric(candidate, "event_f1", 0.0),
                -_metric(candidate, "critical_region_recall", 0.0),
                -_metric(candidate, "detected_before_critical_rate", 0.0),
                -_metric(
                    candidate,
                    "median_lead_time_to_critical_minutes",
                    float("-inf"),
                ),
                _metric(candidate, "false_alarms_per_day", float("inf")),
                _metric(
                    candidate,
                    "median_detection_delay_minutes",
                    float("inf"),
                ),
                -_metric(candidate, "point_f1", 0.0),
                -_metric(candidate, "channel_hit_at_3", 0.0),
            ),
        )
        return ThresholdSelectionResult(
            candidate=selected,
            constraints_satisfied=True,
            selection_reason="constraints_satisfied",
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            -_metric(candidate, "event_recall", 0.0),
            -_metric(candidate, "critical_region_recall", 0.0),
            -_metric(candidate, "detected_before_critical_rate", 0.0),
            _metric(candidate, "false_alarms_per_day", float("inf")),
            -_metric(
                candidate,
                "median_lead_time_to_critical_minutes",
                float("-inf"),
            ),
            -_metric(candidate, "event_f1", 0.0),
            _metric(
                candidate,
                "median_detection_delay_minutes",
                float("inf"),
            ),
            -_metric(candidate, "point_f1", 0.0),
        ),
    )
    has_calibration_events = any(
        int(candidate.get("true_events", 0)) > 0 for candidate in candidates
    )
    return ThresholdSelectionResult(
        candidate=selected,
        constraints_satisfied=False,
        selection_reason=(
            "constraints_not_satisfied"
            if has_calibration_events
            else "no_calibration_events"
        ),
    )
