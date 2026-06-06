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
    maximum_false_alarms_per_day: float,
) -> ThresholdSelectionResult:
    """Select by constraints, Event F1, delay, false alarms and Point F1."""

    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    feasible = [
        candidate
        for candidate in candidates
        if _metric(candidate, "event_recall", 0.0) >= minimum_event_recall
        and _metric(candidate, "false_alarms_per_day", float("inf"))
        <= maximum_false_alarms_per_day
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                -_metric(candidate, "event_f1", 0.0),
                _metric(
                    candidate,
                    "median_detection_delay_minutes",
                    float("inf"),
                ),
                _metric(candidate, "false_alarms_per_day", float("inf")),
                -_metric(candidate, "point_f1", 0.0),
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
            _metric(candidate, "false_alarms_per_day", float("inf")),
            -_metric(candidate, "event_f1", 0.0),
            _metric(
                candidate,
                "median_detection_delay_minutes",
                float("inf"),
            ),
            -_metric(candidate, "point_f1", 0.0),
        ),
    )
    has_validation_events = any(
        int(candidate.get("true_events", 0)) > 0 for candidate in candidates
    )
    return ThresholdSelectionResult(
        candidate=selected,
        constraints_satisfied=False,
        selection_reason=(
            "constraints_not_satisfied"
            if has_validation_events
            else "no_validation_events"
        ),
    )
