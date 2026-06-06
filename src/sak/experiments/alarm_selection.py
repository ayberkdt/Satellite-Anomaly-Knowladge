"""Leakage-safe threshold and alarm-filter selection on validation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from sak.anomaly import (
    EarlyWarningFilter,
    ModeAwareThresholdFilter,
    build_detected_events,
    calibrate_mode_thresholds,
    ewma_smooth,
)
from sak.anomaly.filter_sweep import (
    build_filter_sweep_grid,
    mark_selected_candidate,
)
from sak.anomaly.threshold_selection import select_threshold_candidate
from sak.evaluation import event_metrics, point_metrics

ThresholdMode = Literal["global", "mode_aware"]


@dataclass(frozen=True)
class AlarmSelection:
    """Validation-selected quantile and alarm-filter parameters."""

    threshold_mode: ThresholdMode
    threshold_selection_strategy: str
    quantile: float
    ewma_alpha: float
    minimum_hits: int
    lookback_steps: int
    merge_gap_steps: int
    constraints_satisfied: bool
    selection_reason: str
    minimum_event_recall: float
    maximum_false_alarms_per_day: float
    sweep_rows: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        """Serialize selected parameters for metrics artefacts."""

        return {
            "threshold_selection_strategy": self.threshold_selection_strategy,
            "selected_quantile": self.quantile,
            "selected_ewma_alpha": self.ewma_alpha,
            "selected_minimum_hits": self.minimum_hits,
            "selected_lookback_steps": self.lookback_steps,
            "selected_merge_gap_steps": self.merge_gap_steps,
            "constraints_satisfied": self.constraints_satisfied,
            "selection_reason": self.selection_reason,
            "minimum_event_recall": self.minimum_event_recall,
            "maximum_false_alarms_per_day": self.maximum_false_alarms_per_day,
        }


@dataclass(frozen=True)
class AppliedAlarmSelection:
    """Timestamp-aligned output from one selected alarm configuration."""

    threshold: float
    thresholds: np.ndarray
    smoothed_scores: np.ndarray
    alarm_mask: np.ndarray
    threshold_metadata: dict[str, Any]


def _validation_true_events(frame: pd.DataFrame) -> tuple[Any, ...]:
    labels = frame["is_anomaly"].to_numpy(dtype=bool)
    return build_detected_events(
        timestamps=frame.index,
        alarm_mask=labels,
        scores=labels.astype(float),
        merge_gap_steps=0,
    )


def _candidate_filter_grid(settings: dict[str, Any]) -> list[dict[str, float | int]]:
    early_warning = settings["early_warning"]
    defaults = {
        "ewma_alpha": float(early_warning["ewma_alpha"]),
        "minimum_hits": int(early_warning["minimum_hits"]),
        "lookback_steps": int(early_warning["lookback_steps"]),
        "merge_gap_steps": int(early_warning["merge_gap_steps"]),
    }
    sweep = settings.get("alarm_filter_sweep", {})
    if not bool(sweep.get("enabled", False)):
        return [defaults]
    rows = build_filter_sweep_grid(
        ewma_alpha=[float(value) for value in sweep["ewma_alpha"]],
        minimum_hits=[int(value) for value in sweep["minimum_hits"]],
        lookback_steps=[int(value) for value in sweep["lookback_steps"]],
        merge_gap_steps=[int(value) for value in sweep["merge_gap_steps"]],
    )
    if defaults not in rows:
        rows.append(defaults)
    return rows


def _candidate_quantiles(settings: dict[str, Any]) -> list[float]:
    selection = settings.get("threshold_selection", {})
    strategy = str(selection.get("strategy", "quantile"))
    if strategy == "quantile":
        return [float(settings["early_warning"]["threshold_quantile"])]
    if strategy != "constrained_event_f1":
        raise ValueError(
            "threshold_selection.strategy must be quantile or constrained_event_f1"
        )
    quantiles = [float(value) for value in selection["candidate_quantiles"]]
    if not quantiles:
        raise ValueError("candidate_quantiles cannot be empty")
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("candidate_quantiles must be in [0, 1]")
    return quantiles


def _apply_candidate(
    *,
    scores: np.ndarray,
    frame: pd.DataFrame,
    threshold_mode: ThresholdMode,
    quantile: float,
    filter_settings: dict[str, float | int],
    settings: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    alpha = float(filter_settings["ewma_alpha"])
    smoothed = ewma_smooth(scores, alpha)
    nominal_mask = ~frame["is_anomaly"].to_numpy(dtype=bool)
    if not nominal_mask.any():
        raise ValueError("validation must contain nominal samples")
    if threshold_mode == "global":
        threshold = float(np.quantile(smoothed[nominal_mask], quantile))
        warning = EarlyWarningFilter(
            threshold=threshold,
            ewma_alpha=alpha,
            minimum_hits=int(filter_settings["minimum_hits"]),
            lookback_steps=int(filter_settings["lookback_steps"]),
        ).apply(scores)
        return (
            threshold,
            np.full(len(scores), threshold, dtype=float),
            warning.smoothed_scores,
            warning.alarm_mask,
            {"strategy": "global", "quantile": quantile},
        )

    early_warning = settings["early_warning"]
    context_column = str(
        early_warning.get("mode_context_column", "operational_mode")
    )
    calibration = calibrate_mode_thresholds(
        smoothed[nominal_mask],
        frame.loc[nominal_mask],
        quantile=quantile,
        context_column=context_column,
        minimum_samples=int(early_warning.get("mode_minimum_samples", 1)),
    )
    warning = ModeAwareThresholdFilter(
        calibration=calibration,
        ewma_alpha=alpha,
        minimum_hits=int(filter_settings["minimum_hits"]),
        lookback_steps=int(filter_settings["lookback_steps"]),
    ).apply(scores, frame)
    return (
        calibration.global_threshold,
        warning.thresholds,
        warning.smoothed_scores,
        warning.alarm_mask,
        {"strategy": "mode_aware", **calibration.to_dict()},
    )


def select_alarm_configuration(
    *,
    validation_scores: np.ndarray,
    validation_frame: pd.DataFrame,
    threshold_mode: ThresholdMode,
    settings: dict[str, Any],
) -> AlarmSelection:
    """Evaluate threshold/filter candidates exclusively on validation data."""

    selection_settings = settings.get("threshold_selection", {})
    strategy = str(selection_settings.get("strategy", "quantile"))
    minimum_recall = float(
        selection_settings.get("minimum_event_recall", 0.90)
    )
    maximum_false_alarms = float(
        selection_settings.get("maximum_false_alarms_per_day", 0.50)
    )
    true_events = _validation_true_events(validation_frame)
    observation_duration = (
        validation_frame.index[-1]
        - validation_frame.index[0]
        + pd.Timedelta(minutes=1)
    )
    candidates: list[dict[str, Any]] = []
    for quantile in _candidate_quantiles(settings):
        for filter_settings in _candidate_filter_grid(settings):
            threshold, _, smoothed, alarm_mask, threshold_metadata = (
                _apply_candidate(
                    scores=validation_scores,
                    frame=validation_frame,
                    threshold_mode=threshold_mode,
                    quantile=quantile,
                    filter_settings=filter_settings,
                    settings=settings,
                )
            )
            detected_events = build_detected_events(
                timestamps=validation_frame.index,
                alarm_mask=alarm_mask,
                scores=smoothed,
                merge_gap_steps=int(filter_settings["merge_gap_steps"]),
            )
            event_result = event_metrics(
                predicted_events=detected_events,
                true_events=true_events,
                tolerance=pd.Timedelta(minutes=5),
                observation_duration=observation_duration,
            )
            point_result = point_metrics(
                validation_frame["is_anomaly"].to_numpy(dtype=bool),
                alarm_mask,
            )
            candidates.append(
                {
                    "threshold_mode": threshold_mode,
                    "quantile": quantile,
                    "threshold": threshold,
                    **filter_settings,
                    "event_precision": event_result["precision"],
                    "event_recall": event_result["recall"],
                    "event_f1": event_result["f1"],
                    "false_alarms_per_day": event_result[
                        "false_alarms_per_day"
                    ],
                    "median_detection_delay_minutes": event_result[
                        "median_detection_delay_minutes"
                    ],
                    "point_f1": point_result["f1"],
                    "predicted_events": event_result["predicted_events"],
                    "true_events": event_result["true_events"],
                    "threshold_metadata": threshold_metadata,
                }
            )

    if strategy == "quantile":
        early_warning = settings["early_warning"]
        selected = next(
            candidate
            for candidate in candidates
            if candidate["quantile"]
            == float(early_warning["threshold_quantile"])
            and candidate["ewma_alpha"] == float(early_warning["ewma_alpha"])
            and candidate["minimum_hits"]
            == int(early_warning["minimum_hits"])
            and candidate["lookback_steps"]
            == int(early_warning["lookback_steps"])
            and candidate["merge_gap_steps"]
            == int(early_warning["merge_gap_steps"])
        )
        has_events = bool(true_events)
        constraints_satisfied = (
            has_events
            and float(selected["event_recall"]) >= minimum_recall
            and float(selected["false_alarms_per_day"]) <= maximum_false_alarms
        )
        selection_reason = (
            "fixed_quantile"
            if has_events
            else "fixed_quantile_no_validation_events"
        )
    else:
        selection_result = select_threshold_candidate(
            candidates,
            minimum_event_recall=minimum_recall,
            maximum_false_alarms_per_day=maximum_false_alarms,
        )
        selected = selection_result.candidate
        constraints_satisfied = selection_result.constraints_satisfied
        selection_reason = selection_result.selection_reason

    marked_rows = mark_selected_candidate(candidates, selected)
    return AlarmSelection(
        threshold_mode=threshold_mode,
        threshold_selection_strategy=strategy,
        quantile=float(selected["quantile"]),
        ewma_alpha=float(selected["ewma_alpha"]),
        minimum_hits=int(selected["minimum_hits"]),
        lookback_steps=int(selected["lookback_steps"]),
        merge_gap_steps=int(selected["merge_gap_steps"]),
        constraints_satisfied=constraints_satisfied,
        selection_reason=selection_reason,
        minimum_event_recall=minimum_recall,
        maximum_false_alarms_per_day=maximum_false_alarms,
        sweep_rows=marked_rows,
    )


def apply_alarm_selection(
    *,
    selection: AlarmSelection,
    validation_scores: np.ndarray,
    validation_frame: pd.DataFrame,
    test_scores: np.ndarray,
    test_frame: pd.DataFrame,
    settings: dict[str, Any],
) -> AppliedAlarmSelection:
    """Calibrate on validation and apply the selected filter to test scores."""

    filter_settings: dict[str, float | int] = {
        "ewma_alpha": selection.ewma_alpha,
        "minimum_hits": selection.minimum_hits,
        "lookback_steps": selection.lookback_steps,
        "merge_gap_steps": selection.merge_gap_steps,
    }
    validation_smoothed = ewma_smooth(
        validation_scores,
        selection.ewma_alpha,
    )
    nominal_mask = ~validation_frame["is_anomaly"].to_numpy(dtype=bool)
    if selection.threshold_mode == "global":
        threshold = float(
            np.quantile(
                validation_smoothed[nominal_mask],
                selection.quantile,
            )
        )
        warning = EarlyWarningFilter(
            threshold=threshold,
            ewma_alpha=selection.ewma_alpha,
            minimum_hits=selection.minimum_hits,
            lookback_steps=selection.lookback_steps,
        ).apply(test_scores)
        return AppliedAlarmSelection(
            threshold=threshold,
            thresholds=np.full(len(test_scores), threshold, dtype=float),
            smoothed_scores=warning.smoothed_scores,
            alarm_mask=warning.alarm_mask,
            threshold_metadata={
                "strategy": "global",
                "quantile": selection.quantile,
            },
        )

    early_warning = settings["early_warning"]
    calibration = calibrate_mode_thresholds(
        validation_smoothed[nominal_mask],
        validation_frame.loc[nominal_mask],
        quantile=selection.quantile,
        context_column=str(
            early_warning.get("mode_context_column", "operational_mode")
        ),
        minimum_samples=int(early_warning.get("mode_minimum_samples", 1)),
    )
    warning = ModeAwareThresholdFilter(
        calibration=calibration,
        ewma_alpha=selection.ewma_alpha,
        minimum_hits=selection.minimum_hits,
        lookback_steps=selection.lookback_steps,
    ).apply(test_scores, test_frame)
    return AppliedAlarmSelection(
        threshold=calibration.global_threshold,
        thresholds=warning.thresholds,
        smoothed_scores=warning.smoothed_scores,
        alarm_mask=warning.alarm_mask,
        threshold_metadata={
            "strategy": "mode_aware",
            **calibration.to_dict(),
        },
    )
