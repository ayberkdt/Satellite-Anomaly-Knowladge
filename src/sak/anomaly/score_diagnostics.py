"""Score distribution, threshold margin and false-positive context diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _vector(values: np.ndarray, name: str, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if dtype is float and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def score_distribution_summary(
    scores: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Summarize an anomaly score distribution and optional label subsets."""

    values = _vector(scores, "scores")
    if len(values) == 0:
        raise ValueError("scores cannot be empty")
    summary: dict[str, float | int | None] = {
        "count": len(values),
        "score_mean": float(np.mean(values)),
        "score_std": float(np.std(values)),
        "score_median": float(np.median(values)),
        "score_p95": float(np.quantile(values, 0.95)),
        "score_p99": float(np.quantile(values, 0.99)),
        "score_p995": float(np.quantile(values, 0.995)),
        "score_p999": float(np.quantile(values, 0.999)),
        "anomaly_score_mean": None,
        "nominal_score_mean": None,
    }
    if labels is not None:
        truth = _vector(labels, "labels", dtype=bool)
        if truth.shape != values.shape:
            raise ValueError("labels and scores must have equal shape")
        if truth.any():
            summary["anomaly_score_mean"] = float(np.mean(values[truth]))
        if (~truth).any():
            summary["nominal_score_mean"] = float(np.mean(values[~truth]))
    return summary


def threshold_margin_summary(
    scores: np.ndarray,
    thresholds: np.ndarray,
    alarm_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize score distance and ratio relative to aligned thresholds."""

    values = _vector(scores, "scores")
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.ndim == 0:
        threshold_values = np.full(len(values), float(threshold_values))
    if threshold_values.shape != values.shape:
        raise ValueError("thresholds and scores must have equal shape")
    if not np.all(np.isfinite(threshold_values)):
        raise ValueError("thresholds must contain only finite values")
    alarms = _vector(alarm_mask, "alarm_mask", dtype=bool)
    if alarms.shape != values.shape:
        raise ValueError("alarm_mask and scores must have equal shape")
    margins = values - threshold_values
    alarm_ratios = np.divide(
        values[alarms],
        threshold_values[alarms],
        out=np.zeros(int(alarms.sum()), dtype=float),
        where=np.abs(threshold_values[alarms]) > 1e-12,
    )
    return {
        "threshold_margin_mean": float(np.mean(margins)),
        "threshold_margin_p95": float(np.quantile(margins, 0.95)),
        "alarm_count": int(alarms.sum()),
        "alarm_score_ratio_mean": (
            float(np.mean(alarm_ratios)) if len(alarm_ratios) else None
        ),
        "alarm_score_ratio_max": (
            float(np.max(alarm_ratios)) if len(alarm_ratios) else None
        ),
    }


def false_positive_score_context(
    *,
    scores: np.ndarray,
    smoothed_scores: np.ndarray,
    thresholds: np.ndarray,
    alarm_mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return point-level false alarms with operational context."""

    raw = _vector(scores, "scores")
    smoothed = _vector(smoothed_scores, "smoothed_scores")
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.ndim == 0:
        threshold_values = np.full(len(raw), float(threshold_values))
    alarms = _vector(alarm_mask, "alarm_mask", dtype=bool)
    if not (
        raw.shape
        == smoothed.shape
        == threshold_values.shape
        == alarms.shape
        == (len(timestamps),)
        == (len(frame),)
    ):
        raise ValueError("all score context inputs must have equal length")
    if "is_anomaly" not in frame:
        raise ValueError("frame must contain 'is_anomaly'")
    false_positive_mask = alarms & ~frame["is_anomaly"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(false_positive_mask):
        threshold = float(threshold_values[index])
        rows.append(
            {
                "timestamp": timestamps[index].isoformat(),
                "raw_score": float(raw[index]),
                "smoothed_score": float(smoothed[index]),
                "threshold": threshold,
                "threshold_margin": float(smoothed[index] - threshold),
                "score_to_threshold_ratio": (
                    float(smoothed[index] / threshold)
                    if abs(threshold) > 1e-12
                    else None
                ),
                "operational_mode": str(frame.iloc[index].get("operational_mode", "")),
                "eclipse": bool(frame.iloc[index].get("eclipse", False)),
                "orbit_phase": float(frame.iloc[index].get("orbit_phase", 0.0)),
            }
        )
    return rows


def context_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count false-positive points by mode and eclipse state."""

    modes: dict[str, int] = {}
    eclipse: dict[str, int] = {}
    for row in rows:
        mode = str(row["operational_mode"])
        eclipse_key = str(bool(row["eclipse"])).lower()
        modes[mode] = modes.get(mode, 0) + 1
        eclipse[eclipse_key] = eclipse.get(eclipse_key, 0) + 1
    return {
        "operational_mode": dict(sorted(modes.items())),
        "eclipse": dict(sorted(eclipse.items())),
    }
