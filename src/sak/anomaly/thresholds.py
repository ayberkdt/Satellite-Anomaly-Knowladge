"""Threshold calibration utilities for anomaly scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from sak.anomaly.early_warning import ewma_smooth


def _as_score_vector(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    return values


def _validate_quantile(quantile: float) -> float:
    value = float(quantile)
    if not 0.0 <= value <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return value


@dataclass(frozen=True)
class ModeThresholdCalibration:
    """Mode-specific thresholds with a global fallback for unseen modes."""

    global_threshold: float
    mode_thresholds: Mapping[str, float]
    quantile: float
    context_column: str = "operational_mode"
    minimum_samples: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.global_threshold):
            raise ValueError("global_threshold must be finite")
        _validate_quantile(self.quantile)
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        for mode, threshold in self.mode_thresholds.items():
            if not str(mode):
                raise ValueError("mode names must be non-empty")
            if not np.isfinite(threshold):
                raise ValueError("mode thresholds must be finite")

    def threshold_for_mode(self, mode: object) -> float:
        """Return the threshold for one mode, falling back to the global value."""

        return float(self.mode_thresholds.get(str(mode), self.global_threshold))

    def thresholds_for_modes(self, modes: Sequence[object]) -> np.ndarray:
        """Vectorize mode lookup for timestamp-aligned thresholding."""

        return np.asarray([self.threshold_for_mode(mode) for mode in modes], dtype=float)

    def thresholds_for_frame(self, context_frame: pd.DataFrame) -> np.ndarray:
        """Return timestamp-aligned thresholds from a context DataFrame."""

        if self.context_column not in context_frame:
            raise ValueError(f"context_frame must contain {self.context_column!r}")
        return self.thresholds_for_modes(context_frame[self.context_column].to_numpy())

    def to_dict(self) -> dict[str, object]:
        """Serialize calibration metadata for experiment artefacts."""

        return {
            "global_threshold": float(self.global_threshold),
            "mode_thresholds": {
                str(mode): float(threshold)
                for mode, threshold in sorted(self.mode_thresholds.items())
            },
            "quantile": float(self.quantile),
            "context_column": self.context_column,
            "minimum_samples": int(self.minimum_samples),
        }


def calibrate_mode_thresholds(
    scores: np.ndarray,
    context_frame: pd.DataFrame,
    *,
    quantile: float,
    context_column: str = "operational_mode",
    minimum_samples: int = 1,
) -> ModeThresholdCalibration:
    """Calibrate global and operational-mode thresholds from nominal calibration scores."""

    values = _as_score_vector(scores)
    quantile_value = _validate_quantile(quantile)
    if len(values) != len(context_frame):
        raise ValueError("scores and context_frame must have equal length")
    if context_column not in context_frame:
        raise ValueError(f"context_frame must contain {context_column!r}")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")

    global_threshold = float(np.quantile(values, quantile_value))
    modes = context_frame[context_column].to_numpy()
    mode_thresholds: dict[str, float] = {}

    for mode in sorted({str(item) for item in modes if not pd.isna(item)}):
        mask = np.asarray([str(item) == mode for item in modes], dtype=bool)
        if int(mask.sum()) < minimum_samples:
            continue
        mode_thresholds[mode] = float(np.quantile(values[mask], quantile_value))

    return ModeThresholdCalibration(
        global_threshold=global_threshold,
        mode_thresholds=mode_thresholds,
        quantile=quantile_value,
        context_column=context_column,
        minimum_samples=minimum_samples,
    )


@dataclass(frozen=True)
class DynamicThresholdResult:
    """Smoothed scores, timestamp thresholds and alarm decisions."""

    smoothed_scores: np.ndarray
    thresholds: np.ndarray
    alarm_mask: np.ndarray


@dataclass(frozen=True)
class ModeAwareThresholdFilter:
    """Apply EWMA and m-of-n persistence against timestamp-specific thresholds."""

    calibration: ModeThresholdCalibration
    ewma_alpha: float = 0.2
    minimum_hits: int = 3
    lookback_steps: int = 5
    cooldown_steps: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.minimum_hits < 1:
            raise ValueError("minimum_hits must be positive")
        if self.lookback_steps < self.minimum_hits:
            raise ValueError("lookback_steps must be >= minimum_hits")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps cannot be negative")

    def apply(self, scores: np.ndarray, context_frame: pd.DataFrame) -> DynamicThresholdResult:
        """Filter scores using the threshold attached to each row's operational mode."""

        values = _as_score_vector(scores)
        if len(values) != len(context_frame):
            raise ValueError("scores and context_frame must have equal length")
        if len(values) == 0:
            return DynamicThresholdResult(
                values.copy(),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=bool),
            )

        smoothed = ewma_smooth(values, self.ewma_alpha)
        thresholds = self.calibration.thresholds_for_frame(context_frame)
        exceedance = smoothed > thresholds
        alarm_mask = np.zeros(len(values), dtype=bool)
        cooldown_remaining = 0

        for index in range(len(values)):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            start = max(0, index - self.lookback_steps + 1)
            hit_count = int(exceedance[start : index + 1].sum())
            if exceedance[index] and hit_count >= self.minimum_hits:
                alarm_mask[index] = True
                cooldown_remaining = self.cooldown_steps

        return DynamicThresholdResult(smoothed, thresholds, alarm_mask)
