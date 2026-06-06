"""Leakage-aware chronological splitting and robust telemetry scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalFrames:
    """Non-overlapping chronological dataset partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class CalibrationFrames:
    """Four non-overlapping chronological experiment partitions."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _validate_chronological_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("frame cannot be empty")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame timestamps must be monotonically increasing")


def _validate_split_fractions(
    fractions: tuple[float, ...],
    names: tuple[str, ...],
) -> None:
    if any(value <= 0.0 for value in fractions):
        joined = ", ".join(names)
        raise ValueError(f"{joined} fractions must be positive")
    if sum(fractions) >= 1.0:
        raise ValueError("split fractions must leave a test partition")


def _split_offsets(total_rows: int, fractions: tuple[float, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for fraction in fractions:
        cursor += int(total_rows * fraction)
        offsets.append(cursor)
    segment_lengths = [offsets[0]]
    segment_lengths.extend(right - left for left, right in zip(offsets, offsets[1:]))
    segment_lengths.append(total_rows - offsets[-1])
    if any(length <= 0 for length in segment_lengths):
        raise ValueError("split fractions produce an empty partition")
    return tuple(offsets)


def chronological_calibration_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.50,
    calibration_fraction: float = 0.20,
    validation_fraction: float = 0.10,
) -> CalibrationFrames:
    """Create train/calibration/validation/test partitions without overlap."""

    fractions = (train_fraction, calibration_fraction, validation_fraction)
    _validate_chronological_frame(frame)
    _validate_split_fractions(fractions, ("train", "calibration", "validation"))
    train_end, calibration_end, validation_end = _split_offsets(len(frame), fractions)
    return CalibrationFrames(
        train=frame.iloc[:train_end].copy(),
        calibration=frame.iloc[train_end:calibration_end].copy(),
        validation=frame.iloc[calibration_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalFrames:
    """Split before any learned preprocessing to prevent future leakage."""

    fractions = (train_fraction, validation_fraction)
    _validate_chronological_frame(frame)
    _validate_split_fractions(fractions, ("train", "validation"))
    train_end, validation_end = _split_offsets(len(frame), fractions)
    return ChronologicalFrames(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


@dataclass
class RobustTelemetryPreprocessor:
    """Causal short-gap filling and train-only robust scaling."""

    channel_names: tuple[str, ...]
    max_forward_fill_steps: int = 5
    epsilon: float = 1e-8
    medians_: np.ndarray | None = None
    scales_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "RobustTelemetryPreprocessor":
        values = frame.loc[:, self.channel_names].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("cannot fit preprocessor with infinite values")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("cannot fit preprocessor with all-missing channels")
        medians = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25.0, axis=0)
        q75 = np.nanpercentile(values, 75.0, axis=0)
        scales = q75 - q25
        if not np.all(np.isfinite(medians)) or not np.all(np.isfinite(scales)):
            raise ValueError("cannot fit preprocessor with all-missing channels")
        scales[scales < self.epsilon] = 1.0
        self.medians_ = medians
        self.scales_ = scales
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.scales_ is None:
            raise RuntimeError("preprocessor must be fitted before transform")

        telemetry = frame.loc[:, self.channel_names].copy()
        telemetry = telemetry.ffill(limit=self.max_forward_fill_steps)
        values = telemetry.to_numpy(dtype=float)
        missing_rows, missing_columns = np.where(~np.isfinite(values))
        if len(missing_rows):
            values[missing_rows, missing_columns] = self.medians_[missing_columns]
        return (values - self.medians_) / self.scales_

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
