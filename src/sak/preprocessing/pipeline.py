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


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalFrames:
    """Split before any learned preprocessing to prevent future leakage."""

    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test partition")

    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
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
        medians = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25.0, axis=0)
        q75 = np.nanpercentile(values, 75.0, axis=0)
        scales = q75 - q25
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

