"""Leakage-safe anomaly score transforms fitted on calibration scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScoreTransform = Literal["identity", "none", "log1p", "robust_zscore"]


def _score_vector(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if len(values) == 0:
        raise ValueError("scores cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    return values


@dataclass
class ScoreCalibrator:
    """Fit and apply a deterministic score transform."""

    method: ScoreTransform = "identity"
    epsilon: float = 1e-8
    median_: float | None = None
    scale_: float | None = None
    fitted_: bool = False

    def __post_init__(self) -> None:
        if self.method == "none":
            self.method = "identity"
        if self.method not in {"identity", "log1p", "robust_zscore"}:
            raise ValueError(
                "method must be one of: identity, none, log1p, robust_zscore"
            )
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def fit(self, scores: np.ndarray) -> "ScoreCalibrator":
        """Fit transform parameters on nominal calibration scores only."""

        values = _score_vector(scores)
        if self.method == "log1p" and np.any(values <= -1.0):
            raise ValueError("log1p scores must be greater than -1")
        if self.method == "robust_zscore":
            self.median_ = float(np.median(values))
            q25, q75 = np.percentile(values, [25.0, 75.0])
            scale = float(q75 - q25)
            self.scale_ = scale if scale >= self.epsilon else 1.0
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transform scores after fitting."""

        if not self.fitted_:
            raise RuntimeError("calibrator must be fitted before transform")
        values = _score_vector(scores)
        if self.method == "identity":
            return values.copy()
        if self.method == "log1p":
            if np.any(values <= -1.0):
                raise ValueError("log1p scores must be greater than -1")
            return np.log1p(values)
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("robust_zscore parameters are unavailable")
        return (values - self.median_) / self.scale_

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Fit and transform one calibration score vector."""

        return self.fit(scores).transform(scores)

    def to_dict(self) -> dict[str, float | str | None]:
        """Serialize fitted transform metadata."""

        return {
            "score_transform": self.method,
            "median": self.median_,
            "scale": self.scale_,
        }
