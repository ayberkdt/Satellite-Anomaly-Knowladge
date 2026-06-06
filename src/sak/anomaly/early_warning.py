"""Convert noisy anomaly scores into persistent early-warning decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ewma_smooth(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Apply causal exponentially weighted smoothing."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if len(values) == 0:
        return values.copy()

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


@dataclass(frozen=True)
class EarlyWarningResult:
    """Smoothed scores and alarm mask from the early-warning filter."""

    smoothed_scores: np.ndarray
    alarm_mask: np.ndarray


@dataclass(frozen=True)
class EarlyWarningFilter:
    """Apply EWMA smoothing, m-of-n persistence and alarm cooldown."""

    threshold: float
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

    def apply(self, scores: np.ndarray) -> EarlyWarningResult:
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if len(values) == 0:
            return EarlyWarningResult(values.copy(), np.zeros(0, dtype=bool))
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain only finite values")

        smoothed = ewma_smooth(values, self.ewma_alpha)

        exceedance = smoothed > self.threshold
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

        return EarlyWarningResult(smoothed, alarm_mask)
