"""Shared result contracts between SAK pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreResult:
    """Anomaly scores and optional channel-level reconstruction errors."""

    timestamps: tuple[datetime, ...]
    scores: np.ndarray
    channel_errors: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.scores):
            raise ValueError("timestamps and scores must have the same length")
        if self.channel_errors is not None:
            expected_shape = (len(self.scores), len(self.channel_names))
            if self.channel_errors.shape != expected_shape:
                raise ValueError(
                    f"channel_errors shape must be {expected_shape}, "
                    f"got {self.channel_errors.shape}"
                )


@dataclass(frozen=True)
class ChannelContribution:
    """Contribution of one telemetry channel to an anomaly explanation."""

    channel: str
    contribution: float
    subsystem: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class ExplanationResult:
    """Model-independent explanation payload."""

    method: str
    contributions: tuple[ChannelContribution, ...]
    critical_start: datetime
    critical_end: datetime
    possible_subsystems: tuple[str, ...] = ()
    confidence: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlarmEvent:
    """Filtered anomaly event consumed by reports and dashboards."""

    event_id: str
    start_time: datetime
    end_time: datetime
    peak_time: datetime
    peak_score: float
    threshold: float
    risk_level: str
    explanation: ExplanationResult | None = None
    context: dict[str, Any] = field(default_factory=dict)

