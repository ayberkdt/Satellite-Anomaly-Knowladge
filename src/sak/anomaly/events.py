"""Build contiguous anomaly events from filtered alarm decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectedEvent:
    """Index and time bounds for one predicted anomaly event."""

    event_id: str
    start_index: int
    end_index: int
    peak_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_time: pd.Timestamp
    peak_score: float


def build_detected_events(
    timestamps: pd.DatetimeIndex,
    alarm_mask: np.ndarray,
    scores: np.ndarray,
    merge_gap_steps: int = 10,
    minimum_alarm_points: int = 1,
) -> tuple[DetectedEvent, ...]:
    """Merge nearby alarm points into event intervals."""

    mask = np.asarray(alarm_mask, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if len(timestamps) != len(mask) or len(mask) != len(values):
        raise ValueError("timestamps, alarm_mask and scores must have equal length")
    if merge_gap_steps < 0:
        raise ValueError("merge_gap_steps cannot be negative")

    alarm_indices = np.flatnonzero(mask)
    if len(alarm_indices) == 0:
        return ()

    groups: list[list[int]] = [[int(alarm_indices[0])]]
    for raw_index in alarm_indices[1:]:
        index = int(raw_index)
        if index - groups[-1][-1] <= merge_gap_steps + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    events: list[DetectedEvent] = []
    for group in groups:
        if len(group) < minimum_alarm_points:
            continue
        start_index = group[0]
        end_index = group[-1]
        local_scores = values[start_index : end_index + 1]
        peak_index = start_index + int(np.argmax(local_scores))
        events.append(
            DetectedEvent(
                event_id=f"SAK-{len(events) + 1:04d}",
                start_index=start_index,
                end_index=end_index,
                peak_index=peak_index,
                start_time=timestamps[start_index],
                end_time=timestamps[end_index],
                peak_time=timestamps[peak_index],
                peak_score=float(values[peak_index]),
            )
        )
    return tuple(events)

