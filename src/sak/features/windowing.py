"""Leakage-aware utilities for building temporal telemetry windows.

Windowing must be applied independently after the chronological train,
validation and test split. Passing a concatenated dataset can create windows
that cross split boundaries and leak future information into model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

LabelMode = Literal["any", "last", "majority"]


@dataclass(frozen=True)
class WindowedData:
    """Temporal windows and their aligned optional metadata."""

    X_windows: np.ndarray
    window_start_time: np.ndarray | None
    window_end_time: np.ndarray | None
    window_label: np.ndarray | None
    source_indices: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        """Expose the feature shape for convenient array-like inspection."""

        return self.X_windows.shape


def _as_feature_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError("values must have shape [samples] or [samples, channels]")
    return matrix


def _window_labels(
    labels: np.ndarray,
    indices: np.ndarray,
    label_mode: LabelMode,
) -> np.ndarray:
    label_windows = np.asarray(labels)[indices]
    binary = label_windows.astype(bool)
    if label_mode == "any":
        return binary.any(axis=1).astype(int)
    if label_mode == "last":
        return label_windows[:, -1].copy()
    if label_mode == "majority":
        return (binary.sum(axis=1) > binary.shape[1] / 2.0).astype(int)
    raise ValueError("label_mode must be one of: any, last, majority")


@overload
def build_windows(
    values: np.ndarray,
    timestamps: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    window_size: int = 60,
    stride: int = 1,
    label_mode: LabelMode = "any",
    return_index: Literal[True] = True,
) -> WindowedData: ...


@overload
def build_windows(
    values: np.ndarray,
    timestamps: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    window_size: int = 60,
    stride: int = 1,
    label_mode: LabelMode = "any",
    return_index: Literal[False] = False,
) -> np.ndarray: ...


def build_windows(
    values: np.ndarray,
    timestamps: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    window_size: int = 60,
    stride: int = 1,
    label_mode: LabelMode = "any",
    return_index: bool = True,
) -> WindowedData | np.ndarray:
    """Build fixed-length windows without crossing an input partition boundary.

    Call this function separately for train, validation and test partitions.
    The function cannot infer split boundaries from a concatenated input.
    """

    matrix = _as_feature_matrix(values)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if label_mode not in {"any", "last", "majority"}:
        raise ValueError("label_mode must be one of: any, last, majority")
    if timestamps is not None:
        if np.asarray(timestamps).ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if len(timestamps) != len(matrix):
            raise ValueError("timestamps and values must have equal length")
    if labels is not None:
        if np.asarray(labels).ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if len(labels) != len(matrix):
            raise ValueError("labels and values must have equal length")

    starts = np.arange(
        0,
        max(len(matrix) - window_size + 1, 0),
        stride,
        dtype=int,
    )
    if len(starts):
        offsets = np.arange(window_size, dtype=int)
        source_indices = starts[:, np.newaxis] + offsets
        windows = matrix[source_indices]
    else:
        source_indices = np.empty((0, window_size), dtype=int)
        windows = np.empty(
            (0, window_size, matrix.shape[1]),
            dtype=matrix.dtype,
        )

    if not return_index:
        return windows

    timestamp_values = np.asarray(timestamps) if timestamps is not None else None
    start_times = (
        timestamp_values[source_indices[:, 0]]
        if timestamp_values is not None and len(source_indices)
        else (
            np.empty(0, dtype=timestamp_values.dtype)
            if timestamp_values is not None
            else None
        )
    )
    end_times = (
        timestamp_values[source_indices[:, -1]]
        if timestamp_values is not None and len(source_indices)
        else (
            np.empty(0, dtype=timestamp_values.dtype)
            if timestamp_values is not None
            else None
        )
    )
    window_labels = (
        _window_labels(np.asarray(labels), source_indices, label_mode)
        if labels is not None and len(source_indices)
        else (
            np.empty(0, dtype=np.asarray(labels).dtype)
            if labels is not None
            else None
        )
    )
    return WindowedData(
        X_windows=windows,
        window_start_time=start_times,
        window_end_time=end_times,
        window_label=window_labels,
        source_indices=source_indices,
    )
