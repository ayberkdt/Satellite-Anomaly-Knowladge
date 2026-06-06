"""Map overlapping temporal reconstruction errors back to source timestamps."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

Aggregation = Literal["mean", "max"]


def aggregate_window_errors_to_timestamps(
    *,
    source_indices: np.ndarray,
    window_channel_errors: np.ndarray,
    n_samples: int,
    aggregation: Aggregation = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window-position errors into timestamp scores and channel errors.

    Timestamps not covered by any window are assigned zero and reported with a
    runtime warning. For leakage safety, ``source_indices`` should come from
    windows built independently inside one chronological data partition.
    """

    indices = np.asarray(source_indices)
    errors = np.asarray(window_channel_errors, dtype=float)
    if indices.ndim != 2:
        raise ValueError("source_indices must have shape [windows, window_size]")
    if errors.ndim != 3:
        raise ValueError(
            "window_channel_errors must have shape [windows, window_size, channels]"
        )
    if indices.shape != errors.shape[:2]:
        raise ValueError(
            "source_indices and window_channel_errors must align on windows "
            "and window_size"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("source_indices must contain integers")
    if n_samples < 0:
        raise ValueError("n_samples cannot be negative")
    if errors.shape[2] < 1:
        raise ValueError("window_channel_errors must contain at least one channel")
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: mean, max")
    if not np.all(np.isfinite(errors)):
        raise ValueError("window_channel_errors must contain only finite values")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_samples):
        raise ValueError("source_indices contain values outside [0, n_samples)")

    channel_count = errors.shape[2]
    flat_indices = indices.reshape(-1)
    flat_errors = errors.reshape(-1, channel_count)
    counts = np.zeros(n_samples, dtype=np.int64)
    np.add.at(counts, flat_indices, 1)

    if aggregation == "mean":
        timestamp_channel_errors = np.zeros(
            (n_samples, channel_count),
            dtype=float,
        )
        np.add.at(timestamp_channel_errors, flat_indices, flat_errors)
        covered = counts > 0
        timestamp_channel_errors[covered] /= counts[covered, np.newaxis]
    else:
        timestamp_channel_errors = np.full(
            (n_samples, channel_count),
            -np.inf,
            dtype=float,
        )
        np.maximum.at(timestamp_channel_errors, flat_indices, flat_errors)
        covered = counts > 0
        timestamp_channel_errors[~covered] = 0.0

    uncovered_count = int(np.sum(~covered))
    if uncovered_count:
        warnings.warn(
            f"{uncovered_count} timestamps were not covered by temporal windows; "
            "their reconstruction errors were set to zero",
            RuntimeWarning,
            stacklevel=2,
        )
    timestamp_scores = timestamp_channel_errors.mean(axis=1)
    return timestamp_scores, timestamp_channel_errors
