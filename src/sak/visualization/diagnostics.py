"""Diagnostic score traces and channel-time error heatmaps."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from sak.anomaly.events import DetectedEvent

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_score_timeline(
    timestamps: pd.DatetimeIndex,
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    threshold: float | np.ndarray,
    labels: np.ndarray,
    events: Sequence[DetectedEvent],
    output_path: Path,
    title: str,
) -> None:
    """Save raw/filtered scores with anomaly and alarm context."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(15, 5))
    axis.plot(timestamps, raw_scores, color="#9aa4b2", linewidth=0.8, label="Raw score")
    axis.plot(
        timestamps,
        smoothed_scores,
        color="#235789",
        linewidth=1.2,
        label="EWMA score",
    )
    threshold_values = np.asarray(threshold, dtype=float)
    if threshold_values.ndim == 0:
        axis.axhline(
            float(threshold_values),
            color="#d1495b",
            linestyle="--",
            label="Threshold",
        )
        scale_threshold = float(threshold_values)
    elif threshold_values.shape == raw_scores.shape:
        axis.plot(
            timestamps,
            threshold_values,
            color="#d1495b",
            linestyle="--",
            linewidth=1.0,
            label="Threshold",
        )
        scale_threshold = float(np.nanmedian(threshold_values))
    else:
        raise ValueError("threshold must be scalar or aligned with raw_scores")

    label_indices = np.flatnonzero(labels)
    if len(label_indices):
        groups = np.split(label_indices, np.where(np.diff(label_indices) > 1)[0] + 1)
        for group in groups:
            axis.axvspan(
                timestamps[int(group[0])],
                timestamps[int(group[-1])],
                color="#f4b942",
                alpha=0.18,
            )
    for event in events:
        axis.axvspan(event.start_time, event.end_time, color="#2e8b57", alpha=0.15)

    axis.set_title(title)
    axis.set_ylabel("Anomaly score")
    axis.set_xlabel("Time")
    axis.set_yscale("symlog", linthresh=max(scale_threshold / 2.0, 1e-6))
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_error_heatmap(
    timestamps: pd.DatetimeIndex,
    channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    """Save a channel-by-time log reconstruction-error heatmap."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 7))
    image = axis.imshow(
        np.log1p(channel_errors.T),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axis.set_yticks(np.arange(len(channel_names)))
    axis.set_yticklabels(channel_names)
    tick_count = min(8, len(timestamps))
    tick_indices = np.linspace(0, len(timestamps) - 1, tick_count, dtype=int)
    axis.set_xticks(tick_indices)
    axis.set_xticklabels(
        [timestamps[index].strftime("%m-%d %H:%M") for index in tick_indices],
        rotation=30,
        ha="right",
    )
    axis.set_title(title)
    axis.set_xlabel("Test timeline")
    figure.colorbar(image, ax=axis, label="log(1 + channel error)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_temporal_window_error_heatmap(
    window_channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    """Save mean reconstruction error per temporal window and channel."""

    errors = np.asarray(window_channel_errors, dtype=float)
    if errors.ndim != 3 or errors.shape[2] != len(channel_names):
        raise ValueError(
            "window_channel_errors must have shape [windows, time, channels]"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_channel_mean = errors.mean(axis=1)
    figure, axis = plt.subplots(figsize=(16, 7))
    image = axis.imshow(
        np.log1p(window_channel_mean.T),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axis.set_yticks(np.arange(len(channel_names)))
    axis.set_yticklabels(channel_names)
    axis.set_xlabel("Window index")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="log(1 + mean window error)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
