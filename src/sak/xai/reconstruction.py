"""Reconstruction-error explanations for PCA and autoencoder models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sak.anomaly.events import DetectedEvent
from sak.contracts import ChannelContribution, ExplanationResult


def load_subsystem_mapping(path: Path) -> dict[str, str]:
    """Load channel-to-subsystem mapping from the project YAML catalog."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for subsystem, details in payload["subsystems"].items():
        for channel in details.get("channels", []):
            result.setdefault(channel, subsystem)
    return result


def build_reconstruction_explanation(
    event: DetectedEvent,
    timestamps: pd.DatetimeIndex,
    scaled_values: np.ndarray,
    channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    top_k: int = 5,
    critical_radius_steps: int = 5,
) -> ExplanationResult:
    """Summarize channel and temporal reconstruction errors for one event."""

    event_errors = channel_errors[event.start_index : event.end_index + 1]
    mean_errors = event_errors.mean(axis=0)
    total_error = float(mean_errors.sum())
    normalized = (
        mean_errors / total_error
        if total_error > 0.0
        else np.zeros_like(mean_errors, dtype=float)
    )
    ranking = np.argsort(normalized)[::-1][:top_k]

    event_values = scaled_values[event.start_index : event.end_index + 1]
    contributions: list[ChannelContribution] = []
    subsystem_scores: dict[str, float] = {}
    for channel_index in ranking:
        channel = channel_names[int(channel_index)]
        subsystem = subsystem_mapping.get(channel)
        direction = "high" if float(event_values[:, channel_index].mean()) >= 0.0 else "low"
        contribution = float(normalized[channel_index])
        contributions.append(
            ChannelContribution(
                channel=channel,
                contribution=contribution,
                subsystem=subsystem,
                direction=direction,
            )
        )
        if subsystem is not None:
            subsystem_scores[subsystem] = subsystem_scores.get(subsystem, 0.0) + contribution

    temporal_errors = event_errors.sum(axis=1)
    local_peak = int(np.argmax(temporal_errors))
    critical_center = event.start_index + local_peak
    critical_start_index = max(event.start_index, critical_center - critical_radius_steps)
    critical_end_index = min(event.end_index, critical_center + critical_radius_steps)
    possible_subsystems = tuple(
        subsystem
        for subsystem, _ in sorted(
            subsystem_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
    )
    concentration = float(normalized[ranking[: min(3, len(ranking))]].sum())
    top_channels = ", ".join(item.channel for item in contributions[:3])
    subsystem_text = possible_subsystems[0] if possible_subsystems else "unknown"
    notes = (
        (
            f"{subsystem_text} grubunda reconstruction error yükseldi. "
            f"Başlıca kanallar: {top_channels}."
        ),
        "Bu çıktı kök neden kanıtı değil, mühendislik inceleme önceliğidir.",
    )

    return ExplanationResult(
        method="reconstruction_error_attribution",
        contributions=tuple(contributions),
        critical_start=timestamps[critical_start_index].to_pydatetime(),
        critical_end=timestamps[critical_end_index].to_pydatetime(),
        possible_subsystems=possible_subsystems,
        confidence=concentration,
        notes=notes,
    )
