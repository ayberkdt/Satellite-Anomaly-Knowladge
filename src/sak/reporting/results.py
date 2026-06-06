"""Tabular summaries for SAK experiment outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "pca": "PCA",
    "dense_autoencoder": "Dense Autoencoder",
}


def _metric(payload: dict[str, Any], *path: str, default: float = 0.0) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value) if value is not None else default


def experiment_model_rows(comparison: dict[str, Any]) -> list[dict[str, str | float]]:
    """Flatten model metrics into rows suitable for CSV, Markdown and HTML."""

    rows: list[dict[str, str | float]] = []
    for model_key, model_payload in comparison.items():
        if model_key == "dataset":
            continue
        rows.append(
            {
                "model": MODEL_LABELS.get(model_key, model_key),
                "event_precision": _metric(model_payload, "event_metrics", "precision"),
                "event_recall": _metric(model_payload, "event_metrics", "recall"),
                "event_f1": _metric(model_payload, "event_metrics", "f1"),
                "false_alarms_per_day": _metric(
                    model_payload,
                    "event_metrics",
                    "false_alarms_per_day",
                ),
                "median_delay_min": _metric(
                    model_payload,
                    "event_metrics",
                    "median_detection_delay_minutes",
                ),
                "point_f1": _metric(model_payload, "point_metrics", "f1"),
                "channel_hit_at_3": _metric(
                    model_payload,
                    "xai_metrics",
                    "channel_hit_at_3",
                ),
                "subsystem_hit_at_2": _metric(
                    model_payload,
                    "xai_metrics",
                    "subsystem_hit_at_2",
                ),
                "critical_window_hit_rate": _metric(
                    model_payload,
                    "xai_metrics",
                    "critical_window_hit_rate",
                ),
            }
        )
    return rows


def threshold_sweep_rows(comparison: dict[str, Any]) -> list[dict[str, str | float]]:
    """Flatten threshold sweep entries across all models."""

    rows: list[dict[str, str | float]] = []
    for model_key, model_payload in comparison.items():
        if model_key == "dataset":
            continue
        for entry in model_payload.get("threshold_sweep", []):
            rows.append(
                {
                    "model": MODEL_LABELS.get(model_key, model_key),
                    "quantile": float(entry["quantile"]),
                    "threshold": float(entry["threshold"]),
                    "event_precision": float(entry["event_precision"]),
                    "event_recall": float(entry["event_recall"]),
                    "event_f1": float(entry["event_f1"]),
                    "false_alarms_per_day": float(entry["false_alarms_per_day"]),
                    "median_delay_min": float(
                        entry["median_detection_delay_minutes"]
                    ),
                }
            )
    return rows


def delay_rows(comparison: dict[str, Any]) -> list[dict[str, str | float]]:
    """Flatten true-event match delays for side-by-side model inspection."""

    rows: list[dict[str, str | float]] = []
    for model_key, model_payload in comparison.items():
        if model_key == "dataset":
            continue
        for match in model_payload.get("event_metrics", {}).get("matches", []):
            rows.append(
                {
                    "model": MODEL_LABELS.get(model_key, model_key),
                    "true_event_id": str(match["true_event_id"]),
                    "predicted_event_id": str(match["predicted_event_id"]),
                    "detection_delay_min": float(match["detection_delay_minutes"]),
                }
            )
    return rows


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array at {path}")
    return payload


def event_diagnostic_rows(
    comparison: dict[str, Any],
    artifact_dir: Path,
    manifest_path: Path,
) -> list[dict[str, str | float]]:
    """Join ground-truth injections, matched events and top-channel explanations."""

    manifest = _load_json_array(manifest_path)
    rows: list[dict[str, str | float]] = []
    for model_key, model_payload in comparison.items():
        if model_key == "dataset":
            continue
        model_label = MODEL_LABELS.get(model_key, model_key)
        prefix = model_key.upper()
        predicted_events = {
            str(event["event_id"]): event
            for event in _load_json_array(artifact_dir / model_key / "events.json")
        }
        matches = {
            str(match["true_event_id"]): match
            for match in model_payload.get("event_metrics", {}).get("matches", [])
        }

        for truth in manifest:
            match = matches.get(str(truth["event_id"]))
            if match is None:
                rows.append(
                    {
                        "model": model_label,
                        "true_event_id": str(truth["event_id"]),
                        "anomaly_type": str(truth["anomaly_type"]),
                        "expected_subsystem": str(truth["expected_subsystem"]),
                        "predicted_event_id": "MISS",
                        "detection_delay_min": float("nan"),
                        "top_channels": "",
                        "top_subsystems": "",
                        "channel_hit": "no",
                        "subsystem_hit": "no",
                    }
                )
                continue

            predicted_event_id = str(match["predicted_event_id"])
            event = predicted_events.get(f"{prefix}-{predicted_event_id}", {})
            top_channels_payload = event.get("top_channels", [])
            top_channels = [str(item["channel"]) for item in top_channels_payload[:3]]
            top_subsystems = [
                str(item.get("subsystem", ""))
                for item in top_channels_payload[:3]
                if item.get("subsystem")
            ]
            expected_channels = {str(channel) for channel in truth["affected_channels"]}
            expected_subsystem = str(truth["expected_subsystem"])
            rows.append(
                {
                    "model": model_label,
                    "true_event_id": str(truth["event_id"]),
                    "anomaly_type": str(truth["anomaly_type"]),
                    "expected_subsystem": expected_subsystem,
                    "predicted_event_id": predicted_event_id,
                    "detection_delay_min": float(match["detection_delay_minutes"]),
                    "top_channels": ", ".join(top_channels),
                    "top_subsystems": ", ".join(dict.fromkeys(top_subsystems)),
                    "channel_hit": "yes"
                    if expected_channels.intersection(top_channels)
                    else "no",
                    "subsystem_hit": "yes"
                    if expected_subsystem in top_subsystems
                    else "no",
                }
            )
    return rows


def false_positive_rows(
    comparison: dict[str, Any],
    artifact_dir: Path,
) -> list[dict[str, str | float]]:
    """List predicted events that did not match any ground-truth injection."""

    rows: list[dict[str, str | float]] = []
    for model_key, model_payload in comparison.items():
        if model_key == "dataset":
            continue
        matched_ids = {
            str(match["predicted_event_id"])
            for match in model_payload.get("event_metrics", {}).get("matches", [])
        }
        prefix = model_key.upper()
        model_label = MODEL_LABELS.get(model_key, model_key)
        for event in _load_json_array(artifact_dir / model_key / "events.json"):
            internal_id = str(event["event_id"]).replace(f"{prefix}-", "", 1)
            if internal_id in matched_ids:
                continue
            rows.append(
                {
                    "model": model_label,
                    "predicted_event_id": internal_id,
                    "start": str(event["start"]),
                    "end": str(event["end"]),
                    "risk_level": str(event["risk_level"]),
                    "operational_mode": str(
                        event.get("context", {}).get("operational_mode", "")
                    ),
                    "top_channels": ", ".join(
                        str(item["channel"]) for item in event.get("top_channels", [])[:3]
                    ),
                }
            )
    return rows


def channel_summary_rows(
    artifact_dir: Path,
) -> list[dict[str, str | float]]:
    """Aggregate how often channels appear in top-3 explanations."""

    summary: dict[tuple[str, str], dict[str, str | float]] = {}
    for model_key, model_label in MODEL_LABELS.items():
        for event in _load_json_array(artifact_dir / model_key / "events.json"):
            for rank, channel in enumerate(event.get("top_channels", [])[:3], start=1):
                key = (model_label, str(channel["channel"]))
                entry = summary.setdefault(
                    key,
                    {
                        "model": model_label,
                        "channel": str(channel["channel"]),
                        "subsystem": str(channel.get("subsystem", "")),
                        "top3_count": 0.0,
                        "mean_contribution": 0.0,
                        "rank1_count": 0.0,
                    },
                )
                count = float(entry["top3_count"])
                contribution = float(channel.get("contribution", 0.0))
                entry["mean_contribution"] = (
                    (float(entry["mean_contribution"]) * count + contribution)
                    / (count + 1.0)
                )
                entry["top3_count"] = count + 1.0
                if rank == 1:
                    entry["rank1_count"] = float(entry["rank1_count"]) + 1.0

    return sorted(
        summary.values(),
        key=lambda row: (
            str(row["model"]),
            -float(row["top3_count"]),
            -float(row["mean_contribution"]),
        ),
    )


def subsystem_summary_rows(
    artifact_dir: Path,
) -> list[dict[str, str | float]]:
    """Aggregate top-channel contribution mass by subsystem."""

    summary: dict[tuple[str, str], dict[str, str | float]] = {}
    for model_key, model_label in MODEL_LABELS.items():
        for event in _load_json_array(artifact_dir / model_key / "events.json"):
            for channel in event.get("top_channels", [])[:5]:
                subsystem = str(channel.get("subsystem", "UNKNOWN"))
                key = (model_label, subsystem)
                entry = summary.setdefault(
                    key,
                    {
                        "model": model_label,
                        "subsystem": subsystem,
                        "total_contribution": 0.0,
                        "channel_mentions": 0.0,
                    },
                )
                entry["total_contribution"] = float(entry["total_contribution"]) + float(
                    channel.get("contribution", 0.0)
                )
                entry["channel_mentions"] = float(entry["channel_mentions"]) + 1.0

    return sorted(
        summary.values(),
        key=lambda row: (str(row["model"]), -float(row["total_contribution"])),
    )


def write_csv(path: Path, rows: Iterable[dict[str, str | float]]) -> None:
    """Write rows to CSV, creating parent folders when needed."""

    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def markdown_table(rows: list[dict[str, str | float]]) -> str:
    """Render a small GitHub-flavored Markdown table."""

    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_markdown(path: Path, comparison: dict[str, Any]) -> None:
    """Write a compact Markdown summary of the first experiment."""

    dataset = comparison["dataset"]
    model_rows = experiment_model_rows(comparison)
    sweep_rows = threshold_sweep_rows(comparison)
    delays = delay_rows(comparison)
    text = f"""# SAK Synthetic Results Summary

## Dataset

- Rows: {dataset["rows"]}
- Channels: {dataset["channels"]}
- Train rows: {dataset["train_rows"]}
- Validation rows: {dataset["validation_rows"]}
- Test rows: {dataset["test_rows"]}
- Test events: {dataset["test_events"]}

## Model Comparison

{markdown_table(model_rows)}

## Threshold Sweep

{markdown_table(sweep_rows)}

## Detection Delays

{markdown_table(delays)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
