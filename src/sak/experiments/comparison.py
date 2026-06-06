"""Comparison artefacts shared by single-seed and multi-seed runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sak.experiments.artifacts import write_csv, write_json


def comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten stable headline metrics for each model variant."""

    rows: list[dict[str, Any]] = []
    for model_variant, payload in comparison.items():
        if model_variant == "dataset":
            continue
        point = payload.get("point_metrics", {})
        event = payload.get("event_metrics", {})
        xai = payload.get("xai_metrics", {})
        calibration = payload.get("calibration", {})
        rows.append(
            {
                "model_variant": model_variant,
                "threshold_strategy": payload.get("thresholding", {}).get(
                    "strategy", ""
                ),
                "calibration_strategy": calibration.get(
                    "threshold_selection_strategy", ""
                ),
                "score_transform": calibration.get("score_transform", ""),
                "constraints_satisfied": calibration.get(
                    "constraints_satisfied", False
                ),
                "point_precision": point.get("precision", 0.0),
                "point_recall": point.get("recall", 0.0),
                "point_f1": point.get("f1", 0.0),
                "event_precision": event.get("precision", 0.0),
                "event_recall": event.get("recall", 0.0),
                "event_f1": event.get("f1", 0.0),
                "false_alarms_per_day": event.get("false_alarms_per_day", 0.0),
                "detection_delay_minutes": event.get(
                    "median_detection_delay_minutes"
                ),
                "channel_hit_at_1": xai.get("channel_hit_at_1", 0.0),
                "channel_hit_at_3": xai.get("channel_hit_at_3", 0.0),
                "subsystem_hit_at_1": xai.get("subsystem_hit_at_1", 0.0),
                "subsystem_hit_at_3": xai.get("subsystem_hit_at_3", 0.0),
            }
        )
    return rows


def write_comparison_artifacts(
    output_dir: Path,
    comparison: dict[str, Any],
) -> None:
    """Write comparison JSON and CSV with identical model variant keys."""

    write_json(output_dir / "comparison.json", comparison)
    write_csv(output_dir / "comparison.csv", comparison_rows(comparison))
