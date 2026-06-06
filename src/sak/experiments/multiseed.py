"""Sequential multi-seed execution and metric aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sak.experiments.artifacts import write_csv, write_json
from sak.experiments.runner import run_synthetic_experiment

METRIC_PATHS: dict[str, tuple[str, str]] = {
    "point_precision": ("point_metrics", "precision"),
    "point_recall": ("point_metrics", "recall"),
    "point_f1": ("point_metrics", "f1"),
    "event_precision": ("event_metrics", "precision"),
    "event_recall": ("event_metrics", "recall"),
    "event_f1": ("event_metrics", "f1"),
    "false_alarms_per_day": ("event_metrics", "false_alarms_per_day"),
    "detection_delay": ("event_metrics", "median_detection_delay_minutes"),
    "early_warning_time": ("event_metrics", "mean_early_warning_time_minutes"),
    "channel_hit_at_1": ("xai_metrics", "channel_hit_at_1"),
    "channel_hit_at_3": ("xai_metrics", "channel_hit_at_3"),
    "subsystem_hit_at_1": ("xai_metrics", "subsystem_hit_at_1"),
    "subsystem_hit_at_3": ("xai_metrics", "subsystem_hit_at_3"),
    "critical_window_hit_rate": ("xai_metrics", "critical_window_hit_rate"),
    "mean_critical_window_iou": ("xai_metrics", "mean_critical_window_iou"),
}


def _nested_metric(payload: dict[str, Any], path: tuple[str, str]) -> float | None:
    value = payload.get(path[0], {}).get(path[1])
    return float(value) if value is not None else None


def aggregate_seed_results(
    seed_results: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate mean and population standard deviation by model variant."""

    model_variants = sorted(
        {
            model_variant
            for _, comparison in seed_results
            for model_variant in comparison
            if model_variant != "dataset"
        }
    )
    rows: list[dict[str, Any]] = []
    for model_variant in model_variants:
        row: dict[str, Any] = {
            "model_variant": model_variant,
            "seed_count": len(seed_results),
        }
        for metric_name, metric_path in METRIC_PATHS.items():
            values = [
                value
                for _, comparison in seed_results
                if model_variant in comparison
                for value in [_nested_metric(comparison[model_variant], metric_path)]
                if value is not None
            ]
            row[f"{metric_name}_mean"] = (
                float(np.mean(values)) if values else None
            )
            row[f"{metric_name}_std"] = (
                float(np.std(values, ddof=0)) if values else None
            )
        rows.append(row)
    return rows


def run_multiseed_synthetic(
    *,
    config_path: Path,
    output_dir: Path,
    seeds: list[int],
    models: Sequence[str] | None = None,
    render_dashboards: bool = False,
) -> list[dict[str, Any]]:
    """Run isolated synthetic experiments and write aggregate CSV and JSON."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, dict[str, Any]]] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:03d}"
        comparison = run_synthetic_experiment(
            config_path=config_path,
            output_dir=seed_dir,
            seed=seed,
            models=models,
            generated_report_root=seed_dir / "reports" / "generated",
            data_output_dir=seed_dir / "data" / "synthetic",
            dashboard_path=seed_dir / "dashboard.html",
            render_dashboard=render_dashboards,
        )
        results.append((seed, comparison))

    aggregate_rows = aggregate_seed_results(results)
    write_csv(output_dir / "aggregate_results.csv", aggregate_rows)
    write_json(
        output_dir / "aggregate_results.json",
        {
            "seeds": seeds,
            "models": list(models) if models is not None else None,
            "results": aggregate_rows,
        },
    )
    return aggregate_rows
