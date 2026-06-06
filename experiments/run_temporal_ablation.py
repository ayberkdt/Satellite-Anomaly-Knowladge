"""Run a bounded TCN architecture ablation over the configured parameter grid."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sak.experiments.artifacts import write_csv, write_json  # noqa: E402
from sak.experiments.runner import run_synthetic_experiment  # noqa: E402


def _configurations(settings: dict[str, Any]) -> list[dict[str, int]]:
    ablation = settings["temporal_ablation"]
    names = (
        "window_size",
        "stride",
        "latent_channels",
        "hidden_channels",
        "kernel_size",
        "num_layers",
    )
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(ablation[name] for name in names))
    ]


def run_temporal_ablation(
    *,
    config_path: Path,
    output_dir: Path,
    seeds: list[int],
    max_configurations: int = 6,
    epochs: int | None = None,
) -> list[dict[str, Any]]:
    """Run a deterministic bounded subset of the configured architecture grid."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if max_configurations < 1:
        raise ValueError("max_configurations must be positive")
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configurations = _configurations(settings)[:max_configurations]
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config_index, configuration in enumerate(configurations, start=1):
        for seed in seeds:
            run_settings = yaml.safe_load(yaml.safe_dump(settings))
            run_settings["temporal_autoencoder"].update(configuration)
            if epochs is not None:
                run_settings["temporal_autoencoder"]["epochs"] = epochs
                run_settings["temporal_autoencoder"]["patience"] = min(
                    int(run_settings["temporal_autoencoder"]["patience"]),
                    epochs,
                )
            run_settings["explainability"]["subsystem_mapping"] = str(
                REPOSITORY_ROOT / "configs" / "subsystems.yaml"
            )
            generated_config = (
                config_dir / f"config_{config_index:03d}_seed_{seed:03d}.yaml"
            )
            generated_config.write_text(
                yaml.safe_dump(run_settings, sort_keys=False),
                encoding="utf-8",
            )
            run_dir = output_dir / f"config_{config_index:03d}" / f"seed_{seed:03d}"
            started = time.perf_counter()
            comparison = run_synthetic_experiment(
                config_path=generated_config,
                output_dir=run_dir,
                seed=seed,
                models=["tcn_autoencoder"],
                generated_report_root=run_dir / "reports",
                data_output_dir=run_dir / "data",
                render_dashboard=False,
                threshold_selection_strategy="constrained_event_f1",
            )
            training_time = time.perf_counter() - started
            result = comparison["tcn_autoencoder_global"]
            event = result["event_metrics"]
            rows.append(
                {
                    "configuration_id": config_index,
                    "seed": seed,
                    **configuration,
                    "event_recall": event["recall"],
                    "critical_region_recall": event["critical_region_recall"],
                    "event_f1": event["f1"],
                    "false_alarms_per_day": event["false_alarms_per_day"],
                    "median_detection_delay": event[
                        "median_detection_delay_minutes"
                    ],
                    "median_lead_time_to_critical": event[
                        "median_lead_time_to_critical_minutes"
                    ],
                    "point_f1": result["point_metrics"]["f1"],
                    "channel_hit_at_3": result["xai_metrics"]["channel_hit_at_3"],
                    "training_time_seconds": training_time,
                    "epochs_trained": result["model"]["epochs_trained"],
                }
            )
    write_csv(output_dir / "results.csv", rows)
    write_json(
        output_dir / "results.json",
        {
            "seeds": seeds,
            "max_configurations": max_configurations,
            "results": rows,
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/temporal_ablation"),
    )
    parser.add_argument("--max-configurations", type=int, default=6)
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args()
    rows = run_temporal_ablation(
        config_path=arguments.config.resolve(),
        output_dir=arguments.output.resolve(),
        seeds=arguments.seeds,
        max_configurations=arguments.max_configurations,
        epochs=arguments.epochs,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
