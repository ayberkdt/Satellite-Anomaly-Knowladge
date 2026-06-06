"""Run SAK-v3.0 real dataset benchmark models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sak.data import AdapterDataNotFoundError, UnsupportedDatasetLayoutError  # noqa: E402
from sak.experiments.dataset_runner import run_real_dataset_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=["nasa_smap_msl"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/real/nasa_smap_msl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_experiment.yaml"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pca", "dense_autoencoder", "tcn_autoencoder"),
        default=("pca",),
    )
    parser.add_argument("--channel-id")
    parser.add_argument(
        "--calibration",
        choices=("quantile", "constrained_event_f1"),
        default="constrained_event_f1",
    )
    parser.add_argument(
        "--score-transform",
        choices=("none", "log1p", "robust_zscore"),
        default="none",
    )
    parser.add_argument("--max-channels", type=int)
    parser.add_argument("--render-dashboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        summary = run_real_dataset_experiment(
            adapter_name=arguments.adapter,
            data_path=arguments.data,
            output_dir=arguments.output,
            config_path=arguments.config,
            models=arguments.models,
            channel_id=arguments.channel_id,
            calibration=arguments.calibration,
            score_transform=arguments.score_transform,
            max_channels=arguments.max_channels,
            render_dashboard=arguments.render_dashboard,
        )
    except (AdapterDataNotFoundError, UnsupportedDatasetLayoutError, ValueError) as error:
        print(f"Real dataset run could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    concise = {
        model_variant: {
            "event_f1": payload["event_metrics"]["f1"],
            "event_recall": payload["event_metrics"]["recall"],
            "false_alarms_per_day": payload["event_metrics"]["false_alarms_per_day"],
            "test_used_for_selection": payload["calibration"][
                "test_partition_used_for_selection"
            ],
            "selection_reason": payload["calibration"]["selection_reason"],
        }
        for model_variant, payload in summary.items()
        if model_variant != "dataset"
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
