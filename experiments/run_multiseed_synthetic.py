"""Run isolated SAK synthetic experiments for multiple random seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sak.experiments.multiseed import run_multiseed_synthetic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multiseed"),
    )
    parser.add_argument(
        "--render-dashboards",
        action="store_true",
        help="Render a dashboard inside every seed directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pca", "dense_autoencoder", "tcn_autoencoder"),
        help="Optional model list; defaults to PCA and Dense Autoencoder.",
    )
    parser.add_argument(
        "--calibration",
        choices=("quantile", "constrained_event_f1"),
        help="Override threshold_selection.strategy from the config.",
    )
    parser.add_argument(
        "--score-transform",
        choices=("none", "log1p", "robust_zscore"),
        help="Override temporal_calibration.score_transform from the config.",
    )
    arguments = parser.parse_args()
    aggregate = run_multiseed_synthetic(
        config_path=arguments.config,
        output_dir=arguments.output,
        seeds=arguments.seeds,
        models=arguments.models,
        render_dashboards=arguments.render_dashboards,
        threshold_selection_strategy=arguments.calibration,
        temporal_score_transform=arguments.score_transform,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
