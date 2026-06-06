"""Run the reproducible SAK synthetic model experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sak.experiments.runner import run_synthetic_experiment  # noqa: E402


def run(
    config_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
    models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Preserve the original experiment entry point."""

    return run_synthetic_experiment(
        config_path,
        output_dir,
        seed=seed,
        models=models,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_models"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pca", "dense_autoencoder"),
    )
    arguments = parser.parse_args()
    summary = run(
        arguments.config,
        arguments.output,
        seed=arguments.seed,
        models=arguments.models,
    )
    concise = {
        model_variant: {
            "event_f1": result["event_metrics"]["f1"],
            "event_recall": result["event_metrics"]["recall"],
            "false_alarms_per_day": result["event_metrics"][
                "false_alarms_per_day"
            ],
            "channel_hit_at_3": result["xai_metrics"]["channel_hit_at_3"],
        }
        for model_variant, result in summary.items()
        if model_variant != "dataset"
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
