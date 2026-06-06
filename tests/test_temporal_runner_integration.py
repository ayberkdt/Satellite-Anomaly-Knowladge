import json
from pathlib import Path

import pandas as pd
import yaml

from sak.experiments.runner import run_synthetic_experiment


def test_temporal_model_runs_through_shared_pipeline(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load(
        (repository_root / "configs" / "synthetic_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    settings["synthetic"]["periods"] = 18000
    settings["synthetic"]["missing_fraction"] = 0.0
    settings["temporal_autoencoder"].update(
        {
            "window_size": 12,
            "stride": 12,
            "hidden_channels": 4,
            "latent_channels": 2,
            "kernel_size": 3,
            "num_layers": 1,
            "epochs": 1,
            "batch_size": 64,
            "patience": 1,
        }
    )
    settings["early_warning"].update(
        {
            "threshold_quantile": 0.9,
            "threshold_sweep_quantiles": [0.9],
            "minimum_hits": 1,
            "lookback_steps": 1,
            "merge_gap_steps": 2,
        }
    )
    settings["explainability"]["subsystem_mapping"] = str(
        repository_root / "configs" / "subsystems.yaml"
    )
    config_path = tmp_path / "temporal_test.yaml"
    config_path.write_text(
        yaml.safe_dump(settings, sort_keys=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    comparison = run_synthetic_experiment(
        config_path=config_path,
        output_dir=output_dir,
        models=["tcn_autoencoder"],
        generated_report_root=tmp_path / "reports",
        data_output_dir=tmp_path / "data",
        render_dashboard=False,
    )

    expected_variants = {
        "tcn_autoencoder_global",
        "tcn_autoencoder_mode_aware",
    }
    assert set(comparison) == {"dataset", *expected_variants}
    for variant in expected_variants:
        variant_dir = output_dir / variant
        assert (variant_dir / "model.pt").exists()
        assert (variant_dir / "metrics.json").exists()
        assert (
            variant_dir / "diagnostics" / "score_distribution.json"
        ).exists()
        assert (
            variant_dir / "diagnostics" / "false_positive_context.json"
        ).exists()
        assert (variant_dir / "diagnostics" / "filter_sweep.csv").exists()
        assert (
            variant_dir / "diagnostics" / "anomaly_type_performance.csv"
        ).exists()
        assert (variant_dir / "xai" / "explanations.json").exists()
        assert (variant_dir / "xai" / "temporal_error_summary.json").exists()
        assert (
            variant_dir / "plots" / "temporal_window_error_heatmap.png"
        ).exists()
        scores = pd.read_csv(variant_dir / "scores.csv")
        assert {
            "raw_score",
            "smoothed_score",
            "threshold",
            "alarm",
        } <= set(scores.columns)
        report_files = list((variant_dir / "reports").glob("SAK-*.json"))
        assert report_files
        report = json.loads(report_files[0].read_text(encoding="utf-8"))
        assert report["model_name"] == "tcn_autoencoder"
        assert (variant_dir / "reports" / f"{report['report_id']}.md").exists()
