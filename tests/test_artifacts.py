import json
from pathlib import Path

from sak.experiments.artifacts import (
    create_variant_artifact_paths,
    model_variant_name,
)
from sak.experiments.comparison import write_comparison_artifacts


def test_model_variant_path_uses_comparison_key(tmp_path: Path) -> None:
    model_variant = model_variant_name("pca", "mode_aware")

    paths = create_variant_artifact_paths(tmp_path, model_variant)

    assert model_variant == "pca_mode_aware"
    assert paths.root == tmp_path / model_variant
    assert paths.reports.is_dir()
    assert paths.xai.is_dir()
    assert paths.plots.is_dir()


def test_comparison_json_and_csv_use_same_model_keys(tmp_path: Path) -> None:
    comparison = {
        "dataset": {"rows": 10},
        "pca_global": {
            "point_metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "event_metrics": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "false_alarms_per_day": 0.0,
                "median_detection_delay_minutes": 2.0,
            },
            "xai_metrics": {
                "channel_hit_at_1": 1.0,
                "channel_hit_at_3": 1.0,
                "subsystem_hit_at_1": 1.0,
                "subsystem_hit_at_3": 1.0,
            },
        },
    }

    write_comparison_artifacts(tmp_path, comparison)

    json_payload = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf-8")
    )
    csv_text = (tmp_path / "comparison.csv").read_text(encoding="utf-8")
    assert "pca_global" in json_payload
    assert "pca_global" in csv_text
