import json
from pathlib import Path

from sak.reporting import experiment_model_rows, render_synthetic_dashboard


def _comparison_payload() -> dict:
    base_model = {
        "point_metrics": {"f1": 0.7},
        "event_metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "false_alarms_per_day": 0.0,
            "median_detection_delay_minutes": 2.0,
            "matches": [
                {
                    "true_event_id": "SYN-0001",
                    "predicted_event_id": "SAK-0001",
                    "detection_delay_minutes": 2.0,
                }
            ],
        },
        "xai_metrics": {
            "channel_hit_at_3": 1.0,
            "subsystem_hit_at_2": 1.0,
            "critical_window_hit_rate": 1.0,
        },
        "threshold_sweep": [
            {
                "quantile": 0.995,
                "threshold": 0.1,
                "event_precision": 1.0,
                "event_recall": 1.0,
                "event_f1": 1.0,
                "false_alarms_per_day": 0.0,
                "median_detection_delay_minutes": 2.0,
            }
        ],
    }
    return {
        "dataset": {
            "rows": 100,
            "channels": 3,
            "train_rows": 60,
            "validation_rows": 20,
            "test_rows": 20,
            "test_events": 1,
        },
        "pca": base_model,
        "dense_autoencoder": base_model,
    }


def test_model_rows_are_flattened_for_tables() -> None:
    rows = experiment_model_rows(_comparison_payload())

    assert rows[0]["model"] == "PCA"
    assert rows[0]["event_f1"] == 1.0
    assert rows[1]["model"] == "Dense Autoencoder"


def test_static_dashboard_is_rendered(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "synthetic_models"
    for model in ("pca", "dense_autoencoder"):
        model_dir = artifact_dir / model
        model_dir.mkdir(parents=True)
        (model_dir / "score_timeline.png").write_bytes(b"fake")
        (model_dir / "channel_error_heatmap.png").write_bytes(b"fake")
        (model_dir / "events.json").write_text(
            json.dumps(
                [
                    {
                        "event_id": f"{model}-event",
                        "start": "2026-01-01T00:00:00Z",
                        "end": "2026-01-01T00:05:00Z",
                        "peak_time": "2026-01-01T00:02:00Z",
                        "peak_score": 1.2,
                        "risk_level": "HIGH",
                        "context": {"mode": "nominal"},
                        "top_channels": [
                            {
                                "channel": "battery_voltage",
                                "subsystem": "EPS",
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

    dashboard_path = tmp_path / "dashboards" / "sak_synthetic_dashboard.html"
    render_synthetic_dashboard(
        comparison=_comparison_payload(),
        artifact_dir=artifact_dir,
        dashboard_path=dashboard_path,
    )

    assert dashboard_path.exists()
    assert "SAK Synthetic Experiment Dashboard" in dashboard_path.read_text(
        encoding="utf-8"
    )
    assert (artifact_dir / "dashboard" / "model_comparison.csv").exists()
    assert (artifact_dir / "dashboard" / "model_comparison.png").exists()
    assert (artifact_dir / "dashboard" / "event_diagnostics.csv").exists()
    assert (artifact_dir / "dashboard" / "event_diagnostics.png").exists()
    assert (artifact_dir / "dashboard" / "channel_summary.csv").exists()
    assert (artifact_dir / "dashboard" / "subsystem_summary.png").exists()
