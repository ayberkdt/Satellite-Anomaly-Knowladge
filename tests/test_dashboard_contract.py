import json
from pathlib import Path

from sak.reporting import render_synthetic_dashboard


def _model_payload() -> dict:
    return {
        "point_metrics": {"f1": 0.7},
        "event_metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "critical_region_recall": 1.0,
            "detected_before_critical_rate": 1.0,
            "f1": 1.0,
            "false_alarms_per_day": 0.0,
            "median_detection_delay_minutes": 2.0,
            "median_lead_time_to_critical_minutes": 12.0,
            "matches": [
                {
                    "true_event_id": "SYN-0001",
                    "predicted_event_id": "SAK-0001",
                    "detection_delay_minutes": 2.0,
                    "lead_time_to_critical_region_minutes": 12.0,
                    "detected_before_critical_region": True,
                    "critical_region_covered": True,
                }
            ],
        },
        "xai_metrics": {
            "channel_hit_at_3": 1.0,
            "subsystem_hit_at_2": 1.0,
            "critical_window_hit_rate": 1.0,
        },
        "threshold_sweep": [],
    }


def test_dashboard_contract_handles_missing_optional_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    for model in ("pca_global", "dense_autoencoder_global"):
        model_dir = artifact_dir / model
        (model_dir / "plots").mkdir(parents=True)
        (model_dir / "plots" / "score_timeline.png").write_bytes(b"fake")
        (model_dir / "plots" / "channel_error_heatmap.png").write_bytes(b"fake")
        (model_dir / "events.json").write_text(
            json.dumps(
                [
                    {
                        "event_id": "SAK-0001",
                        "start": "2026-01-01T00:00:00Z",
                        "end": "2026-01-01T00:05:00Z",
                        "peak_time": "2026-01-01T00:02:00Z",
                        "peak_score": 1.2,
                        "risk_level": "HIGH",
                        "context": {"operational_mode": "nominal"},
                        "top_channels": [
                            {
                                "channel": "battery_voltage",
                                "subsystem": "EPS",
                                "contribution": 0.42,
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

    manifest = tmp_path / "injection_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "event_id": "SYN-0001",
                    "partition": "test",
                    "event_class": "anomaly",
                    "anomaly_type": "battery_voltage_sag",
                    "affected_channels": ["battery_voltage"],
                    "expected_subsystem": "EPS",
                }
            ]
        ),
        encoding="utf-8",
    )
    comparison = {
        "dataset": {
            "rows": 100,
            "channels": 3,
            "train_rows": 50,
            "calibration_rows": 20,
            "validation_rows": 10,
            "test_rows": 20,
            "test_events": 1,
        },
        "pca_global": _model_payload(),
        "dense_autoencoder_global": _model_payload(),
        "tcn_autoencoder_global": _model_payload(),
        "tcn_autoencoder_mode_aware": _model_payload(),
    }
    dashboard_path = tmp_path / "dashboard.html"

    render_synthetic_dashboard(
        comparison=comparison,
        artifact_dir=artifact_dir,
        dashboard_path=dashboard_path,
        manifest_path=manifest,
    )

    html = dashboard_path.read_text(encoding="utf-8")
    assert "Subsystem Legend" in html
    assert "EPS" in html
    assert "subsystem-badge" in html
    assert "Advanced / Legacy" in html
    assert "Data Contract Warnings" in html
