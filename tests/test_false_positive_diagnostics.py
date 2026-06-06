import pandas as pd
import pytest

from sak.experiments.diagnostics import build_false_positive_rows


def test_false_positive_context_includes_ratio_nearest_event_and_hint() -> None:
    index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=10,
        freq="1min",
    )
    frame = pd.DataFrame(
        {
            "operational_mode": ["nominal"] * 10,
            "eclipse": [False] * 10,
            "orbit_phase": [0.1] * 10,
        },
        index=index,
    )
    predictions = [
        {
            "event_id": "SAK-0001",
            "start": index[2].isoformat(),
            "end": index[3].isoformat(),
            "peak_score": 1.2,
            "threshold": 1.0,
            "risk_level": "LOW",
            "context": {
                "operational_mode": "nominal",
                "eclipse": False,
                "orbit_phase": 0.1,
            },
            "top_channels": [
                {"channel": "battery_voltage", "subsystem": "EPS"}
            ],
            "top_subsystems": [{"subsystem": "EPS"}],
        }
    ]
    truth = [
        {
            "event_id": "SYN-0001",
            "start": index[6].isoformat(),
            "end": index[7].isoformat(),
        }
    ]

    rows = build_false_positive_rows(
        model_variant="tcn_autoencoder_global",
        event_metrics={"matches": []},
        predicted_events=predictions,
        frame=frame,
        true_events=truth,
    )

    assert rows[0]["score_to_threshold_ratio"] == pytest.approx(1.2)
    assert rows[0]["nearest_true_event_id"] == "SYN-0001"
    assert rows[0]["distance_to_nearest_true_event_minutes"] == 3.0
    assert rows[0]["likely_reason"] == "threshold_margin_low"
