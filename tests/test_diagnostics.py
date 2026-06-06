from sak.experiments.diagnostics import (
    build_event_diagnostic_rows,
    build_false_positive_rows,
)


def test_event_diagnostics_join_variant_attribution() -> None:
    truth = [
        {
            "event_id": "SYN-0001",
            "anomaly_type": "voltage_drop",
            "affected_channels": ["battery_voltage"],
            "expected_subsystem": "EPS",
        }
    ]
    metrics = {
        "matches": [
            {
                "true_event_id": "SYN-0001",
                "predicted_event_id": "SAK-0001",
                "detection_delay_minutes": -2.0,
            }
        ]
    }
    predictions = [
        {
            "event_id": "SAK-0001",
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-01T00:05:00+00:00",
            "risk_level": "HIGH",
            "context": {"operational_mode": "nominal"},
            "top_channels": [
                {
                    "channel": "battery_voltage",
                    "subsystem": "EPS",
                    "contribution": 0.8,
                }
            ],
        }
    ]

    rows = build_event_diagnostic_rows(
        model_variant="pca_global",
        truth_events=truth,
        event_metrics=metrics,
        predicted_events=predictions,
    )

    assert rows[0]["top_channels"] == "battery_voltage"
    assert rows[0]["top_subsystems"] == "EPS"
    assert rows[0]["channel_hit"] == "yes"
    assert rows[0]["subsystem_hit"] == "yes"


def test_false_positive_diagnostics_exclude_matched_events() -> None:
    metrics = {
        "matches": [
            {
                "true_event_id": "SYN-0001",
                "predicted_event_id": "SAK-0001",
            }
        ]
    }
    predictions = [
        {
            "event_id": event_id,
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-01T00:05:00+00:00",
            "risk_level": "LOW",
            "context": {},
            "top_channels": [],
        }
        for event_id in ("SAK-0001", "SAK-0002")
    ]

    rows = build_false_positive_rows(
        model_variant="pca_global",
        event_metrics=metrics,
        predicted_events=predictions,
    )

    assert [row["predicted_event_id"] for row in rows] == ["SAK-0002"]


def test_variant_prefixed_event_id_joins_and_uses_canonical_subsystems() -> None:
    truth = [
        {
            "event_id": "SYN-0001",
            "anomaly_type": "battery_degradation",
            "affected_channels": ["battery_voltage"],
            "expected_subsystem": "EPS",
        }
    ]
    metrics = {
        "matches": [
            {
                "true_event_id": "SYN-0001",
                "predicted_event_id": "SAK-0001",
                "detection_delay_minutes": 1.0,
            }
        ]
    }
    predictions = [
        {
            "event_id": "pca_global-SAK-0001",
            "top_channels": [{"channel": "battery_voltage", "subsystem": None}],
            "top_subsystems": [{"subsystem": "EPS", "contribution": 1.0}],
        }
    ]

    rows = build_event_diagnostic_rows(
        model_variant="pca_global",
        truth_events=truth,
        event_metrics=metrics,
        predicted_events=predictions,
    )

    assert rows[0]["predicted_event_id"] == "SAK-0001"
    assert rows[0]["top_subsystems"] == "EPS"
    assert rows[0]["subsystem_hit"] == "yes"
