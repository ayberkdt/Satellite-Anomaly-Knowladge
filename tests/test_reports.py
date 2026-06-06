from datetime import datetime, timezone

from sak.contracts import AlarmEvent, ChannelContribution, ExplanationResult
from sak.reporting import (
    build_early_warning_report_payload,
    render_early_warning_report_payload,
)


def _alarm_event() -> AlarmEvent:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    explanation = ExplanationResult(
        method="reconstruction_error_attribution",
        contributions=(
            ChannelContribution(
                channel="battery_voltage",
                contribution=0.8,
                subsystem="EPS",
                direction="low",
            ),
        ),
        critical_start=start,
        critical_end=start,
        possible_subsystems=("EPS",),
        confidence=0.8,
        notes=("Battery voltage reconstruction error increased.",),
    )
    return AlarmEvent(
        event_id="SAK-0001",
        start_time=start,
        end_time=start,
        peak_time=start,
        peak_score=2.5,
        threshold=1.0,
        risk_level="MEDIUM",
        explanation=explanation,
        context={"operational_mode": "nominal"},
    )


def test_json_report_contains_required_schema_and_matches_markdown() -> None:
    payload = build_early_warning_report_payload(
        _alarm_event(),
        model_name="pca",
        model_variant="pca_global",
        threshold_strategy="global",
        source_event_id="SYN-0001",
        metadata={"seed": 42},
    )
    markdown = render_early_warning_report_payload(payload)
    required = {
        "report_id",
        "model_name",
        "model_variant",
        "threshold_strategy",
        "explanation_method",
        "alarm_time",
        "event_id",
        "event_start",
        "event_end",
        "critical_window_start",
        "critical_window_end",
        "anomaly_score",
        "threshold",
        "risk_level",
        "top_channels",
        "possible_subsystems",
        "engineering_interpretation",
        "suggested_next_inspection",
        "confidence_or_uncertainty",
        "metadata",
    }

    assert required <= payload.keys()
    assert payload["top_channels"][0]["channel"] == "battery_voltage"
    assert payload["possible_subsystems"][0]["subsystem"] == "EPS"
    assert payload["report_id"] in markdown
    assert payload["model_variant"] in markdown
    assert payload["event_id"] in markdown
