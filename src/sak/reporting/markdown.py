"""Render model-independent SAK reports as Markdown and structured JSON."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sak.contracts import AlarmEvent


def build_early_warning_report_payload(
    event: AlarmEvent,
    *,
    model_name: str = "unknown",
    model_variant: str = "unknown",
    threshold_strategy: str = "unknown",
    source_event_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical payload used by both Markdown and JSON reports."""

    explanation = event.explanation
    if explanation is None:
        top_channels: list[dict[str, Any]] = []
        possible_subsystems: list[dict[str, Any]] = []
        critical_start = None
        critical_end = None
        interpretation = "Model explanation is not available."
        confidence = "Attribution confidence is not available."
        explanation_method = "unavailable"
    else:
        top_channels = [
            {
                "channel": item.channel,
                "contribution": float(item.contribution),
                "subsystem": item.subsystem,
                "direction": item.direction,
            }
            for item in explanation.contributions
        ]
        subsystem_scores: dict[str, float] = {}
        for item in explanation.contributions:
            if item.subsystem:
                subsystem_scores[item.subsystem] = (
                    subsystem_scores.get(item.subsystem, 0.0)
                    + float(item.contribution)
                )
        possible_subsystems = [
            {"subsystem": subsystem, "contribution": contribution}
            for subsystem, contribution in sorted(
                subsystem_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        critical_start = explanation.critical_start.isoformat()
        critical_end = explanation.critical_end.isoformat()
        interpretation = " ".join(explanation.notes) or "Expert review is required."
        confidence = (
            f"Top attribution concentration is {explanation.confidence:.1%}; "
            "model probability is not calibrated."
            if explanation.confidence is not None
            else "Model probability and attribution confidence are not calibrated."
        )
        explanation_method = explanation.method

    report_metadata = {
        "dataset": "synthetic",
        "seed": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "SAK-v2.2",
        **(metadata or {}),
    }
    return {
        "report_id": event.event_id,
        "model_name": model_name,
        "model_variant": model_variant,
        "threshold_strategy": threshold_strategy,
        "explanation_method": explanation_method,
        "alarm_time": event.peak_time.isoformat(),
        "event_id": source_event_id or event.event_id,
        "event_start": event.start_time.isoformat(),
        "event_end": event.end_time.isoformat(),
        "critical_window_start": critical_start,
        "critical_window_end": critical_end,
        "anomaly_score": float(event.peak_score),
        "threshold": float(event.threshold),
        "risk_level": event.risk_level,
        "top_channels": top_channels,
        "possible_subsystems": possible_subsystems,
        "engineering_interpretation": interpretation,
        "suggested_next_inspection": (
            "Validate subsystem limits and state telemetry, compare nominal samples "
            "from the same operating context, and align command and maintenance logs "
            "with the critical window."
        ),
        "confidence_or_uncertainty": confidence,
        "operational_context": dict(event.context),
        "metadata": report_metadata,
    }


def render_early_warning_report(
    event: AlarmEvent,
    *,
    model_name: str = "unknown",
    model_variant: str = "unknown",
    threshold_strategy: str = "unknown",
    source_event_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render an alarm event from the same payload exported as JSON."""

    report = build_early_warning_report_payload(
        event,
        model_name=model_name,
        model_variant=model_variant,
        threshold_strategy=threshold_strategy,
        source_event_id=source_event_id,
        metadata=metadata,
    )
    return render_early_warning_report_payload(report)


def render_early_warning_report_payload(report: dict[str, Any]) -> str:
    """Render a previously built report payload without recomputing fields."""

    channel_lines = "\n".join(
        f"- {item['channel']}: {item['contribution']:.3f}"
        + (f" ({item['subsystem']})" if item["subsystem"] else "")
        + (f" [{item['direction']}]" if item["direction"] else "")
        for item in report["top_channels"]
    ) or "- Explanation is not available."
    subsystem = ", ".join(
        f"{item['subsystem']} ({item['contribution']:.3f})"
        for item in report["possible_subsystems"]
    ) or "Not determined"
    critical_window = (
        f"{report['critical_window_start']} - {report['critical_window_end']}"
        if report["critical_window_start"]
        else "Not determined"
    )
    context = ", ".join(
        f"{key}={value}" for key, value in report["operational_context"].items()
    ) or "Not determined"

    return f"""# SAK Early Warning Report

- **Report ID:** {report["report_id"]}
- **Event ID:** {report["event_id"]}
- **Model variant:** {report["model_variant"]}
- **Threshold strategy:** {report["threshold_strategy"]}
- **Alarm time:** {report["alarm_time"]}
- **Event interval:** {report["event_start"]} - {report["event_end"]}
- **Anomaly score:** {report["anomaly_score"]:.4f}
- **Threshold:** {report["threshold"]:.4f}
- **Risk level:** {report["risk_level"]}
- **Operational context:** {context}
- **Critical time window:** {critical_window}
- **Possible subsystem:** {subsystem}
- **Model confidence / uncertainty:** {report["confidence_or_uncertainty"]}

## Top Contributing Telemetry Channels

{channel_lines}

## Engineering Interpretation

{report["engineering_interpretation"]}

## Suggested Next Inspection

{report["suggested_next_inspection"]}
"""
