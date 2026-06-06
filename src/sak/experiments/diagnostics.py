"""Event and false-positive diagnostics with attribution-aware joins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _event_lookup(events: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event["event_id"])
        lookup[event_id] = event
        if "-SAK-" in event_id:
            lookup[f"SAK-{event_id.rsplit('-SAK-', maxsplit=1)[1]}"] = event
    return lookup


def build_event_diagnostic_rows(
    *,
    model_variant: str,
    truth_events: Sequence[dict[str, Any]],
    event_metrics: dict[str, Any],
    predicted_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join truth, event matches and the exact variant's XAI attribution."""

    predicted_by_id = _event_lookup(predicted_events)
    matches = {
        str(match["true_event_id"]): match
        for match in event_metrics.get("matches", [])
    }
    rows: list[dict[str, Any]] = []
    for truth in truth_events:
        true_event_id = str(truth["event_id"])
        match = matches.get(true_event_id)
        if match is None:
            rows.append(
                {
                    "model_variant": model_variant,
                    "true_event_id": true_event_id,
                    "anomaly_type": str(truth["anomaly_type"]),
                    "expected_subsystem": str(truth["expected_subsystem"]),
                    "predicted_event_id": "MISS",
                    "detection_delay_min": "",
                    "top_channels": "",
                    "top_subsystems": "",
                    "channel_hit": "no",
                    "subsystem_hit": "no",
                }
            )
            continue

        predicted_event_id = str(match["predicted_event_id"])
        event = predicted_by_id.get(predicted_event_id)
        if event is None:
            raise ValueError(
                f"{model_variant} match references missing event {predicted_event_id}"
            )
        top_channel_payload = event.get("top_channels", [])[:3]
        top_channels = [str(item["channel"]) for item in top_channel_payload]
        top_subsystems = list(
            dict.fromkeys(
                str(item["subsystem"])
                for item in top_channel_payload
                if item.get("subsystem")
            )
        )
        expected_channels = {str(item) for item in truth["affected_channels"]}
        expected_subsystem = str(truth["expected_subsystem"])
        rows.append(
            {
                "model_variant": model_variant,
                "true_event_id": true_event_id,
                "anomaly_type": str(truth["anomaly_type"]),
                "expected_subsystem": expected_subsystem,
                "predicted_event_id": predicted_event_id,
                "detection_delay_min": float(match["detection_delay_minutes"]),
                "top_channels": ", ".join(top_channels),
                "top_subsystems": ", ".join(top_subsystems),
                "channel_hit": (
                    "yes" if expected_channels.intersection(top_channels) else "no"
                ),
                "subsystem_hit": (
                    "yes" if expected_subsystem in top_subsystems else "no"
                ),
            }
        )
    return rows


def build_false_positive_rows(
    *,
    model_variant: str,
    event_metrics: dict[str, Any],
    predicted_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return predicted events that were not matched to a truth event."""

    matched = {
        str(match["predicted_event_id"])
        for match in event_metrics.get("matches", [])
    }
    rows: list[dict[str, Any]] = []
    for event in predicted_events:
        event_id = str(event["event_id"])
        normalized_id = (
            f"SAK-{event_id.rsplit('-SAK-', maxsplit=1)[1]}"
            if "-SAK-" in event_id
            else event_id
        )
        if normalized_id in matched:
            continue
        rows.append(
            {
                "model_variant": model_variant,
                "predicted_event_id": normalized_id,
                "start": str(event["start"]),
                "end": str(event["end"]),
                "risk_level": str(event["risk_level"]),
                "operational_mode": str(
                    event.get("context", {}).get("operational_mode", "")
                ),
                "top_channels": ", ".join(
                    str(item["channel"])
                    for item in event.get("top_channels", [])[:3]
                ),
            }
        )
    return rows
