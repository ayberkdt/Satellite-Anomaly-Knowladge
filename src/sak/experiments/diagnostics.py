"""Event and false-positive diagnostics with attribution-aware joins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


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
    frame: pd.DataFrame | None = None,
    true_events: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return unmatched events with contextual diagnostic hints.

    ``likely_reason`` is a heuristic diagnostic hint, not a root-cause claim.
    """

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
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        duration_minutes = (end - start).total_seconds() / 60.0 + 1.0
        peak_score = float(event.get("peak_score", 0.0))
        threshold = float(event.get("threshold", 0.0))
        ratio = peak_score / threshold if abs(threshold) > 1e-12 else None
        context = event.get("context", {})
        segment = (
            frame.loc[(frame.index >= start) & (frame.index <= end)]
            if frame is not None
            else None
        )
        mode_transition = bool(
            segment is not None
            and "operational_mode" in segment
            and segment["operational_mode"].nunique() > 1
        )
        eclipse_boundary = bool(
            segment is not None
            and "eclipse" in segment
            and segment["eclipse"].nunique() > 1
        )
        if mode_transition:
            likely_reason = "mode_transition"
        elif eclipse_boundary:
            likely_reason = "eclipse_boundary"
        elif ratio is not None and ratio <= 1.25:
            likely_reason = "threshold_margin_low"
        elif duration_minutes <= 2.0:
            likely_reason = "isolated_spike"
        elif duration_minutes >= 30.0 and (ratio is None or ratio < 1.5):
            likely_reason = "long_low_confidence_alarm"
        else:
            likely_reason = "unknown"

        nearest_id = ""
        nearest_distance: float | str = ""
        for truth in true_events:
            truth_start = pd.Timestamp(truth["start"])
            truth_end = pd.Timestamp(truth["end"])
            if end < truth_start:
                distance = (truth_start - end).total_seconds() / 60.0
            elif start > truth_end:
                distance = (start - truth_end).total_seconds() / 60.0
            else:
                distance = 0.0
            if nearest_distance == "" or distance < float(nearest_distance):
                nearest_id = str(truth["event_id"])
                nearest_distance = float(distance)

        top_channel_payload = event.get("top_channels", [])[:3]
        top_channels = [str(item["channel"]) for item in top_channel_payload]
        top_subsystems = [
            str(item["subsystem"])
            for item in event.get("top_subsystems", [])[:3]
            if isinstance(item, dict) and item.get("subsystem")
        ]
        if not top_subsystems:
            top_subsystems = list(
                dict.fromkeys(
                    str(item["subsystem"])
                    for item in top_channel_payload
                    if item.get("subsystem")
                )
            )
        rows.append(
            {
                "model_variant": model_variant,
                "predicted_event_id": normalized_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_minutes": duration_minutes,
                "peak_score": peak_score,
                "threshold_at_peak": threshold,
                "score_to_threshold_ratio": ratio,
                "risk_level": str(event["risk_level"]),
                "operational_mode": str(context.get("operational_mode", "")),
                "eclipse": bool(context.get("eclipse", False)),
                "orbit_phase": float(context.get("orbit_phase", 0.0)),
                "top_channels": ", ".join(top_channels),
                "top_subsystems": ", ".join(top_subsystems),
                "nearest_true_event_id": nearest_id,
                "distance_to_nearest_true_event_minutes": nearest_distance,
                "likely_reason": likely_reason,
            }
        )
    return rows


def _temporal_iou(
    first_start: pd.Timestamp,
    first_end: pd.Timestamp,
    second_start: pd.Timestamp,
    second_end: pd.Timestamp,
) -> float:
    intersection_start = max(first_start, second_start)
    intersection_end = min(first_end, second_end)
    intersection = max(
        0.0,
        (intersection_end - intersection_start).total_seconds() + 60.0,
    )
    union_start = min(first_start, second_start)
    union_end = max(first_end, second_end)
    union = max(60.0, (union_end - union_start).total_seconds() + 60.0)
    return intersection / union


def build_anomaly_type_performance_rows(
    *,
    model_variant: str,
    truth_events: Sequence[dict[str, Any]],
    event_metrics: dict[str, Any],
    predicted_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one detection, attribution and temporal-XAI row per anomaly."""

    predicted_by_id = _event_lookup(predicted_events)
    matches = {
        str(match["true_event_id"]): match
        for match in event_metrics.get("matches", [])
    }
    rows: list[dict[str, Any]] = []
    for truth in truth_events:
        true_event_id = str(truth["event_id"])
        expected_channels = [str(item) for item in truth["affected_channels"]]
        expected_subsystem = str(truth["expected_subsystem"])
        match = matches.get(true_event_id)
        if match is None:
            rows.append(
                {
                    "model_variant": model_variant,
                    "anomaly_type": str(truth["anomaly_type"]),
                    "true_event_id": true_event_id,
                    "detected": False,
                    "detection_delay_minutes": "",
                    "top_channels": "",
                    "expected_channels": ", ".join(expected_channels),
                    "channel_hit": False,
                    "expected_subsystem": expected_subsystem,
                    "subsystem_hit": False,
                    "critical_window_hit": False,
                    "critical_window_iou": 0.0,
                }
            )
            continue

        event = predicted_by_id.get(str(match["predicted_event_id"]))
        if event is None:
            raise ValueError(
                f"{model_variant} match references missing event "
                f"{match['predicted_event_id']}"
            )
        top_channel_payload = event.get("top_channels", [])[:3]
        top_channels = [str(item["channel"]) for item in top_channel_payload]
        top_subsystems = {
            str(item["subsystem"])
            for item in top_channel_payload
            if item.get("subsystem")
        }
        critical_start = pd.Timestamp(event["critical_window_start"])
        critical_end = pd.Timestamp(event["critical_window_end"])
        truth_start = pd.Timestamp(truth["start"])
        truth_end = pd.Timestamp(truth["end"])
        critical_hit = critical_end >= truth_start and critical_start <= truth_end
        rows.append(
            {
                "model_variant": model_variant,
                "anomaly_type": str(truth["anomaly_type"]),
                "true_event_id": true_event_id,
                "detected": True,
                "detection_delay_minutes": float(
                    match["detection_delay_minutes"]
                ),
                "top_channels": ", ".join(top_channels),
                "expected_channels": ", ".join(expected_channels),
                "channel_hit": bool(set(expected_channels) & set(top_channels)),
                "expected_subsystem": expected_subsystem,
                "subsystem_hit": expected_subsystem in top_subsystems,
                "critical_window_hit": critical_hit,
                "critical_window_iou": _temporal_iou(
                    critical_start,
                    critical_end,
                    truth_start,
                    truth_end,
                ),
            }
        )
    return rows
