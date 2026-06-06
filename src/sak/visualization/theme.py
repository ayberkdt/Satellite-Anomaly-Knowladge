"""Shared visualization theme constants for SAK reports and dashboards."""

from __future__ import annotations

import re

SUBSYSTEM_COLORS = {
    "EPS": "#F59E0B",
    "THERMAL": "#EF4444",
    "AOCS": "#3B82F6",
    "COMM": "#8B5CF6",
    "PAYLOAD": "#10B981",
    "UNKNOWN": "#6B7280",
}

SUBSYSTEM_LABELS = {
    "EPS": "EPS",
    "THERMAL": "Thermal",
    "AOCS": "AOCS",
    "ADCS": "AOCS",
    "COMM": "Comm",
    "PAYLOAD": "Payload",
    "UNKNOWN": "Unknown",
}

RISK_COLORS = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

LABEL_TAXONOMY_COLORS = {
    "nominal": "#E5E7EB",
    "benign_transient": "#93C5FD",
    "precursor": "#FDE68A",
    "anomaly": "#FDBA74",
    "critical": "#EF4444",
}

MODEL_COLORS = {
    "pca_global": "#64748B",
    "dense_autoencoder_global": "#14B8A6",
    "tcn_autoencoder_global": "#2563EB",
    "pca_mode_aware": "#94A3B8",
    "dense_autoencoder_mode_aware": "#5EEAD4",
    "tcn_autoencoder_mode_aware": "#60A5FA",
}

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def subsystem_color(subsystem: object) -> str:
    """Return a stable color for a subsystem name."""

    key = str(subsystem or "UNKNOWN").upper()
    if key == "ADCS":
        key = "AOCS"
    return SUBSYSTEM_COLORS.get(key, SUBSYSTEM_COLORS["UNKNOWN"])


def subsystem_label(subsystem: object) -> str:
    """Return a normalized display label for a subsystem name."""

    key = str(subsystem or "UNKNOWN").upper()
    if key == "ADCS":
        key = "AOCS"
    return SUBSYSTEM_LABELS.get(key, key if key else "Unknown")


def is_hex_color(value: str) -> bool:
    """Validate CSS hex color literals used by the static dashboard."""

    return bool(_HEX_PATTERN.fullmatch(value))
