"""Telemetry plots, score traces and attribution heatmaps."""

from sak.visualization.diagnostics import (
    plot_error_heatmap,
    plot_score_timeline,
    plot_temporal_window_error_heatmap,
)
from sak.visualization.theme import (
    LABEL_TAXONOMY_COLORS,
    MODEL_COLORS,
    RISK_COLORS,
    SUBSYSTEM_COLORS,
    SUBSYSTEM_LABELS,
    is_hex_color,
    subsystem_color,
    subsystem_label,
)

__all__ = [
    "LABEL_TAXONOMY_COLORS",
    "MODEL_COLORS",
    "RISK_COLORS",
    "SUBSYSTEM_COLORS",
    "SUBSYSTEM_LABELS",
    "is_hex_color",
    "plot_error_heatmap",
    "plot_score_timeline",
    "plot_temporal_window_error_heatmap",
    "subsystem_color",
    "subsystem_label",
]
