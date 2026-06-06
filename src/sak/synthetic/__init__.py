"""Synthetic satellite telemetry generation and anomaly injection."""

from sak.synthetic.generator import (
    ANOMALY_TYPES,
    CHANNEL_GROUPS,
    DEFAULT_ANOMALY_SCHEDULE,
    TELEMETRY_CHANNELS,
    InjectionRecord,
    SyntheticConfig,
    SyntheticDataset,
    generate_synthetic_telemetry,
)

__all__ = [
    "ANOMALY_TYPES",
    "CHANNEL_GROUPS",
    "DEFAULT_ANOMALY_SCHEDULE",
    "TELEMETRY_CHANNELS",
    "InjectionRecord",
    "SyntheticConfig",
    "SyntheticDataset",
    "generate_synthetic_telemetry",
]
