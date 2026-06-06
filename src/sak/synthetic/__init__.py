"""Synthetic satellite telemetry generation and anomaly injection."""

from sak.synthetic.generator import (
    TELEMETRY_CHANNELS,
    InjectionRecord,
    SyntheticConfig,
    SyntheticDataset,
    generate_synthetic_telemetry,
)

__all__ = [
    "TELEMETRY_CHANNELS",
    "InjectionRecord",
    "SyntheticConfig",
    "SyntheticDataset",
    "generate_synthetic_telemetry",
]

