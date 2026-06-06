"""Adapter for generated SAK synthetic telemetry artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sak.data.adapters.base import TelemetryDataset
from sak.synthetic import TELEMETRY_CHANNELS


class SyntheticTelemetryAdapter:
    """Load telemetry.csv and its sibling injection manifest."""

    def load(self, path: Path) -> TelemetryDataset:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        manifest_path = path.with_name("injection_manifest.json")
        events = (
            tuple(json.loads(manifest_path.read_text(encoding="utf-8")))
            if manifest_path.exists()
            else ()
        )
        context_columns = tuple(
            column
            for column in (
                "orbit_phase",
                "eclipse",
                "sunlight",
                "beta_angle",
                "operational_mode",
                "maneuver_flag",
                "safe_mode_flag",
            )
            if column in frame
        )
        channels = tuple(channel for channel in TELEMETRY_CHANNELS if channel in frame)
        return TelemetryDataset(
            frame=frame,
            channel_names=channels,
            context_columns=context_columns,
            events=events,
            metadata={"source": "sak_synthetic", "path": str(path)},
        )
