"""Dataset-independent telemetry adapter API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class TelemetryDataset:
    """Canonical dataset returned by synthetic and future open-data adapters."""

    frame: pd.DataFrame
    channel_names: tuple[str, ...]
    context_columns: tuple[str, ...]
    events: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryDatasetAdapter(Protocol):
    """Load one source into the canonical telemetry dataset contract."""

    def load(self, path: Path) -> TelemetryDataset:
        """Load telemetry and event metadata from ``path``."""
        ...
