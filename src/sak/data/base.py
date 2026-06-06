"""Dataset-independent telemetry source interface."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class TelemetryDataSource(Protocol):
    """Load telemetry into the canonical tabular representation."""

    def load(self) -> pd.DataFrame:
        """Return a timestamp-indexed telemetry frame."""
        ...

