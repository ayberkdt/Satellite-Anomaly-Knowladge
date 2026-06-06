"""CSV and Parquet telemetry data adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CsvParquetDataSource:
    """Load CSV or Parquet and enforce a monotonic datetime index."""

    path: Path
    timestamp_column: str = "timestamp"
    timezone: str = "UTC"

    def load(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(self.path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(self.path)
        else:
            raise ValueError(f"Unsupported telemetry format: {suffix}")

        if self.timestamp_column not in frame.columns:
            raise ValueError(f"Missing timestamp column: {self.timestamp_column}")

        timestamps = pd.to_datetime(frame.pop(self.timestamp_column), errors="raise")
        if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
            timestamps = timestamps.dt.tz_convert("UTC")
        else:
            timestamps = timestamps.dt.tz_localize(
                self.timezone,
                ambiguous="raise",
                nonexistent="raise",
            ).dt.tz_convert("UTC")
        frame.index = timestamps
        frame.index.name = self.timestamp_column
        frame = frame.sort_index()

        if frame.index.has_duplicates:
            raise ValueError("Telemetry timestamps must be unique after ingestion")

        return frame
