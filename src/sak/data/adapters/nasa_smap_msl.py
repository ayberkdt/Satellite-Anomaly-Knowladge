"""NASA SMAP/MSL adapter preparation utilities.

Expected layouts supported at inspection level:

- Telemanom-style root with ``train/``, ``test/`` or ``data/`` arrays and an
  optional ``labeled_anomalies.csv`` file.
- A normalized SAK staging folder with ``telemetry.csv`` or ``telemetry.parquet``
  plus optional interval labels.

Full benchmark normalization is intentionally conservative; unknown layouts
return an explanatory inspection report instead of pretending to load data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sak.data.adapters.base import TelemetryDataset


class AdapterDataNotFoundError(FileNotFoundError):
    """Raised when an adapter input path does not exist."""


class NasaSmapMslAdapter:
    """Inspect and load normalized NASA/JPL SMAP-MSL telemetry folders."""

    label_filenames = (
        "labeled_anomalies.csv",
        "labels.csv",
        "anomaly_intervals.csv",
    )
    telemetry_filenames = ("telemetry.parquet", "telemetry.csv")

    def validate_path(self, path: Path) -> dict[str, Any]:
        """Validate a candidate dataset path and return an inspection report."""

        return self.inspect(path)

    def inspect(self, path: Path) -> dict[str, Any]:
        """Inspect available files without requiring the full benchmark mapping."""

        root = Path(path)
        if not root.exists():
            raise AdapterDataNotFoundError(
                f"NASA SMAP/MSL dataset path does not exist: {root}"
            )
        if not root.is_dir():
            raise ValueError(f"NASA SMAP/MSL path must be a directory: {root}")

        telemetry_files = [
            str(root / name) for name in self.telemetry_filenames if (root / name).exists()
        ]
        label_files = [
            str(root / name) for name in self.label_filenames if (root / name).exists()
        ]
        array_files = sorted(str(path) for path in root.rglob("*.npy"))
        csv_files = sorted(str(path) for path in root.rglob("*.csv"))
        recognized_layout = bool(
            telemetry_files
            or array_files
            or (root / "train").exists()
            or (root / "test").exists()
            or (root / "data").exists()
        )
        channel_count: int | None = None
        if telemetry_files:
            frame = self._read_telemetry(Path(telemetry_files[0]), nrows=5)
            channel_count = len(self._channel_names(frame))
        event_count: int | None = None
        if label_files:
            event_count = len(pd.read_csv(label_files[0]))
        return {
            "adapter": "nasa_smap_msl",
            "path": str(root),
            "exists": True,
            "recognized_layout": recognized_layout,
            "telemetry_files": telemetry_files,
            "label_files": label_files,
            "array_file_count": len(array_files),
            "csv_file_count": len(csv_files),
            "channel_count": channel_count,
            "labels_available": bool(label_files),
            "event_count": event_count,
            "load_supported": bool(telemetry_files),
            "notes": (
                "telemetry.csv/parquet can be loaded directly; raw Telemanom "
                ".npy arrays still require channel-name and sequence mapping."
            ),
        }

    def load(self, path: Path) -> TelemetryDataset:
        """Load a normalized telemetry.csv/parquet staging folder."""

        report = self.inspect(path)
        telemetry_files = report["telemetry_files"]
        if not telemetry_files:
            raise NotImplementedError(
                "NASA SMAP/MSL raw array loading is not implemented yet. Stage "
                "the dataset as telemetry.csv/parquet or add channel-name and "
                "sequence mapping for the Telemanom .npy layout."
            )
        telemetry_path = Path(telemetry_files[0])
        frame = self._read_telemetry(telemetry_path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.date_range(
                "1970-01-01T00:00:00Z",
                periods=len(frame),
                freq="1min",
            )
            frame.index.name = "timestamp"

        channels = self._channel_names(frame)
        context_columns = tuple(
            column
            for column in ("spacecraft", "channel_id", "sequence_id", "split")
            if column in frame
        )
        events = self._load_events(Path(path), frame.index)
        return TelemetryDataset(
            frame=frame,
            channel_names=channels,
            context_columns=context_columns,
            events=events,
            metadata=report,
        )

    def _read_telemetry(self, path: Path, nrows: int | None = None) -> pd.DataFrame:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            return frame.head(nrows) if nrows is not None else frame
        return pd.read_csv(path, nrows=nrows)

    def _channel_names(self, frame: pd.DataFrame) -> tuple[str, ...]:
        excluded = {
            "timestamp",
            "is_anomaly",
            "label",
            "anomaly",
            "spacecraft",
            "channel_id",
            "sequence_id",
            "split",
        }
        return tuple(
            column
            for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        )

    def _load_events(
        self,
        root: Path,
        index: pd.Index,
    ) -> tuple[dict[str, Any], ...]:
        del index
        for filename in self.label_filenames:
            label_path = root / filename
            if not label_path.exists():
                continue
            labels = pd.read_csv(label_path)
            events: list[dict[str, Any]] = []
            for number, row in enumerate(labels.to_dict(orient="records"), start=1):
                start = row.get("start") or row.get("start_time")
                end = row.get("end") or row.get("end_time")
                if start is None or end is None:
                    continue
                events.append(
                    {
                        "event_id": str(row.get("event_id", f"NASA-{number:04d}")),
                        "partition": str(row.get("partition", "unknown")),
                        "anomaly_type": str(row.get("anomaly_type", "unknown")),
                        "severity": str(row.get("severity", "unknown")),
                        "start": pd.Timestamp(start).isoformat(),
                        "end": pd.Timestamp(end).isoformat(),
                    }
            )
            return tuple(events)
        return ()
