"""NASA SMAP/MSL adapter with leakage-safe canonical telemetry mapping.

The adapter supports two deliberately explicit layouts:

* normalized SAK staging folders with ``telemetry.csv`` or ``telemetry.parquet``;
* Telemanom-style folders with ``train/*.npy`` and ``test/*.npy`` arrays plus
  optional ``labeled_anomalies.csv`` interval labels.

Unsupported layouts fail loudly. The adapter never invents subsystem or critical
region annotations when the source dataset does not provide them.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sak.data.adapters.base import TelemetryDataset


class AdapterDataNotFoundError(FileNotFoundError):
    """Raised when an adapter input path does not exist."""


class UnsupportedDatasetLayoutError(NotImplementedError):
    """Raised when a dataset path exists but no supported layout is detected."""


@dataclass(frozen=True)
class DatasetInspection:
    """Human-readable and JSON-serializable adapter inspection result."""

    source: str
    path: str
    exists: bool
    supported: bool
    detected_layout: str
    channel_count: int
    has_train_data: bool
    has_test_data: bool
    has_labels: bool
    has_anomaly_intervals: bool
    warnings: list[str]
    errors: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict, including legacy inspection aliases."""

        payload = asdict(self)
        payload.update(self.details)
        payload.setdefault("adapter", self.source)
        payload.setdefault("recognized_layout", self.detected_layout != "unsupported")
        payload.setdefault("load_supported", self.supported)
        payload.setdefault("labels_available", self.has_labels)
        payload.setdefault("notes", "; ".join(self.warnings + self.errors))
        return payload

    def __getitem__(self, key: str) -> Any:
        """Preserve old dict-style access used by existing tests and scripts."""

        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Preserve the small mapping surface older callers used."""

        return self.to_dict().get(key, default)


@dataclass(frozen=True)
class RealTelemetryEvent:
    """Canonical event record for real telemetry benchmark datasets."""

    event_id: str
    source_event_id: str
    partition: str
    anomaly_type: str
    severity: str
    start: pd.Timestamp
    end: pd.Timestamp
    affected_channels: tuple[str, ...]
    expected_subsystem: str
    early_warning_region_start: pd.Timestamp
    failure_region_start: pd.Timestamp
    event_class: str
    notes: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize using the same shape as synthetic injection records."""

        return {
            "event_id": self.event_id,
            "source_event_id": self.source_event_id,
            "partition": self.partition,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "affected_channels": list(self.affected_channels),
            "expected_subsystem": self.expected_subsystem,
            "early_warning_region_start": self.early_warning_region_start.isoformat(),
            "failure_region_start": self.failure_region_start.isoformat(),
            "event_class": self.event_class,
            "notes": self.notes,
            "parameters": self.parameters,
        }


class NasaSmapMslAdapter:
    """Inspect and load NASA/JPL SMAP-MSL telemetry folders."""

    label_filenames = (
        "labeled_anomalies.csv",
        "labels.csv",
        "anomaly_intervals.csv",
    )
    telemetry_filenames = ("telemetry.parquet", "telemetry.csv")

    def validate_path(self, path: Path) -> DatasetInspection:
        """Validate a candidate dataset path and return an inspection report."""

        return self.inspect(path)

    def inspect(self, path: Path) -> DatasetInspection:
        """Inspect available files and report whether a supported layout exists."""

        root = Path(path)
        if not root.exists():
            raise AdapterDataNotFoundError(
                f"NASA SMAP/MSL dataset path does not exist: {root}"
            )
        if not root.is_dir():
            raise UnsupportedDatasetLayoutError(
                f"NASA SMAP/MSL path must be a directory: {root}"
            )

        telemetry_files = [
            str(root / name) for name in self.telemetry_filenames if (root / name).exists()
        ]
        label_files = [
            str(candidate)
            for candidate in self._candidate_label_paths(root)
            if candidate.exists()
        ]
        train_dir, test_dir = self._source_train_test_dirs(root)
        array_channels = self._array_channels(train_dir, test_dir)
        array_files = sorted(str(path) for path in root.rglob("*.npy"))
        csv_files = sorted(str(path) for path in root.rglob("*.csv"))

        warnings: list[str] = []
        errors: list[str] = []
        detected_layout = "unsupported"
        supported = False
        channel_count = 0
        has_train_data = False
        has_test_data = False
        has_anomaly_intervals = False

        if telemetry_files:
            detected_layout = "normalized_telemetry"
            supported = True
            sample = self._read_telemetry(Path(telemetry_files[0]), nrows=25)
            channel_count = len(self._channel_names(sample))
            has_train_data = "partition" in sample or "split" in sample
            has_test_data = has_train_data
            if channel_count == 0:
                errors.append("normalized telemetry file has no numeric channel columns")
                supported = False
        elif train_dir is not None and test_dir is not None and array_channels:
            detected_layout = "source_train_test_arrays"
            supported = True
            channel_count = len(array_channels)
            has_train_data = True
            has_test_data = True
        else:
            errors.append(
                "expected telemetry.csv/parquet or train/test .npy arrays; "
                "no supported NASA SMAP/MSL layout was detected"
            )

        if label_files:
            try:
                labels = pd.read_csv(label_files[0])
                has_anomaly_intervals = self._labels_have_intervals(labels)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                warnings.append(f"label file could not be inspected: {error}")

        if detected_layout == "source_train_test_arrays" and not label_files:
            warnings.append("no labeled_anomalies.csv found; all rows will be nominal")
        if detected_layout == "source_train_test_arrays" and channel_count > 1:
            warnings.append(
                "multiple source channel files found; load(channel_id=...) selects "
                "a single source channel"
            )

        return DatasetInspection(
            source="nasa_smap_msl",
            path=str(root),
            exists=True,
            supported=supported,
            detected_layout=detected_layout,
            channel_count=channel_count,
            has_train_data=has_train_data,
            has_test_data=has_test_data,
            has_labels=bool(label_files),
            has_anomaly_intervals=has_anomaly_intervals,
            warnings=warnings,
            errors=errors,
            details={
                "telemetry_files": telemetry_files,
                "label_files": label_files,
                "array_file_count": len(array_files),
                "csv_file_count": len(csv_files),
                "array_channels": array_channels,
                "event_count": self._safe_event_count(label_files),
            },
        )

    def list_channels(self, path: Path) -> list[str]:
        """Return source channel IDs or normalized telemetry channel columns."""

        inspection = self.inspect(path)
        if inspection.detected_layout == "source_train_test_arrays":
            return list(inspection.details.get("array_channels", []))
        if inspection.detected_layout == "normalized_telemetry":
            telemetry_files = inspection.details.get("telemetry_files", [])
            if telemetry_files:
                frame = self._read_telemetry(Path(telemetry_files[0]), nrows=25)
                return list(self._channel_names(frame))
        return []

    def load(self, path: Path, *, channel_id: str | None = None) -> TelemetryDataset:
        """Load a supported NASA SMAP/MSL layout as canonical telemetry."""

        inspection = self.inspect(path)
        if not inspection.supported:
            raise UnsupportedDatasetLayoutError(
                "Unsupported NASA SMAP/MSL dataset layout at "
                f"{path}: {'; '.join(inspection.errors) or inspection.detected_layout}"
            )
        if inspection.detected_layout == "normalized_telemetry":
            return self._load_normalized(Path(path), inspection, channel_id=channel_id)
        if inspection.detected_layout == "source_train_test_arrays":
            return self._load_source_arrays(Path(path), inspection, channel_id=channel_id)
        raise UnsupportedDatasetLayoutError(
            f"Unsupported NASA SMAP/MSL dataset layout: {inspection.detected_layout}"
        )

    def _candidate_label_paths(self, root: Path) -> tuple[Path, ...]:
        return tuple(
            candidate
            for filename in self.label_filenames
            for candidate in (root / filename, root / "data" / filename)
        )

    def _source_train_test_dirs(self, root: Path) -> tuple[Path | None, Path | None]:
        candidates = (
            (root / "train", root / "test"),
            (root / "data" / "train", root / "data" / "test"),
        )
        for train_dir, test_dir in candidates:
            if train_dir.is_dir() and test_dir.is_dir():
                return train_dir, test_dir
        return None, None

    def _array_channels(self, train_dir: Path | None, test_dir: Path | None) -> list[str]:
        if train_dir is None or test_dir is None:
            return []
        train_channels = {path.stem for path in train_dir.glob("*.npy")}
        test_channels = {path.stem for path in test_dir.glob("*.npy")}
        return sorted(train_channels & test_channels)

    def _safe_event_count(self, label_files: list[str]) -> int | None:
        if not label_files:
            return None
        try:
            labels = pd.read_csv(label_files[0])
        except (OSError, ValueError, pd.errors.ParserError):
            return None
        if "anomaly_sequences" not in labels:
            return int(len(labels))
        count = 0
        for raw in labels["anomaly_sequences"].fillna("[]"):
            count += len(self._parse_sequence_list(raw))
        return count

    def _labels_have_intervals(self, labels: pd.DataFrame) -> bool:
        columns = {column.lower() for column in labels.columns}
        return (
            {"start", "end"}.issubset(columns)
            or {"start_time", "end_time"}.issubset(columns)
            or "anomaly_sequences" in columns
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
            "anomaly_event_id",
            "anomaly_type",
            "label_taxonomy",
            "operational_mode",
            "source_channel_id",
            "spacecraft",
            "channel_id",
            "chan_id",
            "sequence_id",
            "split",
            "partition",
        }
        return tuple(
            column
            for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        )

    def _load_normalized(
        self,
        root: Path,
        inspection: DatasetInspection,
        *,
        channel_id: str | None,
    ) -> TelemetryDataset:
        telemetry_path = Path(inspection.details["telemetry_files"][0])
        frame = self._read_telemetry(telemetry_path)
        if channel_id is not None:
            selector = None
            for column in ("source_channel_id", "channel_id", "chan_id"):
                if column in frame:
                    selector = column
                    break
            if selector is None:
                raise UnsupportedDatasetLayoutError(
                    "channel_id was requested, but normalized telemetry has no "
                    "source_channel_id/channel_id column"
                )
            frame = frame.loc[frame[selector].astype(str) == channel_id].copy()
            if frame.empty:
                raise UnsupportedDatasetLayoutError(
                    f"channel_id {channel_id!r} is not present in normalized telemetry"
                )
        frame, timestamp_synthetic = self._ensure_timestamp_index(frame)
        channels = self._channel_names(frame)
        if not channels:
            raise UnsupportedDatasetLayoutError(
                "normalized telemetry has no numeric telemetry channels"
            )
        self._ensure_canonical_columns(frame, source_channel_id=channel_id or "all")
        events = self._load_interval_events(root, frame, channels, channel_id=channel_id)
        self._apply_events_to_frame(frame, events)
        metadata = inspection.to_dict()
        metadata.update(
            {
                "source": "nasa_smap_msl",
                "source_layout": inspection.detected_layout,
                "selected_channel_id": channel_id,
                "timestamp_synthetic": timestamp_synthetic,
                "sample_period_unknown": timestamp_synthetic,
                "critical_region_unavailable": True,
                "critical_region_available": False,
                "subsystem_mapping_available": False,
            }
        )
        return TelemetryDataset(
            frame=frame,
            channel_names=channels,
            context_columns=self._context_columns(frame),
            events=events,
            metadata=metadata,
        )

    def _load_source_arrays(
        self,
        root: Path,
        inspection: DatasetInspection,
        *,
        channel_id: str | None,
    ) -> TelemetryDataset:
        train_dir, test_dir = self._source_train_test_dirs(root)
        if train_dir is None or test_dir is None:
            raise UnsupportedDatasetLayoutError("train/test array directories are missing")
        channels = list(inspection.details.get("array_channels", []))
        if not channels:
            raise UnsupportedDatasetLayoutError("no matching train/test .npy files found")
        selected_channel = channel_id or channels[0]
        if selected_channel not in channels:
            raise UnsupportedDatasetLayoutError(
                f"channel_id {selected_channel!r} is not present; available: {channels}"
            )

        train_values = self._load_array(train_dir / f"{selected_channel}.npy")
        test_values = self._load_array(test_dir / f"{selected_channel}.npy")
        feature_count = train_values.shape[1]
        if test_values.shape[1] != feature_count:
            raise UnsupportedDatasetLayoutError(
                f"train/test feature count mismatch for channel {selected_channel!r}: "
                f"{feature_count} vs {test_values.shape[1]}"
            )
        channel_names = self._array_feature_names(selected_channel, feature_count)
        values = np.vstack([train_values, test_values])
        frame = pd.DataFrame(values, columns=channel_names)
        frame["partition"] = ["source_train"] * len(train_values) + ["source_test"] * len(
            test_values
        )
        frame["source_channel_id"] = selected_channel
        frame.index = pd.date_range(
            "1970-01-01T00:00:00Z",
            periods=len(frame),
            freq="1min",
        )
        frame.index.name = "timestamp"
        self._ensure_canonical_columns(frame, source_channel_id=selected_channel)
        events = self._load_nasa_sequence_events(
            root=root,
            selected_channel=selected_channel,
            channel_names=channel_names,
            train_rows=len(train_values),
            frame_index=frame.index,
        )
        self._apply_events_to_frame(frame, events)
        metadata = inspection.to_dict()
        metadata.update(
            {
                "source": "nasa_smap_msl",
                "source_layout": inspection.detected_layout,
                "selected_channel_id": selected_channel,
                "timestamp_synthetic": True,
                "sample_period_unknown": True,
                "critical_region_unavailable": True,
                "critical_region_available": False,
                "subsystem_mapping_available": False,
                "train_rows_source": len(train_values),
                "test_rows_source": len(test_values),
                "loaded_first_channel_by_default": channel_id is None and len(channels) > 1,
            }
        )
        return TelemetryDataset(
            frame=frame,
            channel_names=channel_names,
            context_columns=self._context_columns(frame),
            events=events,
            metadata=metadata,
        )

    def _load_array(self, path: Path) -> np.ndarray:
        try:
            values = np.load(path)
        except ValueError as error:
            raise UnsupportedDatasetLayoutError(
                f"could not read numpy array {path}: {error}"
            ) from error
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        elif matrix.ndim > 2:
            matrix = matrix.reshape(matrix.shape[0], -1)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise UnsupportedDatasetLayoutError(f"array {path} is empty or invalid")
        if not np.all(np.isfinite(matrix)):
            raise UnsupportedDatasetLayoutError(
                f"array {path} contains NaN or infinite values; stage cleaned telemetry"
            )
        return matrix

    def _array_feature_names(self, channel_id: str, feature_count: int) -> tuple[str, ...]:
        if feature_count == 1:
            return (f"{channel_id}_value",)
        return tuple(f"{channel_id}_dim_{index:03d}" for index in range(feature_count))

    def _ensure_timestamp_index(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        result = frame.copy()
        if "timestamp" in result.columns:
            parsed = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
            if parsed.notna().all():
                result = result.drop(columns=["timestamp"])
                result.index = pd.DatetimeIndex(parsed, name="timestamp")
                return result.sort_index(), False
        result.index = pd.date_range(
            "1970-01-01T00:00:00Z",
            periods=len(result),
            freq="1min",
        )
        result.index.name = "timestamp"
        return result, True

    def _ensure_canonical_columns(
        self,
        frame: pd.DataFrame,
        *,
        source_channel_id: str,
    ) -> None:
        if "is_anomaly" not in frame:
            frame["is_anomaly"] = False
        frame["is_anomaly"] = frame["is_anomaly"].fillna(False).astype(bool)
        if "anomaly_event_id" not in frame:
            frame["anomaly_event_id"] = ""
        frame["anomaly_event_id"] = frame["anomaly_event_id"].fillna("").astype(str)
        if "anomaly_type" not in frame:
            frame["anomaly_type"] = ""
        frame["anomaly_type"] = frame["anomaly_type"].fillna("").astype(str)
        if "label_taxonomy" not in frame:
            frame["label_taxonomy"] = np.where(frame["is_anomaly"], "anomaly", "nominal")
        frame["label_taxonomy"] = frame["label_taxonomy"].fillna("nominal").astype(str)
        if "operational_mode" not in frame:
            frame["operational_mode"] = "unknown"
        frame["operational_mode"] = frame["operational_mode"].fillna("unknown").astype(str)
        if "source_channel_id" not in frame:
            frame["source_channel_id"] = source_channel_id
        frame["source_channel_id"] = frame["source_channel_id"].fillna(
            source_channel_id
        ).astype(str)

    def _context_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        candidates = (
            "operational_mode",
            "source_channel_id",
            "partition",
            "spacecraft",
            "channel_id",
            "sequence_id",
        )
        return tuple(column for column in candidates if column in frame)

    def _load_interval_events(
        self,
        root: Path,
        frame: pd.DataFrame,
        channel_names: tuple[str, ...],
        *,
        channel_id: str | None,
    ) -> tuple[RealTelemetryEvent, ...]:
        for label_path in self._candidate_label_paths(root):
            if not label_path.exists():
                continue
            labels = pd.read_csv(label_path)
            if "anomaly_sequences" in labels:
                return self._load_nasa_sequence_events(
                    root=root,
                    selected_channel=channel_id or "all",
                    channel_names=channel_names,
                    train_rows=0,
                    frame_index=frame.index,
                    labels=labels,
                )
            events: list[RealTelemetryEvent] = []
            for number, row in enumerate(labels.to_dict(orient="records"), start=1):
                row_channel = row.get("source_channel_id") or row.get("channel_id") or row.get(
                    "chan_id"
                )
                if channel_id is not None and row_channel is not None:
                    if str(row_channel) != channel_id:
                        continue
                start_raw = row.get("start") or row.get("start_time")
                end_raw = row.get("end") or row.get("end_time")
                if start_raw is None or end_raw is None:
                    continue
                start, end = self._resolve_frame_interval(frame.index, start_raw, end_raw)
                partition = str(row.get("partition", row.get("split", "unknown")))
                source_event_id = str(row.get("event_id", f"{label_path.name}:{number}"))
                event_id = str(row.get("event_id", f"NASA-{number:04d}"))
                events.append(
                    self._event(
                        event_id=event_id,
                        source_event_id=source_event_id,
                        partition=partition,
                        anomaly_type=str(row.get("anomaly_type", row.get("class", "anomaly"))),
                        severity=str(row.get("severity", "unknown")),
                        start=start,
                        end=end,
                        affected_channels=channel_names,
                        notes=(
                            "Real NASA interval label; critical/failure annotations "
                            "are unavailable in this adapter mapping."
                        ),
                    )
                )
            return tuple(events)
        return ()

    def _load_nasa_sequence_events(
        self,
        *,
        root: Path,
        selected_channel: str,
        channel_names: tuple[str, ...],
        train_rows: int,
        frame_index: pd.DatetimeIndex,
        labels: pd.DataFrame | None = None,
    ) -> tuple[RealTelemetryEvent, ...]:
        if labels is None:
            label_path = next(
                (candidate for candidate in self._candidate_label_paths(root) if candidate.exists()),
                None,
            )
            if label_path is None:
                return ()
            labels = pd.read_csv(label_path)
        if "anomaly_sequences" not in labels:
            return ()
        events: list[RealTelemetryEvent] = []
        event_number = 1
        for row_number, row in enumerate(labels.to_dict(orient="records"), start=1):
            row_channel = str(
                row.get("chan_id", row.get("channel_id", row.get("source_channel_id", "")))
            )
            if row_channel and row_channel != selected_channel and selected_channel != "all":
                continue
            sequences = self._parse_sequence_list(row.get("anomaly_sequences", "[]"))
            for sequence_number, sequence in enumerate(sequences, start=1):
                if len(sequence) < 2:
                    continue
                start_index = train_rows + int(sequence[0])
                end_index = train_rows + max(int(sequence[1]) - 1, int(sequence[0]))
                if start_index >= len(frame_index):
                    continue
                end_index = min(end_index, len(frame_index) - 1)
                source_event_id = f"{row_channel or selected_channel}:{row_number}:{sequence_number}"
                events.append(
                    self._event(
                        event_id=f"NASA-{event_number:04d}",
                        source_event_id=source_event_id,
                        partition="test",
                        anomaly_type=str(row.get("class", row.get("anomaly_type", "anomaly"))),
                        severity="unknown",
                        start=frame_index[start_index],
                        end=frame_index[end_index],
                        affected_channels=channel_names,
                        notes=(
                            "NASA SMAP/MSL anomaly sequence mapped from source test "
                            "indices; critical/failure annotations are unavailable."
                        ),
                    )
                )
                event_number += 1
        return tuple(events)

    def _parse_sequence_list(self, raw: Any) -> list[list[int]]:
        if isinstance(raw, float) and np.isnan(raw):
            return []
        try:
            parsed = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        result: list[list[int]] = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append([int(item[0]), int(item[1])])
        return result

    def _resolve_frame_interval(
        self,
        index: pd.DatetimeIndex,
        start_raw: Any,
        end_raw: Any,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self._looks_integer(start_raw) and self._looks_integer(end_raw):
            start_index = max(0, int(start_raw))
            end_index = min(len(index) - 1, max(start_index, int(end_raw)))
            return index[start_index], index[end_index]
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        return start, end

    def _looks_integer(self, value: Any) -> bool:
        try:
            int(value)
        except (TypeError, ValueError):
            return False
        return str(value).strip().lstrip("-").isdigit() or isinstance(value, int)

    def _event(
        self,
        *,
        event_id: str,
        source_event_id: str,
        partition: str,
        anomaly_type: str,
        severity: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        affected_channels: tuple[str, ...],
        notes: str,
    ) -> RealTelemetryEvent:
        return RealTelemetryEvent(
            event_id=event_id,
            source_event_id=source_event_id,
            partition=partition,
            anomaly_type=anomaly_type or "anomaly",
            severity=severity or "unknown",
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            affected_channels=affected_channels,
            expected_subsystem="UNKNOWN",
            early_warning_region_start=pd.Timestamp(start),
            failure_region_start=pd.Timestamp(start),
            event_class="anomaly",
            notes=notes,
            parameters={"critical_region_unavailable": True},
        )

    def _apply_events_to_frame(
        self,
        frame: pd.DataFrame,
        events: tuple[RealTelemetryEvent, ...],
    ) -> None:
        for event in events:
            mask = (frame.index >= event.start) & (frame.index <= event.end)
            if not bool(mask.any()):
                continue
            frame.loc[mask, "is_anomaly"] = True
            frame.loc[mask, "anomaly_event_id"] = event.event_id
            frame.loc[mask, "anomaly_type"] = event.anomaly_type
            frame.loc[mask, "label_taxonomy"] = "anomaly"
