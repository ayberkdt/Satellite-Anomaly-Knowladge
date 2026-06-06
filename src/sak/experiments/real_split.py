"""Leakage-safe split strategies for canonical real telemetry datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from sak.data.adapters.base import TelemetryDataset
from sak.preprocessing import CalibrationFrames, chronological_calibration_split


@dataclass(frozen=True)
class RealDataSplit:
    """Real-data partitions plus partition-aligned event records."""

    frames: CalibrationFrames
    events: tuple[Any, ...]
    manifest: dict[str, Any]

    def partition_events(self, partition: str) -> tuple[Any, ...]:
        """Return anomaly events assigned to one final experiment partition."""

        return tuple(
            event
            for event in self.events
            if getattr(event, "partition", "") == partition
            and getattr(event, "event_class", "anomaly") == "anomaly"
        )


def split_real_dataset(
    dataset: TelemetryDataset,
    *,
    strategy: str = "source_train_test",
    calibration_fraction_from_train: float = 0.20,
    validation_fraction_from_train: float = 0.10,
    train_fraction: float = 0.50,
) -> RealDataSplit:
    """Split real telemetry without using source test labels for selection."""

    frame = dataset.frame.sort_index().copy()
    if strategy not in {
        "source_train_test",
        "chronological_train_calibration_validation_test",
    }:
        raise ValueError(
            "real split strategy must be source_train_test or "
            "chronological_train_calibration_validation_test"
        )

    if strategy == "source_train_test" and _has_source_train_test(frame):
        frames = _source_train_test_split(
            frame,
            calibration_fraction_from_train=calibration_fraction_from_train,
            validation_fraction_from_train=validation_fraction_from_train,
        )
        final_strategy = "source_train_test"
    else:
        frames = chronological_calibration_split(
            frame,
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction_from_train,
            validation_fraction=validation_fraction_from_train,
        )
        final_strategy = "chronological_train_calibration_validation_test"

    if frames.train["is_anomaly"].any():
        raise RuntimeError("real-data train partition contains anomaly labels")

    partition_map = {
        "train": frames.train,
        "calibration": frames.calibration,
        "validation": frames.validation,
        "test": frames.test,
    }
    events = tuple(_assign_event_partition(event, partition_map) for event in dataset.events)
    manifest = {
        "strategy": final_strategy,
        "requested_strategy": strategy,
        "calibration_fraction_from_train": calibration_fraction_from_train,
        "validation_fraction_from_train": validation_fraction_from_train,
        "use_test_for_selection": False,
        "test_used_for_selection": False,
        "partitions": {
            name: {
                "rows": len(partition),
                "start": partition.index[0],
                "end": partition.index[-1],
                "anomaly_rows": int(partition["is_anomaly"].sum()),
                "event_count": sum(
                    getattr(event, "partition", "") == name for event in events
                ),
            }
            for name, partition in partition_map.items()
        },
    }
    return RealDataSplit(frames=frames, events=events, manifest=manifest)


def _has_source_train_test(frame: pd.DataFrame) -> bool:
    if "partition" not in frame:
        return False
    values = {str(value).lower() for value in frame["partition"].dropna().unique()}
    train_values = {"source_train", "train"}
    test_values = {"source_test", "test"}
    return bool(values & train_values) and bool(values & test_values)


def _source_train_test_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction_from_train: float,
    validation_fraction_from_train: float,
) -> CalibrationFrames:
    partition = frame["partition"].astype(str).str.lower()
    source_train = frame.loc[partition.isin({"source_train", "train"})].copy()
    source_test = frame.loc[partition.isin({"source_test", "test"})].copy()
    if source_train.empty or source_test.empty:
        raise ValueError("source train/test split produced an empty partition")

    heldout_fraction = calibration_fraction_from_train + validation_fraction_from_train
    if calibration_fraction_from_train <= 0.0 or validation_fraction_from_train <= 0.0:
        raise ValueError("real-data calibration and validation fractions must be positive")
    if heldout_fraction >= 1.0:
        raise ValueError("real-data calibration+validation fractions must leave training rows")

    train_fraction = 1.0 - heldout_fraction
    train_end = int(len(source_train) * train_fraction)
    calibration_end = train_end + int(len(source_train) * calibration_fraction_from_train)
    if train_end <= 0 or calibration_end <= train_end or calibration_end >= len(source_train):
        raise ValueError("source train split produced an empty train/calibration/validation")

    return CalibrationFrames(
        train=source_train.iloc[:train_end].copy(),
        calibration=source_train.iloc[train_end:calibration_end].copy(),
        validation=source_train.iloc[calibration_end:].copy(),
        test=source_test.copy(),
    )


def _assign_event_partition(
    event: Any,
    partitions: dict[str, pd.DataFrame],
) -> Any:
    current = str(getattr(event, "partition", "unknown")).lower()
    if current in partitions:
        return _replace_partition(event, current)
    if current in {"source_test", "test"}:
        return _replace_partition(event, "test")

    start = pd.Timestamp(getattr(event, "start"))
    for name, frame in partitions.items():
        if frame.index[0] <= start <= frame.index[-1]:
            return _replace_partition(event, name)
    return _replace_partition(event, "unknown")


def _replace_partition(event: Any, partition: str) -> Any:
    try:
        return replace(event, partition=partition)
    except TypeError:
        event.partition = partition
        return event
