"""Data quality and leakage checks for telemetry experiment partitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

REQUIRED_PARTITIONS = ("train", "calibration", "validation", "test")
REQUIRED_FRAME_COLUMNS = ("is_anomaly", "label_taxonomy", "operational_mode")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")


def _validate_partitions(partitions: Mapping[str, pd.DataFrame]) -> None:
    missing = [name for name in REQUIRED_PARTITIONS if name not in partitions]
    if missing:
        raise ValueError(f"partitions are missing required keys: {missing}")
    empty = [name for name in REQUIRED_PARTITIONS if partitions[name].empty]
    if empty:
        raise ValueError(f"partitions cannot be empty: {empty}")


def build_data_quality_report(
    *,
    frame: pd.DataFrame,
    channel_names: Sequence[str],
    partitions: Mapping[str, pd.DataFrame],
    events: Sequence[Any],
    channel_groups: Mapping[str, Sequence[str]],
    strict: bool = True,
) -> dict[str, Any]:
    """Summarize missingness, labels, event coverage and leakage checks."""

    _require_columns(frame, (*REQUIRED_FRAME_COLUMNS, *channel_names))
    _validate_partitions(partitions)
    train_has_anomaly = bool(partitions["train"]["is_anomaly"].any())
    if strict and train_has_anomaly:
        raise RuntimeError("train partition contains anomaly labels")

    event_partitions = Counter(str(event.partition) for event in events)
    event_types = Counter(str(event.anomaly_type) for event in events)
    event_severity = Counter(str(event.severity) for event in events)
    taxonomy = frame["label_taxonomy"].value_counts()
    total_rows = len(frame)
    anomaly_rows = int(
        taxonomy.get("precursor", 0)
        + taxonomy.get("anomaly", 0)
        + taxonomy.get("critical", 0)
    )
    benign_rows = int(taxonomy.get("benign_transient", 0))
    nominal_rows = total_rows - anomaly_rows - benign_rows
    partition_flags = {
        f"{name}_has_anomaly": bool(partition["is_anomaly"].any())
        for name, partition in partitions.items()
    }
    leakage_checks_passed = (
        not train_has_anomaly
        and all(partition.index.is_monotonic_increasing for partition in partitions.values())
        and all(
            partitions[left].index[-1] < partitions[right].index[0]
            for left, right in (
                ("train", "calibration"),
                ("calibration", "validation"),
                ("validation", "test"),
            )
        )
    )
    return {
        "total_rows": total_rows,
        "channel_count": len(channel_names),
        "missing_fraction_per_channel": {
            channel: float(frame[channel].isna().mean())
            for channel in channel_names
        },
        "partition_lengths": {
            name: len(partition) for name, partition in partitions.items()
        },
        "event_count_by_partition": dict(sorted(event_partitions.items())),
        "event_count_by_type": dict(sorted(event_types.items())),
        "event_count_by_severity": dict(sorted(event_severity.items())),
        "label_ratio": {
            "nominal": nominal_rows / total_rows,
            "anomaly": anomaly_rows / total_rows,
            "benign_transient": benign_rows / total_rows,
        },
        **partition_flags,
        "leakage_checks_passed": leakage_checks_passed,
        "channel_groups_present": {
            group: all(channel in frame for channel in channels)
            for group, channels in channel_groups.items()
        },
        "operational_modes_present": sorted(
            str(mode) for mode in frame["operational_mode"].unique()
        ),
    }
