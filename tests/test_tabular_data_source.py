from pathlib import Path

import pandas as pd
import pytest

from sak.data.tabular import CsvParquetDataSource


def test_naive_timestamps_are_localized_then_converted_to_utc(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 03:00:00"],
            "battery_voltage": [28.0],
        }
    ).to_csv(path, index=False)

    frame = CsvParquetDataSource(path, timezone="Europe/Istanbul").load()

    assert frame.index[0] == pd.Timestamp("2026-01-01 00:00:00+00:00")


def test_duplicate_timestamps_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "battery_voltage": [28.0, 27.9],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique"):
        CsvParquetDataSource(path).load()

