from pathlib import Path

import pandas as pd
import pytest

from sak.data import AdapterDataNotFoundError, NasaSmapMslAdapter


def test_nasa_adapter_missing_path_has_clear_error(tmp_path: Path) -> None:
    adapter = NasaSmapMslAdapter()

    with pytest.raises(AdapterDataNotFoundError, match="does not exist"):
        adapter.inspect(tmp_path / "missing")


def test_nasa_adapter_inspects_normalized_telemetry(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="1min",
                tz="UTC",
            ),
            "battery_voltage": [28.0, 27.9, 27.8],
            "operational_mode": ["nominal", "nominal", "safe"],
        }
    )
    frame.to_csv(tmp_path / "telemetry.csv", index=False)
    pd.DataFrame(
        {
            "event_id": ["NASA-0001"],
            "start": ["2026-01-01T00:01:00Z"],
            "end": ["2026-01-01T00:02:00Z"],
        }
    ).to_csv(tmp_path / "labeled_anomalies.csv", index=False)

    adapter = NasaSmapMslAdapter()
    report = adapter.inspect(tmp_path)
    dataset = adapter.load(tmp_path)

    assert report["recognized_layout"] is True
    assert report["load_supported"] is True
    assert report["channel_count"] == 1
    assert report["event_count"] == 1
    assert dataset.channel_names == ("battery_voltage",)
    assert len(dataset.events) == 1


def test_nasa_adapter_raw_arrays_load_canonical_dataset(tmp_path: Path) -> None:
    import numpy as np

    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    np.save(tmp_path / "train" / "A-1.npy", np.arange(20, dtype=float).reshape(10, 2))
    np.save(tmp_path / "test" / "A-1.npy", np.arange(12, dtype=float).reshape(6, 2))
    pd.DataFrame(
        {
            "chan_id": ["A-1"],
            "spacecraft": ["SMAP"],
            "anomaly_sequences": ["[[1, 3]]"],
            "class": ["contextual"],
        }
    ).to_csv(tmp_path / "labeled_anomalies.csv", index=False)

    adapter = NasaSmapMslAdapter()
    report = adapter.inspect(tmp_path)
    dataset = adapter.load(tmp_path, channel_id="A-1")

    assert report["recognized_layout"] is True
    assert report["load_supported"] is True
    assert report["detected_layout"] == "source_train_test_arrays"
    assert dataset.channel_names == ("A-1_dim_000", "A-1_dim_001")
    assert dataset.frame["is_anomaly"].sum() == 2
    assert dataset.metadata["test_rows_source"] == 6
