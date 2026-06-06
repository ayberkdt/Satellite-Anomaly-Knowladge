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


def test_nasa_adapter_raw_arrays_are_inspectable_but_not_loaded(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "A-1.npy").write_bytes(b"placeholder")

    adapter = NasaSmapMslAdapter()
    report = adapter.inspect(tmp_path)

    assert report["recognized_layout"] is True
    assert report["load_supported"] is False
    with pytest.raises(NotImplementedError, match="raw array loading"):
        adapter.load(tmp_path)
