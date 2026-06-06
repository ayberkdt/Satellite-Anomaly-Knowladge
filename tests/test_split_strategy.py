import pandas as pd
import pytest

from sak.preprocessing import chronological_calibration_split, chronological_split
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_four_partition_split_has_expected_ratios_and_boundaries() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)

    assert len(frames.train) == 6000
    assert len(frames.calibration) == 2400
    assert len(frames.validation) == 1200
    assert len(frames.test) == 2400
    assert not frames.train["is_anomaly"].any()
    assert all(
        partition.index.is_monotonic_increasing
        for partition in (
            frames.train,
            frames.calibration,
            frames.validation,
            frames.test,
        )
    )
    assert frames.train.index[-1] < frames.calibration.index[0]
    assert frames.calibration.index[-1] < frames.validation.index[0]
    assert frames.validation.index[-1] < frames.test.index[0]
    assert isinstance(frames.test.index, pd.DatetimeIndex)


def test_chronological_split_rejects_unsorted_timestamps() -> None:
    frame = pd.DataFrame(
        {"is_anomaly": [False, False, False, False]},
        index=pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:02:00Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:03:00Z",
            ]
        ),
    )

    with pytest.raises(ValueError, match="monotonically"):
        chronological_split(frame, train_fraction=0.5, validation_fraction=0.25)


def test_calibration_split_rejects_empty_partition_settings() -> None:
    frame = pd.DataFrame(
        {"is_anomaly": [False, False, False]},
        index=pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC"),
    )

    with pytest.raises(ValueError, match="empty partition"):
        chronological_calibration_split(frame)
