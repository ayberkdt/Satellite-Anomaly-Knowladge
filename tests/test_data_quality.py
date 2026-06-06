import pytest

from sak.data import build_data_quality_report
from sak.preprocessing import chronological_calibration_split
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_data_quality_report_contains_required_leakage_fields() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    report = build_data_quality_report(
        frame=dataset.frame,
        channel_names=dataset.channel_names,
        partitions=vars(frames),
        events=dataset.events,
        channel_groups=dataset.channel_groups or {},
    )

    assert report["leakage_checks_passed"] is True
    assert report["train_has_anomaly"] is False
    assert report["calibration_has_anomaly"] is True
    assert report["validation_has_anomaly"] is True
    assert report["channel_count"] == len(dataset.channel_names)


def test_data_quality_strict_mode_rejects_train_anomaly() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    frames.train.iloc[0, frames.train.columns.get_loc("is_anomaly")] = True

    with pytest.raises(RuntimeError, match="train"):
        build_data_quality_report(
            frame=dataset.frame,
            channel_names=dataset.channel_names,
            partitions=vars(frames),
            events=dataset.events,
            channel_groups=dataset.channel_groups or {},
        )


def test_data_quality_rejects_missing_partition_keys() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    partitions = vars(frames)
    partitions.pop("calibration")

    with pytest.raises(ValueError, match="missing required keys"):
        build_data_quality_report(
            frame=dataset.frame,
            channel_names=dataset.channel_names,
            partitions=partitions,
            events=dataset.events,
            channel_groups=dataset.channel_groups or {},
        )
