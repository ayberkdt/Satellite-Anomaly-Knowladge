from sak.preprocessing import chronological_calibration_split
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_scheduled_events_stay_inside_partition_and_have_metadata() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    partitions = vars(frames)

    for event in dataset.events:
        partition = partitions[event.partition]
        assert event.early_warning_region_start >= partition.index[0]
        assert event.end <= partition.index[-1]
        assert event.severity
        assert event.expected_subsystem
        assert event.affected_channels
        assert event.early_warning_region_start <= event.start
        assert event.start <= event.failure_region_start <= event.end
