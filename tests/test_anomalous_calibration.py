from sak.preprocessing import chronological_calibration_split
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_calibration_contains_distinct_partition_annotated_events() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    calibration = {
        event.event_id
        for event in dataset.events
        if event.partition == "calibration" and event.event_class == "anomaly"
    }
    test = {
        event.event_id
        for event in dataset.events
        if event.partition == "test" and event.event_class == "anomaly"
    }

    assert frames.calibration["is_anomaly"].any()
    assert calibration
    assert test
    assert calibration.isdisjoint(test)
    assert all(event.partition in {"calibration", "validation", "test"} for event in dataset.events)
