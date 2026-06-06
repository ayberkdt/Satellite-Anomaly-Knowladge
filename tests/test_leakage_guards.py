import inspect

from sak.experiments import alarm_selection
from sak.preprocessing import chronological_calibration_split
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_train_is_nominal_and_selection_api_has_no_test_metrics() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    parameters = inspect.signature(
        alarm_selection.select_alarm_configuration
    ).parameters

    assert not frames.train["is_anomaly"].any()
    assert "calibration_scores" in parameters
    assert "calibration_events" in parameters
    assert not any("test_metric" in name for name in parameters)
