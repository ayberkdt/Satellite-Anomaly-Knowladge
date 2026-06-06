import numpy as np
import pandas as pd
import pytest

from sak.anomaly import EarlyWarningFilter, build_detected_events, ewma_smooth
from sak.evaluation import event_metrics
from sak.models.baselines import PCAAnomalyModel
from sak.preprocessing import (
    RobustTelemetryPreprocessor,
    chronological_calibration_split,
)
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_synthetic_anomalies_use_calibration_validation_and_test() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=18000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)

    assert not frames.train["is_anomaly"].any()
    assert frames.calibration["is_anomaly"].any()
    assert frames.validation["is_anomaly"].any()
    assert frames.test["is_anomaly"].any()
    assert len(dataset.events) > 30


def test_pca_scores_injected_events_above_nominal_calibration() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=18000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    preprocessor = RobustTelemetryPreprocessor(dataset.channel_names)
    train = preprocessor.fit_transform(frames.train)
    calibration = preprocessor.transform(frames.calibration)
    test = preprocessor.transform(frames.test)
    model = PCAAnomalyModel(explained_variance=0.95).fit(train)
    calibration_scores, _ = model.score(calibration)
    test_scores, _ = model.score(test)

    anomaly_mask = frames.test["is_anomaly"].to_numpy(dtype=bool)

    assert float(np.median(test_scores[anomaly_mask])) > float(
        np.median(calibration_scores)
    )


def test_preprocessor_rejects_all_missing_channels() -> None:
    frame = pd.DataFrame(
        {"battery_voltage": [np.nan, np.nan, np.nan]},
        index=pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC"),
    )

    with pytest.raises(ValueError, match="all-missing"):
        RobustTelemetryPreprocessor(("battery_voltage",)).fit(frame)


def test_event_metrics_match_a_detected_synthetic_event() -> None:
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(periods=18000, missing_fraction=0.0)
    )
    frames = chronological_calibration_split(dataset.frame)
    test_event = next(
        event
        for event in dataset.events
        if event.partition == "test" and event.event_class == "anomaly"
    )
    labels = frames.test["anomaly_event_id"].eq(test_event.event_id).to_numpy()
    scores = labels.astype(float) * 5.0
    smoothed = ewma_smooth(scores, alpha=1.0)
    alarm = EarlyWarningFilter(
        threshold=1.0,
        ewma_alpha=1.0,
        minimum_hits=2,
        lookback_steps=3,
    ).apply(scores)
    predictions = build_detected_events(
        frames.test.index,
        alarm.alarm_mask,
        smoothed,
        merge_gap_steps=2,
    )

    metrics = event_metrics(
        predictions,
        [test_event],
        tolerance=pd.Timedelta(minutes=5),
    )

    assert metrics["matched_events"] == 1
    assert metrics["recall"] == 1.0
    assert metrics["critical_region_recall"] == 1.0
