import numpy as np
import pandas as pd

from sak.anomaly.thresholds import (
    ModeAwareThresholdFilter,
    calibrate_mode_thresholds,
)


def test_mode_thresholds_are_calibrated_per_operational_mode() -> None:
    scores = np.array([1.0, 2.0, 10.0, 12.0, 3.0, 4.0])
    frame = pd.DataFrame(
        {
            "operational_mode": [
                "nominal",
                "nominal",
                "payload",
                "payload",
                "safe",
                "safe",
            ]
        }
    )

    calibration = calibrate_mode_thresholds(
        scores,
        frame,
        quantile=1.0,
        context_column="operational_mode",
    )

    assert calibration.global_threshold == 12.0
    assert calibration.mode_thresholds["nominal"] == 2.0
    assert calibration.mode_thresholds["payload"] == 12.0
    assert calibration.mode_thresholds["safe"] == 4.0


def test_unknown_mode_uses_global_threshold_fallback() -> None:
    calibration = calibrate_mode_thresholds(
        np.array([1.0, 2.0, 10.0, 12.0]),
        pd.DataFrame(
            {"operational_mode": ["nominal", "nominal", "payload", "payload"]}
        ),
        quantile=1.0,
    )

    thresholds = calibration.thresholds_for_modes(["nominal", "unknown"])

    assert thresholds.tolist() == [2.0, 12.0]


def test_mode_aware_filter_uses_timestamp_specific_thresholds() -> None:
    calibration = calibrate_mode_thresholds(
        np.array([1.0, 2.0, 10.0, 12.0]),
        pd.DataFrame(
            {"operational_mode": ["nominal", "nominal", "payload", "payload"]}
        ),
        quantile=1.0,
    )
    filter_ = ModeAwareThresholdFilter(
        calibration=calibration,
        ewma_alpha=1.0,
        minimum_hits=1,
        lookback_steps=1,
    )
    result = filter_.apply(
        np.array([2.5, 11.0, 13.0]),
        pd.DataFrame({"operational_mode": ["nominal", "payload", "unknown"]}),
    )

    assert result.thresholds.tolist() == [2.0, 12.0, 12.0]
    assert result.alarm_mask.tolist() == [True, False, True]
