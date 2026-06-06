import numpy as np
import pytest

from sak.anomaly.calibration import ScoreCalibrator


def test_log1p_transform_matches_numpy() -> None:
    scores = np.array([0.0, 1.0, 3.0])
    calibrator = ScoreCalibrator(method="log1p").fit(scores)

    np.testing.assert_allclose(
        calibrator.transform(scores),
        np.log1p(scores),
    )


def test_robust_zscore_uses_fitted_validation_statistics() -> None:
    validation = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    test = np.array([3.0, 5.0])
    calibrator = ScoreCalibrator(method="robust_zscore").fit(validation)

    expected = (test - np.median(validation)) / (
        np.quantile(validation, 0.75) - np.quantile(validation, 0.25)
    )
    np.testing.assert_allclose(calibrator.transform(test), expected)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        ScoreCalibrator(method="robust_zscore").transform(np.array([1.0]))


@pytest.mark.parametrize("invalid", [np.array([np.nan]), np.array([np.inf])])
def test_calibrator_rejects_non_finite_scores(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScoreCalibrator().fit(invalid)
