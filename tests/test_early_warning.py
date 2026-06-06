import numpy as np
import pytest

from sak.anomaly.early_warning import EarlyWarningFilter


def test_persistence_suppresses_single_spike() -> None:
    filter_ = EarlyWarningFilter(
        threshold=1.0,
        ewma_alpha=1.0,
        minimum_hits=2,
        lookback_steps=3,
    )

    result = filter_.apply(np.array([0.0, 2.0, 0.0, 0.0]))

    assert not result.alarm_mask.any()


def test_persistence_emits_alarm_after_required_hits() -> None:
    filter_ = EarlyWarningFilter(
        threshold=1.0,
        ewma_alpha=1.0,
        minimum_hits=2,
        lookback_steps=3,
    )

    result = filter_.apply(np.array([0.0, 2.0, 2.0, 0.0]))

    assert result.alarm_mask.tolist() == [False, False, True, False]


def test_invalid_persistence_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="lookback_steps"):
        EarlyWarningFilter(threshold=1.0, minimum_hits=4, lookback_steps=3)

