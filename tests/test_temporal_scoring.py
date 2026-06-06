import numpy as np
import pytest

from sak.models.temporal import aggregate_window_errors_to_timestamps


def test_timestamp_aggregation_returns_expected_shapes() -> None:
    indices = np.array([[0, 1, 2], [1, 2, 3]])
    errors = np.ones((2, 3, 2))

    scores, channel_errors = aggregate_window_errors_to_timestamps(
        source_indices=indices,
        window_channel_errors=errors,
        n_samples=4,
    )

    assert scores.shape == (4,)
    assert channel_errors.shape == (4, 2)


def test_mean_aggregation_averages_overlapping_positions() -> None:
    indices = np.array([[0, 1], [1, 2]])
    errors = np.array(
        [
            [[1.0], [2.0]],
            [[4.0], [8.0]],
        ]
    )

    scores, channel_errors = aggregate_window_errors_to_timestamps(
        source_indices=indices,
        window_channel_errors=errors,
        n_samples=3,
        aggregation="mean",
    )

    assert channel_errors[:, 0].tolist() == [1.0, 3.0, 8.0]
    assert scores.tolist() == [1.0, 3.0, 8.0]


def test_max_aggregation_selects_largest_overlap_error() -> None:
    indices = np.array([[0, 1], [1, 2]])
    errors = np.array(
        [
            [[1.0], [2.0]],
            [[4.0], [8.0]],
        ]
    )

    _, channel_errors = aggregate_window_errors_to_timestamps(
        source_indices=indices,
        window_channel_errors=errors,
        n_samples=3,
        aggregation="max",
    )

    assert channel_errors[:, 0].tolist() == [1.0, 4.0, 8.0]


def test_uncovered_timestamps_are_zero_and_warn() -> None:
    with pytest.warns(RuntimeWarning, match="not covered"):
        scores, channel_errors = aggregate_window_errors_to_timestamps(
            source_indices=np.array([[1, 2]]),
            window_channel_errors=np.ones((1, 2, 1)),
            n_samples=4,
        )

    assert scores[[0, 3]].tolist() == [0.0, 0.0]
    assert channel_errors[[0, 3], 0].tolist() == [0.0, 0.0]


def test_empty_channel_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        aggregate_window_errors_to_timestamps(
            source_indices=np.empty((0, 2), dtype=int),
            window_channel_errors=np.empty((0, 2, 0)),
            n_samples=0,
        )


@pytest.mark.parametrize(
    ("indices", "errors", "message"),
    [
        (np.zeros((1, 2, 1)), np.zeros((1, 2, 1)), "source_indices"),
        (np.zeros((1, 2), dtype=int), np.zeros((1, 2)), "window_channel_errors"),
        (
            np.zeros((2, 2), dtype=int),
            np.zeros((1, 2, 1)),
            "must align",
        ),
    ],
)
def test_invalid_temporal_shapes_raise_value_error(
    indices: np.ndarray,
    errors: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_window_errors_to_timestamps(
            source_indices=indices,
            window_channel_errors=errors,
            n_samples=3,
        )
