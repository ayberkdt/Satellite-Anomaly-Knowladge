import numpy as np
import pandas as pd

from sak.features.windowing import build_windows


def test_windows_have_expected_shape_for_multichannel_data() -> None:
    values = np.arange(30).reshape(10, 3)

    result = build_windows(values, window_size=4)

    assert result.X_windows.shape == (7, 4, 3)
    assert result.source_indices[0].tolist() == [0, 1, 2, 3]


def test_stride_controls_window_starts() -> None:
    result = build_windows(np.arange(10), window_size=3, stride=2)

    assert result.X_windows.shape == (4, 3, 1)
    assert result.source_indices[:, 0].tolist() == [0, 2, 4, 6]


def test_any_label_mode_marks_a_window_with_any_anomaly() -> None:
    labels = np.array([0, 0, 1, 0, 0])

    result = build_windows(np.arange(5), labels=labels, window_size=3, label_mode="any")

    assert result.window_label.tolist() == [1, 1, 1]


def test_last_label_mode_uses_last_timestamp_label() -> None:
    labels = np.array([0, 1, 0, 1, 0])

    result = build_windows(
        np.arange(5),
        labels=labels,
        window_size=3,
        label_mode="last",
    )

    assert result.window_label.tolist() == [0, 1, 0]


def test_majority_label_mode_requires_more_than_half_anomalous() -> None:
    labels = np.array([1, 1, 0, 0, 1])

    result = build_windows(
        np.arange(5),
        labels=labels,
        window_size=3,
        label_mode="majority",
    )

    assert result.window_label.tolist() == [1, 0, 0]


def test_timestamps_align_to_window_boundaries() -> None:
    timestamps = pd.date_range("2026-01-01", periods=5, freq="1min").to_numpy()

    result = build_windows(
        np.arange(5),
        timestamps=timestamps,
        window_size=3,
        stride=2,
    )

    assert result.window_start_time.tolist() == timestamps[[0, 2]].tolist()
    assert result.window_end_time.tolist() == timestamps[[2, 4]].tolist()


def test_return_index_false_returns_only_feature_array() -> None:
    windows = build_windows(
        np.arange(8),
        window_size=4,
        return_index=False,
    )

    assert isinstance(windows, np.ndarray)
    assert windows.shape == (5, 4, 1)


def test_docstring_warns_to_split_before_windowing() -> None:
    documentation = (build_windows.__doc__ or "").lower()

    assert "train" in documentation
    assert "validation" in documentation
    assert "test" in documentation
    assert "partition" in documentation
