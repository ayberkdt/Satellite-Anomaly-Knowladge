import csv
from pathlib import Path

from sak.anomaly.filter_sweep import (
    build_filter_sweep_grid,
    mark_selected_candidate,
)
from sak.experiments.artifacts import write_csv


def test_filter_sweep_builds_valid_cartesian_product() -> None:
    rows = build_filter_sweep_grid(
        ewma_alpha=[0.1, 0.2],
        minimum_hits=[2, 4],
        lookback_steps=[3],
        merge_gap_steps=[10],
    )

    assert len(rows) == 2
    assert {row["minimum_hits"] for row in rows} == {2}


def test_selected_candidate_is_marked_once_and_csv_has_metrics(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "quantile": 0.99,
            "ewma_alpha": 0.1,
            "minimum_hits": 2,
            "lookback_steps": 5,
            "merge_gap_steps": 10,
            "event_precision": 1.0,
            "event_recall": 1.0,
            "event_f1": 1.0,
            "false_alarms_per_day": 0.0,
            "median_detection_delay_minutes": 2.0,
            "point_f1": 0.9,
        },
        {
            "quantile": 0.995,
            "ewma_alpha": 0.2,
            "minimum_hits": 3,
            "lookback_steps": 5,
            "merge_gap_steps": 10,
            "event_precision": 0.8,
            "event_recall": 1.0,
            "event_f1": 0.88,
            "false_alarms_per_day": 0.5,
            "median_detection_delay_minutes": 1.0,
            "point_f1": 0.8,
        },
    ]
    marked = mark_selected_candidate(rows, rows[0])
    output_path = write_csv(tmp_path / "filter_sweep.csv", marked)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert sum(row["selected"] == "True" for row in written) == 1
    assert {
        "event_precision",
        "event_recall",
        "event_f1",
        "false_alarms_per_day",
        "median_detection_delay_minutes",
        "point_f1",
        "selected",
    } <= set(written[0])
