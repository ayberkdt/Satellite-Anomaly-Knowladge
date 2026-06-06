import pytest

from sak.experiments.multiseed import aggregate_seed_results


def _comparison(event_f1: float) -> dict:
    return {
        "dataset": {"seed": 1},
        "pca_global": {
            "point_metrics": {
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
            },
            "event_metrics": {
                "precision": 1.0,
                "recall": event_f1,
                "f1": event_f1,
                "false_alarms_per_day": 0.5,
                "median_detection_delay_minutes": 2.0,
                "mean_early_warning_time_minutes": 1.0,
            },
            "xai_metrics": {
                "channel_hit_at_1": 0.5,
                "channel_hit_at_3": 1.0,
                "subsystem_hit_at_1": 0.5,
                "subsystem_hit_at_3": 1.0,
            },
        },
    }


def test_multiseed_aggregation_reports_mean_and_std() -> None:
    rows = aggregate_seed_results(
        [(1, _comparison(0.8)), (2, _comparison(1.0))]
    )

    assert rows[0]["model_variant"] == "pca_global"
    assert rows[0]["event_f1_mean"] == 0.9
    assert rows[0]["event_f1_std"] == pytest.approx(0.1)
    assert rows[0]["channel_hit_at_3_mean"] == 1.0
