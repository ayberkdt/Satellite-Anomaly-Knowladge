from sak.anomaly.threshold_selection import select_threshold_candidate


def test_operating_point_uses_critical_constraint_and_lead_time() -> None:
    candidates = [
        {
            "event_recall": 1.0,
            "critical_region_recall": 0.8,
            "event_f1": 0.95,
            "false_alarms_per_day": 0.2,
            "median_lead_time_to_critical_minutes": 20.0,
            "median_detection_delay_minutes": 2.0,
            "point_f1": 0.9,
            "true_events": 10,
        },
        {
            "event_recall": 0.9,
            "critical_region_recall": 0.9,
            "event_f1": 0.9,
            "false_alarms_per_day": 0.3,
            "median_lead_time_to_critical_minutes": 12.0,
            "median_detection_delay_minutes": 4.0,
            "point_f1": 0.8,
            "true_events": 10,
        },
    ]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.9,
        minimum_critical_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.candidate is candidates[1]
    assert result.constraints_satisfied is True


def test_operating_point_fallback_preserves_highest_recall() -> None:
    candidates = [
        {
            "event_recall": 0.8,
            "critical_region_recall": 0.8,
            "event_f1": 0.8,
            "false_alarms_per_day": 0.4,
            "true_events": 10,
        },
        {
            "event_recall": 0.9,
            "critical_region_recall": 0.7,
            "event_f1": 0.7,
            "false_alarms_per_day": 1.0,
            "true_events": 10,
        },
    ]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.95,
        minimum_critical_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.candidate is candidates[1]
    assert result.constraints_satisfied is False
