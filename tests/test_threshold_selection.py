from sak.anomaly.threshold_selection import select_threshold_candidate


def _candidate(
    *,
    event_f1: float,
    recall: float = 1.0,
    false_alarms: float = 0.25,
    delay: float = 5.0,
    point_f1: float = 0.5,
    true_events: int = 2,
) -> dict:
    return {
        "event_f1": event_f1,
        "event_recall": recall,
        "false_alarms_per_day": false_alarms,
        "median_detection_delay_minutes": delay,
        "point_f1": point_f1,
        "true_events": true_events,
    }


def test_selects_best_event_f1_among_feasible_candidates() -> None:
    candidates = [_candidate(event_f1=0.8), _candidate(event_f1=0.9)]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.candidate is candidates[1]
    assert result.constraints_satisfied is True


def test_tie_breaker_prefers_lower_delay() -> None:
    candidates = [
        _candidate(event_f1=0.9, delay=8.0),
        _candidate(event_f1=0.9, delay=3.0),
    ]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.candidate is candidates[1]


def test_fallback_prefers_recall_and_marks_constraints_unsatisfied() -> None:
    candidates = [
        _candidate(event_f1=0.8, recall=0.7, false_alarms=0.1),
        _candidate(event_f1=0.7, recall=0.8, false_alarms=1.0),
    ]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.candidate is candidates[1]
    assert result.constraints_satisfied is False
    assert result.selection_reason == "constraints_not_satisfied"


def test_nominal_validation_is_reported_explicitly() -> None:
    candidates = [
        _candidate(
            event_f1=0.0,
            recall=0.0,
            false_alarms=0.0,
            true_events=0,
        )
    ]

    result = select_threshold_candidate(
        candidates,
        minimum_event_recall=0.9,
        maximum_false_alarms_per_day=0.5,
    )

    assert result.constraints_satisfied is False
    assert result.selection_reason == "no_validation_events"
