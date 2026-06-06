"""Alarm-filter grid generation and selected-row marking."""

from __future__ import annotations

from itertools import product
from typing import Any


def build_filter_sweep_grid(
    *,
    ewma_alpha: list[float],
    minimum_hits: list[int],
    lookback_steps: list[int],
    merge_gap_steps: list[int],
) -> list[dict[str, float | int]]:
    """Build deterministic valid combinations for alarm-filter evaluation."""

    rows: list[dict[str, float | int]] = []
    for alpha, hits, lookback, merge_gap in product(
        ewma_alpha,
        minimum_hits,
        lookback_steps,
        merge_gap_steps,
    ):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("ewma_alpha values must be in (0, 1]")
        if int(hits) < 1 or int(lookback) < 1 or int(merge_gap) < 0:
            raise ValueError("filter sweep integer values are invalid")
        if int(hits) > int(lookback):
            continue
        rows.append(
            {
                "ewma_alpha": float(alpha),
                "minimum_hits": int(hits),
                "lookback_steps": int(lookback),
                "merge_gap_steps": int(merge_gap),
            }
        )
    if not rows:
        raise ValueError("filter sweep produced no valid combinations")
    return rows


def mark_selected_candidate(
    rows: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mark exactly one sweep row as selected."""

    selected_index = next(
        (
            index
            for index, row in enumerate(rows)
            if all(
                row.get(key) == selected_candidate.get(key)
                for key in (
                    "quantile",
                    "ewma_alpha",
                    "minimum_hits",
                    "lookback_steps",
                    "merge_gap_steps",
                )
            )
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("selected candidate is not present in sweep rows")
    return [
        {**row, "selected": index == selected_index}
        for index, row in enumerate(rows)
    ]
