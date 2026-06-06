"""Single-run orchestration for the synthetic SAK experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from sak.anomaly import (
    EarlyWarningFilter,
    ModeAwareThresholdFilter,
    build_detected_events,
    calibrate_mode_thresholds,
    ewma_smooth,
)
from sak.contracts import AlarmEvent, ExplanationResult
from sak.evaluation import event_metrics, point_metrics
from sak.experiments.artifacts import (
    VariantArtifactPaths,
    create_variant_artifact_paths,
    model_variant_name,
    write_csv,
    write_json,
)
from sak.experiments.comparison import write_comparison_artifacts
from sak.experiments.diagnostics import (
    build_event_diagnostic_rows,
    build_false_positive_rows,
)
from sak.experiments.manifest import build_run_manifest, write_run_manifest
from sak.models.autoencoders import DenseAutoencoderConfig, DenseAutoencoderModel
from sak.models.baselines import PCAAnomalyModel
from sak.preprocessing import RobustTelemetryPreprocessor, chronological_split
from sak.reporting import (
    build_early_warning_report_payload,
    render_early_warning_report_payload,
    render_synthetic_dashboard,
)
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry
from sak.visualization import plot_error_heatmap, plot_score_timeline
from sak.xai import build_reconstruction_explanation, load_subsystem_mapping


class ReconstructionModel(Protocol):
    """Model surface required by the synthetic evaluation runner."""

    def fit(self, values: np.ndarray) -> Any: ...

    def score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...

    def save(self, path: Path) -> None: ...


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Threshold-specific scores, decisions, events and metadata."""

    strategy: str
    threshold: float
    thresholds: np.ndarray
    smoothed_scores: np.ndarray
    alarm_mask: np.ndarray
    detected_events: tuple[Any, ...]
    point_result: dict[str, Any]
    event_result: dict[str, Any]
    threshold_metadata: dict[str, Any]
    threshold_sweep: list[dict[str, Any]]


def _risk_level(score: float, threshold: float) -> str:
    ratio = score / threshold if threshold > 0.0 else float("inf")
    if ratio >= 5.0:
        return "CRITICAL"
    if ratio >= 3.0:
        return "HIGH"
    if ratio >= 1.5:
        return "MEDIUM"
    return "LOW"


def _temporal_iou(
    explanation_start: pd.Timestamp,
    explanation_end: pd.Timestamp,
    true_start: pd.Timestamp,
    true_end: pd.Timestamp,
) -> float:
    intersection_start = max(explanation_start, true_start)
    intersection_end = min(explanation_end, true_end)
    intersection = max(
        0.0,
        (intersection_end - intersection_start).total_seconds() + 60.0,
    )
    union_start = min(explanation_start, true_start)
    union_end = max(explanation_end, true_end)
    union = max(60.0, (union_end - union_start).total_seconds() + 60.0)
    return intersection / union


def _evaluate_warning(
    *,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    smoothed_scores: np.ndarray,
    alarm_mask: np.ndarray,
    merge_gap_steps: int,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    detected_events = build_detected_events(
        timestamps=test_frame.index,
        alarm_mask=alarm_mask,
        scores=smoothed_scores,
        merge_gap_steps=merge_gap_steps,
    )
    point_result = point_metrics(
        test_frame["is_anomaly"].to_numpy(dtype=bool),
        alarm_mask,
    )
    observation_duration = (
        test_frame.index[-1] - test_frame.index[0] + pd.Timedelta(minutes=1)
    )
    event_result = event_metrics(
        predicted_events=detected_events,
        true_events=true_events,
        tolerance=pd.Timedelta(minutes=5),
        observation_duration=observation_duration,
    )
    delays = [
        float(match["detection_delay_minutes"])
        for match in event_result.get("matches", [])
    ]
    event_result["mean_early_warning_time_minutes"] = (
        float(np.mean([max(-delay, 0.0) for delay in delays])) if delays else None
    )
    return detected_events, point_result, event_result


def _threshold_sweep(
    *,
    strategy: str,
    validation_smoothed: np.ndarray,
    validation_frame: pd.DataFrame,
    test_scores: np.ndarray,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    early_warning = settings["early_warning"]
    alpha = float(early_warning["ewma_alpha"])
    minimum_hits = int(early_warning["minimum_hits"])
    lookback_steps = int(early_warning["lookback_steps"])
    merge_gap_steps = int(early_warning["merge_gap_steps"])
    context_column = str(
        early_warning.get("mode_context_column", "operational_mode")
    )
    minimum_samples = int(early_warning.get("mode_minimum_samples", 1))
    rows: list[dict[str, Any]] = []
    for raw_quantile in early_warning["threshold_sweep_quantiles"]:
        quantile = float(raw_quantile)
        if strategy == "global":
            threshold = float(np.quantile(validation_smoothed, quantile))
            warning = EarlyWarningFilter(
                threshold=threshold,
                ewma_alpha=alpha,
                minimum_hits=minimum_hits,
                lookback_steps=lookback_steps,
            ).apply(test_scores)
            metadata: dict[str, Any] = {}
        else:
            calibration = calibrate_mode_thresholds(
                validation_smoothed,
                validation_frame,
                quantile=quantile,
                context_column=context_column,
                minimum_samples=minimum_samples,
            )
            warning = ModeAwareThresholdFilter(
                calibration=calibration,
                ewma_alpha=alpha,
                minimum_hits=minimum_hits,
                lookback_steps=lookback_steps,
            ).apply(test_scores, test_frame)
            threshold = calibration.global_threshold
            metadata = {"mode_thresholds": calibration.to_dict()["mode_thresholds"]}
        _, _, event_result = _evaluate_warning(
            test_frame=test_frame,
            true_events=true_events,
            smoothed_scores=warning.smoothed_scores,
            alarm_mask=warning.alarm_mask,
            merge_gap_steps=merge_gap_steps,
        )
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "event_precision": event_result["precision"],
                "event_recall": event_result["recall"],
                "event_f1": event_result["f1"],
                "false_alarms_per_day": event_result["false_alarms_per_day"],
                "median_detection_delay_minutes": event_result[
                    "median_detection_delay_minutes"
                ],
                **metadata,
            }
        )
    return rows


def _build_threshold_evaluations(
    *,
    validation_scores: np.ndarray,
    validation_frame: pd.DataFrame,
    test_scores: np.ndarray,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    settings: dict[str, Any],
) -> tuple[ThresholdEvaluation, ThresholdEvaluation]:
    early_warning = settings["early_warning"]
    alpha = float(early_warning["ewma_alpha"])
    quantile = float(early_warning["threshold_quantile"])
    minimum_hits = int(early_warning["minimum_hits"])
    lookback_steps = int(early_warning["lookback_steps"])
    merge_gap_steps = int(early_warning["merge_gap_steps"])
    validation_smoothed = ewma_smooth(validation_scores, alpha)

    global_threshold = float(np.quantile(validation_smoothed, quantile))
    global_warning = EarlyWarningFilter(
        threshold=global_threshold,
        ewma_alpha=alpha,
        minimum_hits=minimum_hits,
        lookback_steps=lookback_steps,
    ).apply(test_scores)
    global_events, global_point, global_event = _evaluate_warning(
        test_frame=test_frame,
        true_events=true_events,
        smoothed_scores=global_warning.smoothed_scores,
        alarm_mask=global_warning.alarm_mask,
        merge_gap_steps=merge_gap_steps,
    )
    global_evaluation = ThresholdEvaluation(
        strategy="global",
        threshold=global_threshold,
        thresholds=np.full(len(test_scores), global_threshold, dtype=float),
        smoothed_scores=global_warning.smoothed_scores,
        alarm_mask=global_warning.alarm_mask,
        detected_events=global_events,
        point_result=global_point,
        event_result=global_event,
        threshold_metadata={"strategy": "global", "quantile": quantile},
        threshold_sweep=_threshold_sweep(
            strategy="global",
            validation_smoothed=validation_smoothed,
            validation_frame=validation_frame,
            test_scores=test_scores,
            test_frame=test_frame,
            true_events=true_events,
            settings=settings,
        ),
    )

    context_column = str(
        early_warning.get("mode_context_column", "operational_mode")
    )
    minimum_samples = int(early_warning.get("mode_minimum_samples", 1))
    calibration = calibrate_mode_thresholds(
        validation_smoothed,
        validation_frame,
        quantile=quantile,
        context_column=context_column,
        minimum_samples=minimum_samples,
    )
    mode_warning = ModeAwareThresholdFilter(
        calibration=calibration,
        ewma_alpha=alpha,
        minimum_hits=minimum_hits,
        lookback_steps=lookback_steps,
    ).apply(test_scores, test_frame)
    mode_events, mode_point, mode_event = _evaluate_warning(
        test_frame=test_frame,
        true_events=true_events,
        smoothed_scores=mode_warning.smoothed_scores,
        alarm_mask=mode_warning.alarm_mask,
        merge_gap_steps=merge_gap_steps,
    )
    mode_evaluation = ThresholdEvaluation(
        strategy="mode_aware",
        threshold=calibration.global_threshold,
        thresholds=mode_warning.thresholds,
        smoothed_scores=mode_warning.smoothed_scores,
        alarm_mask=mode_warning.alarm_mask,
        detected_events=mode_events,
        point_result=mode_point,
        event_result=mode_event,
        threshold_metadata={"strategy": "mode_aware", **calibration.to_dict()},
        threshold_sweep=_threshold_sweep(
            strategy="mode_aware",
            validation_smoothed=validation_smoothed,
            validation_frame=validation_frame,
            test_scores=test_scores,
            test_frame=test_frame,
            true_events=true_events,
            settings=settings,
        ),
    )
    return global_evaluation, mode_evaluation


def _build_explanations(
    *,
    detected_events: tuple[Any, ...],
    test_frame: pd.DataFrame,
    test_values: np.ndarray,
    channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    top_k: int,
) -> dict[str, ExplanationResult]:
    return {
        detected.event_id: build_reconstruction_explanation(
            event=detected,
            timestamps=test_frame.index,
            scaled_values=test_values,
            channel_errors=channel_errors,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            top_k=top_k,
        )
        for detected in detected_events
    }


def _xai_metrics(
    *,
    explanations: dict[str, ExplanationResult],
    event_result: dict[str, Any],
    true_events: tuple[Any, ...],
) -> dict[str, float]:
    true_by_id = {event.event_id: event for event in true_events}
    channel_hit_at_1: list[float] = []
    channel_hit_at_3: list[float] = []
    subsystem_hit_at_1: list[float] = []
    subsystem_hit_at_2: list[float] = []
    subsystem_hit_at_3: list[float] = []
    temporal_ious: list[float] = []
    critical_window_hits: list[float] = []
    for match in event_result.get("matches", []):
        true_event = true_by_id[str(match["true_event_id"])]
        explanation = explanations[str(match["predicted_event_id"])]
        ranked_channels = list(explanation.contributions)
        top_one_channels = {item.channel for item in ranked_channels[:1]}
        top_three_channels = {item.channel for item in ranked_channels[:3]}
        subsystem_scores: dict[str, float] = {}
        for item in ranked_channels:
            if item.subsystem:
                subsystem_scores[item.subsystem] = (
                    subsystem_scores.get(item.subsystem, 0.0)
                    + float(item.contribution)
                )
        ranked_subsystems = [
            subsystem
            for subsystem, _ in sorted(
                subsystem_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        expected_channels = set(true_event.affected_channels)
        channel_hit_at_1.append(float(bool(expected_channels & top_one_channels)))
        channel_hit_at_3.append(float(bool(expected_channels & top_three_channels)))
        subsystem_hit_at_1.append(
            float(true_event.expected_subsystem in ranked_subsystems[:1])
        )
        subsystem_hit_at_2.append(
            float(true_event.expected_subsystem in ranked_subsystems[:2])
        )
        subsystem_hit_at_3.append(
            float(true_event.expected_subsystem in ranked_subsystems[:3])
        )
        temporal_ious.append(
            _temporal_iou(
                pd.Timestamp(explanation.critical_start),
                pd.Timestamp(explanation.critical_end),
                true_event.start,
                true_event.end,
            )
        )
        critical_window_hits.append(
            float(
                pd.Timestamp(explanation.critical_end) >= true_event.start
                and pd.Timestamp(explanation.critical_start) <= true_event.end
            )
        )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "channel_hit_at_1": mean(channel_hit_at_1),
        "channel_hit_at_3": mean(channel_hit_at_3),
        "subsystem_hit_at_1": mean(subsystem_hit_at_1),
        "subsystem_hit_at_2": mean(subsystem_hit_at_2),
        "subsystem_hit_at_3": mean(subsystem_hit_at_3),
        "mean_critical_window_iou": mean(temporal_ious),
        "critical_window_hit_rate": mean(critical_window_hits),
    }


def _serialize_explanation(explanation: ExplanationResult) -> dict[str, Any]:
    return {
        "method": explanation.method,
        "critical_window_start": explanation.critical_start.isoformat(),
        "critical_window_end": explanation.critical_end.isoformat(),
        "confidence": explanation.confidence,
        "possible_subsystems": list(explanation.possible_subsystems),
        "notes": list(explanation.notes),
        "top_channels": [
            {
                "channel": item.channel,
                "contribution": item.contribution,
                "subsystem": item.subsystem,
                "direction": item.direction,
            }
            for item in explanation.contributions
        ],
    }


def _match_source_ids(event_result: dict[str, Any]) -> dict[str, str]:
    return {
        str(match["predicted_event_id"]): str(match["true_event_id"])
        for match in event_result.get("matches", [])
    }


def _write_reports(
    *,
    paths: VariantArtifactPaths,
    generated_report_root: Path,
    model_name: str,
    model_variant: str,
    evaluation: ThresholdEvaluation,
    test_frame: pd.DataFrame,
    explanations: dict[str, ExplanationResult],
    seed: int,
) -> list[dict[str, Any]]:
    generated_variant_dir = generated_report_root / model_variant
    generated_variant_dir.mkdir(parents=True, exist_ok=True)
    source_ids = _match_source_ids(evaluation.event_result)
    serialized_events: list[dict[str, Any]] = []
    for detected in evaluation.detected_events:
        explanation = explanations[detected.event_id]
        peak_row = test_frame.iloc[detected.peak_index]
        event_threshold = float(evaluation.thresholds[detected.peak_index])
        alarm_event = AlarmEvent(
            event_id=detected.event_id,
            start_time=detected.start_time.to_pydatetime(),
            end_time=detected.end_time.to_pydatetime(),
            peak_time=detected.peak_time.to_pydatetime(),
            peak_score=detected.peak_score,
            threshold=event_threshold,
            risk_level=_risk_level(detected.peak_score, event_threshold),
            explanation=explanation,
            context={
                "operational_mode": str(peak_row["operational_mode"]),
                "eclipse": bool(peak_row["eclipse"]),
                "orbit_phase": float(peak_row["orbit_phase"]),
            },
        )
        report_payload = build_early_warning_report_payload(
            alarm_event,
            model_name=model_name,
            model_variant=model_variant,
            threshold_strategy=evaluation.strategy,
            source_event_id=source_ids.get(detected.event_id),
            metadata={"dataset": "synthetic", "seed": seed, "version": "SAK-v2.1"},
        )
        markdown = render_early_warning_report_payload(report_payload)
        for report_dir in (paths.reports, generated_variant_dir):
            (report_dir / f"{detected.event_id}.md").write_text(
                markdown,
                encoding="utf-8",
            )
            write_json(report_dir / f"{detected.event_id}.json", report_payload)
        serialized_events.append(
            {
                "event_id": detected.event_id,
                "model_variant": model_variant,
                "start": detected.start_time.isoformat(),
                "end": detected.end_time.isoformat(),
                "peak_time": detected.peak_time.isoformat(),
                "peak_score": detected.peak_score,
                "threshold": event_threshold,
                "risk_level": alarm_event.risk_level,
                "context": alarm_event.context,
                "top_channels": report_payload["top_channels"],
                "top_subsystems": report_payload["possible_subsystems"],
            }
        )
    return serialized_events


def _write_score_artifacts(
    *,
    paths: VariantArtifactPaths,
    evaluation: ThresholdEvaluation,
    test_scores: np.ndarray,
    test_frame: pd.DataFrame,
) -> None:
    event_ids = np.full(len(test_frame), "", dtype=object)
    for event in evaluation.detected_events:
        event_ids[event.start_index : event.end_index + 1] = event.event_id
    scores = pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "raw_score": test_scores,
            "smoothed_score": evaluation.smoothed_scores,
            "threshold": evaluation.thresholds,
            "alarm": evaluation.alarm_mask,
            "is_anomaly": test_frame["is_anomaly"].to_numpy(dtype=bool),
            "operational_mode": test_frame["operational_mode"].to_numpy(),
        }
    )
    predictions = pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "alarm": evaluation.alarm_mask,
            "predicted_event_id": event_ids,
            "is_anomaly": test_frame["is_anomaly"].to_numpy(dtype=bool),
        }
    )
    scores.to_csv(paths.root / "scores.csv", index=False)
    predictions.to_csv(paths.root / "predictions.csv", index=False)


def _model_metadata(model: ReconstructionModel) -> dict[str, Any]:
    if isinstance(model, PCAAnomalyModel):
        if model.explained_variance_ratio_ is None:
            raise RuntimeError("PCA metadata requested before fitting")
        return {
            "n_components": model.n_components_,
            "explained_variance_retained": float(
                np.sum(model.explained_variance_ratio_)
            ),
        }
    if isinstance(model, DenseAutoencoderModel):
        if not model.training_history_:
            raise RuntimeError("autoencoder metadata requested before fitting")
        return {
            "epochs_trained": len(model.training_history_),
            "final_holdout_loss": model.training_history_[-1]["holdout_loss"],
        }
    return {}


def _save_model(model: ReconstructionModel, paths: VariantArtifactPaths) -> None:
    suffix = "npz" if isinstance(model, PCAAnomalyModel) else "pt"
    model.save(paths.root / f"model.{suffix}")


def _write_variant(
    *,
    model_name: str,
    model: ReconstructionModel,
    evaluation: ThresholdEvaluation,
    test_scores: np.ndarray,
    channel_errors: np.ndarray,
    test_values: np.ndarray,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    settings: dict[str, Any],
    output_dir: Path,
    generated_report_root: Path,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    model_variant = model_variant_name(model_name, evaluation.strategy)
    paths = create_variant_artifact_paths(output_dir, model_variant)
    explanations = _build_explanations(
        detected_events=evaluation.detected_events,
        test_frame=test_frame,
        test_values=test_values,
        channel_errors=channel_errors,
        channel_names=channel_names,
        subsystem_mapping=subsystem_mapping,
        top_k=int(settings["explainability"]["top_k_channels"]),
    )
    xai_result = _xai_metrics(
        explanations=explanations,
        event_result=evaluation.event_result,
        true_events=true_events,
    )
    serialized_events = _write_reports(
        paths=paths,
        generated_report_root=generated_report_root,
        model_name=model_name,
        model_variant=model_variant,
        evaluation=evaluation,
        test_frame=test_frame,
        explanations=explanations,
        seed=seed,
    )
    write_json(paths.root / "events.json", serialized_events)
    write_json(
        paths.xai / "explanations.json",
        {
            event_id: _serialize_explanation(explanation)
            for event_id, explanation in explanations.items()
        },
    )
    _write_score_artifacts(
        paths=paths,
        evaluation=evaluation,
        test_scores=test_scores,
        test_frame=test_frame,
    )
    truth_payload = [event.to_dict() for event in true_events]
    diagnostic_rows = build_event_diagnostic_rows(
        model_variant=model_variant,
        truth_events=truth_payload,
        event_metrics=evaluation.event_result,
        predicted_events=serialized_events,
    )
    false_positive_rows = build_false_positive_rows(
        model_variant=model_variant,
        event_metrics=evaluation.event_result,
        predicted_events=serialized_events,
    )
    write_csv(paths.root / "event_diagnostics.csv", diagnostic_rows)
    write_csv(paths.root / "false_positive_diagnostics.csv", false_positive_rows)
    plot_score_timeline(
        timestamps=test_frame.index,
        raw_scores=test_scores,
        smoothed_scores=evaluation.smoothed_scores,
        threshold=evaluation.thresholds,
        labels=test_frame["is_anomaly"].to_numpy(dtype=bool),
        events=evaluation.detected_events,
        output_path=paths.plots / "score_timeline.png",
        title=f"SAK synthetic telemetry - {model_variant}",
    )
    plot_error_heatmap(
        timestamps=test_frame.index,
        channel_errors=channel_errors,
        channel_names=channel_names,
        output_path=paths.plots / "channel_error_heatmap.png",
        title=f"Channel reconstruction errors - {model_variant}",
    )
    _save_model(model, paths)
    result = {
        "model_name": model_name,
        "model_variant": model_variant,
        "threshold": evaluation.threshold,
        "thresholding": evaluation.threshold_metadata,
        "point_metrics": evaluation.point_result,
        "event_metrics": evaluation.event_result,
        "xai_metrics": xai_result,
        "threshold_sweep": evaluation.threshold_sweep,
        "model": _model_metadata(model),
    }
    write_json(paths.root / "metrics.json", result)
    return model_variant, result


def _evaluate_model(
    *,
    model_name: str,
    model: ReconstructionModel,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    validation_frame: pd.DataFrame,
    test_values: np.ndarray,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    settings: dict[str, Any],
    output_dir: Path,
    generated_report_root: Path,
    seed: int,
) -> dict[str, dict[str, Any]]:
    model.fit(train_values)
    validation_scores, _ = model.score(validation_values)
    test_scores, channel_errors = model.score(test_values)
    if channel_errors is None:
        raise RuntimeError(f"{model_name} must provide channel reconstruction errors")
    evaluations = _build_threshold_evaluations(
        validation_scores=validation_scores,
        validation_frame=validation_frame,
        test_scores=test_scores,
        test_frame=test_frame,
        true_events=true_events,
        settings=settings,
    )
    results: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        model_variant, result = _write_variant(
            model_name=model_name,
            model=model,
            evaluation=evaluation,
            test_scores=test_scores,
            channel_errors=channel_errors,
            test_values=test_values,
            test_frame=test_frame,
            true_events=true_events,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            settings=settings,
            output_dir=output_dir,
            generated_report_root=generated_report_root,
            seed=seed,
        )
        results[model_variant] = result
    return results


def run_synthetic_experiment(
    config_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
    models: Sequence[str] | None = None,
    generated_report_root: Path | None = None,
    data_output_dir: Path | None = None,
    dashboard_path: Path | None = None,
    render_dashboard: bool = True,
) -> dict[str, Any]:
    """Run one deterministic synthetic experiment and write SAK-v2.1 artefacts."""

    config_path = config_path.resolve()
    repository_dir = config_path.parent.parent
    output_dir = output_dir.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_seed = int(seed if seed is not None else settings["project"]["seed"])
    settings["project"]["seed"] = run_seed
    requested_models = tuple(models or ("pca", "dense_autoencoder"))
    supported_models = {"pca", "dense_autoencoder"}
    unknown_models = set(requested_models) - supported_models
    if not requested_models:
        raise ValueError("at least one model must be requested")
    if unknown_models:
        raise ValueError(
            f"unsupported models: {', '.join(sorted(unknown_models))}"
        )
    generated_report_root = (
        generated_report_root.resolve()
        if generated_report_root is not None
        else repository_dir / "reports" / "generated"
    )
    data_output_dir = (
        data_output_dir.resolve()
        if data_output_dir is not None
        else repository_dir / "data" / "synthetic"
    )
    dashboard_path = (
        dashboard_path.resolve()
        if dashboard_path is not None
        else repository_dir / "dashboards" / "sak_synthetic_dashboard.html"
    )

    synthetic_settings = settings["synthetic"]
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(
            periods=int(synthetic_settings["periods"]),
            frequency=str(synthetic_settings["frequency"]),
            start=str(synthetic_settings["start"]),
            orbit_period_steps=int(synthetic_settings["orbit_period_steps"]),
            seed=run_seed,
            missing_fraction=float(synthetic_settings["missing_fraction"]),
        )
    )
    data_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_output_dir / "telemetry.csv"
    dataset.frame.reset_index().to_csv(dataset_path, index=False)
    injection_manifest_path = data_output_dir / "injection_manifest.json"
    write_json(
        injection_manifest_path,
        [event.to_dict() for event in dataset.events],
    )

    split_settings = settings["split"]
    frames = chronological_split(
        dataset.frame,
        train_fraction=float(split_settings["train_fraction"]),
        validation_fraction=float(split_settings["validation_fraction"]),
    )
    if frames.train["is_anomaly"].any() or frames.validation["is_anomaly"].any():
        raise RuntimeError("synthetic anomalies leaked into training or validation")
    preprocessor = RobustTelemetryPreprocessor(
        channel_names=dataset.channel_names,
        max_forward_fill_steps=int(
            settings["preprocessing"]["max_forward_fill_steps"]
        ),
    )
    train_values = preprocessor.fit_transform(frames.train)
    validation_values = preprocessor.transform(frames.validation)
    test_values = preprocessor.transform(frames.test)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "preprocessor.json",
        {
            "channel_names": dataset.channel_names,
            "medians": preprocessor.medians_,
            "scales": preprocessor.scales_,
        },
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False),
        encoding="utf-8",
    )

    test_start = frames.test.index[0]
    test_end = frames.test.index[-1]
    test_events = tuple(
        event
        for event in dataset.events
        if event.start >= test_start and event.end <= test_end
    )
    mapping_path = Path(settings["explainability"]["subsystem_mapping"])
    if not mapping_path.is_absolute():
        mapping_path = repository_dir / mapping_path
    subsystem_mapping = load_subsystem_mapping(mapping_path)

    model_results: dict[str, dict[str, Any]] = {}
    if "pca" in requested_models:
        model_results.update(
            _evaluate_model(
                model_name="pca",
                model=PCAAnomalyModel(
                    explained_variance=float(settings["pca"]["explained_variance"])
                ),
                train_values=train_values,
                validation_values=validation_values,
                validation_frame=frames.validation,
                test_values=test_values,
                test_frame=frames.test,
                true_events=test_events,
                channel_names=dataset.channel_names,
                subsystem_mapping=subsystem_mapping,
                settings=settings,
                output_dir=output_dir,
                generated_report_root=generated_report_root,
                seed=run_seed,
            ),
        )
    if "dense_autoencoder" in requested_models:
        autoencoder_settings = settings["autoencoder"]
        model_results.update(
            _evaluate_model(
                model_name="dense_autoencoder",
                model=DenseAutoencoderModel(
                    input_dim=len(dataset.channel_names),
                    config=DenseAutoencoderConfig(
                        hidden_dim=int(autoencoder_settings["hidden_dim"]),
                        latent_dim=int(autoencoder_settings["latent_dim"]),
                        epochs=int(autoencoder_settings["epochs"]),
                        batch_size=int(autoencoder_settings["batch_size"]),
                        learning_rate=float(autoencoder_settings["learning_rate"]),
                        weight_decay=float(autoencoder_settings["weight_decay"]),
                        patience=int(autoencoder_settings["patience"]),
                        seed=run_seed,
                    ),
                ),
                train_values=train_values,
                validation_values=validation_values,
                validation_frame=frames.validation,
                test_values=test_values,
                test_frame=frames.test,
                true_events=test_events,
                channel_names=dataset.channel_names,
                subsystem_mapping=subsystem_mapping,
                settings=settings,
                output_dir=output_dir,
                generated_report_root=generated_report_root,
                seed=run_seed,
            ),
        )
    summary: dict[str, Any] = {
        "dataset": {
            "name": "synthetic",
            "seed": run_seed,
            "rows": len(dataset.frame),
            "channels": len(dataset.channel_names),
            "train_rows": len(frames.train),
            "validation_rows": len(frames.validation),
            "test_rows": len(frames.test),
            "test_events": len(test_events),
        },
        **model_results,
    }
    write_comparison_artifacts(output_dir, summary)
    model_variants = [key for key in summary if key != "dataset"]
    run_manifest = build_run_manifest(
        config_path=config_path,
        dataset_path=dataset_path,
        seed=run_seed,
        train_index=frames.train.index,
        validation_index=frames.validation.index,
        test_index=frames.test.index,
        models=model_variants,
        repository_dir=repository_dir,
    )
    write_run_manifest(output_dir / "run_manifest.json", run_manifest)
    if render_dashboard:
        render_synthetic_dashboard(
            comparison=summary,
            artifact_dir=output_dir,
            dashboard_path=dashboard_path,
            manifest_path=injection_manifest_path,
        )
    return summary
