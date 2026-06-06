"""Run the first SAK synthetic PCA and dense-autoencoder experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from sak.contracts import AlarmEvent
from sak.evaluation import event_metrics, point_metrics
from sak.models.autoencoders import DenseAutoencoderConfig, DenseAutoencoderModel
from sak.models.baselines import PCAAnomalyModel
from sak.preprocessing import RobustTelemetryPreprocessor, chronological_split
from sak.reporting import render_early_warning_report, render_synthetic_dashboard
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry
from sak.visualization import plot_error_heatmap, plot_score_timeline
from sak.xai import build_reconstruction_explanation, load_subsystem_mapping


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _risk_level(score: float, threshold: float) -> str:
    ratio = score / threshold if threshold > 0.0 else float("inf")
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
    intersection = max(0.0, (intersection_end - intersection_start).total_seconds() + 60.0)
    union_start = min(explanation_start, true_start)
    union_end = max(explanation_end, true_end)
    union = max(60.0, (union_end - union_start).total_seconds() + 60.0)
    return intersection / union


def _build_warning_evaluation(
    *,
    test_frame: pd.DataFrame,
    true_events: tuple[Any, ...],
    warning_result: Any,
    merge_gap_steps: int,
    observation_duration: pd.Timedelta,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    detected_events = build_detected_events(
        timestamps=test_frame.index,
        alarm_mask=warning_result.alarm_mask,
        scores=warning_result.smoothed_scores,
        merge_gap_steps=merge_gap_steps,
    )
    point_result = point_metrics(
        test_frame["is_anomaly"].to_numpy(dtype=bool),
        warning_result.alarm_mask,
    )
    event_result = event_metrics(
        predicted_events=detected_events,
        true_events=true_events,
        tolerance=pd.Timedelta(minutes=5),
        observation_duration=observation_duration,
    )
    return detected_events, point_result, event_result


def _xai_metrics_for_events(
    *,
    model_name: str,
    detected_events: tuple[Any, ...],
    event_result: dict[str, Any],
    true_events: tuple[Any, ...],
    test_frame: pd.DataFrame,
    test_values: np.ndarray,
    channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    explanations: dict[str, Any] = {}
    for detected in detected_events:
        explanations[detected.event_id] = build_reconstruction_explanation(
            event=detected,
            timestamps=test_frame.index,
            scaled_values=test_values,
            channel_errors=channel_errors,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            top_k=top_k,
        )

    true_by_id = {event.event_id: event for event in true_events}
    prediction_id_map = {
        f"{model_name.upper()}-{detected.event_id}": detected.event_id
        for detected in detected_events
    }
    channel_hits: list[float] = []
    subsystem_hits: list[float] = []
    temporal_ious: list[float] = []
    critical_window_hits: list[float] = []
    for match in event_result["matches"]:
        true_event = true_by_id[str(match["true_event_id"])]
        internal_prediction_id = prediction_id_map[
            f"{model_name.upper()}-{str(match['predicted_event_id'])}"
        ]
        explanation = explanations[internal_prediction_id]
        top_three = {item.channel for item in explanation.contributions[:3]}
        channel_hits.append(
            float(bool(top_three.intersection(set(true_event.affected_channels))))
        )
        subsystem_hits.append(
            float(true_event.expected_subsystem in explanation.possible_subsystems)
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

    return explanations, {
        "channel_hit_at_3": float(np.mean(channel_hits)) if channel_hits else 0.0,
        "subsystem_hit_at_2": float(np.mean(subsystem_hits)) if subsystem_hits else 0.0,
        "mean_critical_window_iou": float(np.mean(temporal_ious))
        if temporal_ious
        else 0.0,
        "critical_window_hit_rate": float(np.mean(critical_window_hits))
        if critical_window_hits
        else 0.0,
    }


def _evaluate_model(
    *,
    name: str,
    model: Any,
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
    generated_report_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.fit(train_values)
    validation_scores, _ = model.score(validation_values)
    test_scores, channel_errors = model.score(test_values)
    if channel_errors is None:
        raise RuntimeError(f"{name} must provide channel errors for the first SAK experiment")

    alpha = float(settings["early_warning"]["ewma_alpha"])
    validation_smoothed = ewma_smooth(validation_scores, alpha)
    threshold_quantile = float(settings["early_warning"]["threshold_quantile"])
    threshold = float(
        np.quantile(
            validation_smoothed,
            threshold_quantile,
        )
    )
    mode_context_column = str(
        settings["early_warning"].get("mode_context_column", "operational_mode")
    )
    mode_minimum_samples = int(
        settings["early_warning"].get("mode_minimum_samples", 1)
    )
    mode_calibration = calibrate_mode_thresholds(
        validation_smoothed,
        validation_frame,
        quantile=threshold_quantile,
        context_column=mode_context_column,
        minimum_samples=mode_minimum_samples,
    )
    global_warning = EarlyWarningFilter(
        threshold=threshold,
        ewma_alpha=alpha,
        minimum_hits=int(settings["early_warning"]["minimum_hits"]),
        lookback_steps=int(settings["early_warning"]["lookback_steps"]),
        cooldown_steps=0,
    ).apply(test_scores)
    mode_warning = ModeAwareThresholdFilter(
        calibration=mode_calibration,
        ewma_alpha=alpha,
        minimum_hits=int(settings["early_warning"]["minimum_hits"]),
        lookback_steps=int(settings["early_warning"]["lookback_steps"]),
        cooldown_steps=0,
    ).apply(test_scores, test_frame)

    observation_duration = test_frame.index[-1] - test_frame.index[0] + pd.Timedelta(minutes=1)
    merge_gap_steps = int(settings["early_warning"]["merge_gap_steps"])
    detected_events, point_result, event_result = _build_warning_evaluation(
        test_frame=test_frame,
        true_events=true_events,
        warning_result=global_warning,
        merge_gap_steps=merge_gap_steps,
        observation_duration=observation_duration,
    )
    mode_detected_events, mode_point_result, mode_event_result = (
        _build_warning_evaluation(
            test_frame=test_frame,
            true_events=true_events,
            warning_result=mode_warning,
            merge_gap_steps=merge_gap_steps,
            observation_duration=observation_duration,
        )
    )

    threshold_sweep: list[dict[str, Any]] = []
    mode_threshold_sweep: list[dict[str, Any]] = []
    for quantile in settings["early_warning"]["threshold_sweep_quantiles"]:
        candidate_threshold = float(np.quantile(validation_smoothed, float(quantile)))
        candidate_warning = EarlyWarningFilter(
            threshold=candidate_threshold,
            ewma_alpha=alpha,
            minimum_hits=int(settings["early_warning"]["minimum_hits"]),
            lookback_steps=int(settings["early_warning"]["lookback_steps"]),
            cooldown_steps=0,
        ).apply(test_scores)
        candidate_events = build_detected_events(
            timestamps=test_frame.index,
            alarm_mask=candidate_warning.alarm_mask,
            scores=candidate_warning.smoothed_scores,
            merge_gap_steps=int(settings["early_warning"]["merge_gap_steps"]),
        )
        candidate_metrics = event_metrics(
            predicted_events=candidate_events,
            true_events=true_events,
            tolerance=pd.Timedelta(minutes=5),
            observation_duration=observation_duration,
        )
        threshold_sweep.append(
            {
                "quantile": float(quantile),
                "threshold": candidate_threshold,
                "event_precision": candidate_metrics["precision"],
                "event_recall": candidate_metrics["recall"],
                "event_f1": candidate_metrics["f1"],
                "false_alarms_per_day": candidate_metrics["false_alarms_per_day"],
                "median_detection_delay_minutes": candidate_metrics[
                    "median_detection_delay_minutes"
                ],
            }
        )
        candidate_mode_calibration = calibrate_mode_thresholds(
            validation_smoothed,
            validation_frame,
            quantile=float(quantile),
            context_column=mode_context_column,
            minimum_samples=mode_minimum_samples,
        )
        candidate_mode_warning = ModeAwareThresholdFilter(
            calibration=candidate_mode_calibration,
            ewma_alpha=alpha,
            minimum_hits=int(settings["early_warning"]["minimum_hits"]),
            lookback_steps=int(settings["early_warning"]["lookback_steps"]),
            cooldown_steps=0,
        ).apply(test_scores, test_frame)
        candidate_mode_events = build_detected_events(
            timestamps=test_frame.index,
            alarm_mask=candidate_mode_warning.alarm_mask,
            scores=candidate_mode_warning.smoothed_scores,
            merge_gap_steps=merge_gap_steps,
        )
        candidate_mode_metrics = event_metrics(
            predicted_events=candidate_mode_events,
            true_events=true_events,
            tolerance=pd.Timedelta(minutes=5),
            observation_duration=observation_duration,
        )
        mode_threshold_sweep.append(
            {
                "quantile": float(quantile),
                "threshold": candidate_mode_calibration.global_threshold,
                "event_precision": candidate_mode_metrics["precision"],
                "event_recall": candidate_mode_metrics["recall"],
                "event_f1": candidate_mode_metrics["f1"],
                "false_alarms_per_day": candidate_mode_metrics[
                    "false_alarms_per_day"
                ],
                "median_detection_delay_minutes": candidate_mode_metrics[
                    "median_detection_delay_minutes"
                ],
                "mode_thresholds": candidate_mode_calibration.to_dict()[
                    "mode_thresholds"
                ],
            }
        )

    model_dir = output_dir / name
    report_dir = model_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_report_dir.mkdir(parents=True, exist_ok=True)

    explanations: dict[str, Any] = {}
    alarm_events: list[AlarmEvent] = []
    serialized_events: list[dict[str, Any]] = []
    for detected in detected_events:
        explanation = build_reconstruction_explanation(
            event=detected,
            timestamps=test_frame.index,
            scaled_values=test_values,
            channel_errors=channel_errors,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            top_k=int(settings["explainability"]["top_k_channels"]),
        )
        explanations[detected.event_id] = explanation
        peak_row = test_frame.iloc[detected.peak_index]
        alarm_event = AlarmEvent(
            event_id=f"{name.upper()}-{detected.event_id}",
            start_time=detected.start_time.to_pydatetime(),
            end_time=detected.end_time.to_pydatetime(),
            peak_time=detected.peak_time.to_pydatetime(),
            peak_score=detected.peak_score,
            threshold=threshold,
            risk_level=_risk_level(detected.peak_score, threshold),
            explanation=explanation,
            context={
                "operational_mode": str(peak_row["operational_mode"]),
                "eclipse": bool(peak_row["eclipse"]),
                "orbit_phase": float(peak_row["orbit_phase"]),
            },
        )
        alarm_events.append(alarm_event)
        report_text = render_early_warning_report(alarm_event)
        (report_dir / f"{alarm_event.event_id}.md").write_text(report_text, encoding="utf-8")
        serialized_events.append(
            {
                "event_id": alarm_event.event_id,
                "start": alarm_event.start_time.isoformat(),
                "end": alarm_event.end_time.isoformat(),
                "peak_time": alarm_event.peak_time.isoformat(),
                "peak_score": alarm_event.peak_score,
                "risk_level": alarm_event.risk_level,
                "context": alarm_event.context,
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
        )

    if alarm_events:
        primary_event = max(alarm_events, key=lambda event: event.peak_score)
        primary_report = render_early_warning_report(primary_event)
        (generated_report_dir / f"{name}_synthetic_early_warning.md").write_text(
            primary_report,
            encoding="utf-8",
        )

    _, xai_result = _xai_metrics_for_events(
        model_name=name,
        detected_events=detected_events,
        event_result=event_result,
        true_events=true_events,
        test_frame=test_frame,
        test_values=test_values,
        channel_errors=channel_errors,
        channel_names=channel_names,
        subsystem_mapping=subsystem_mapping,
        top_k=int(settings["explainability"]["top_k_channels"]),
    )
    _, mode_xai_result = _xai_metrics_for_events(
        model_name=f"{name}_mode_aware",
        detected_events=mode_detected_events,
        event_result=mode_event_result,
        true_events=true_events,
        test_frame=test_frame,
        test_values=test_values,
        channel_errors=channel_errors,
        channel_names=channel_names,
        subsystem_mapping=subsystem_mapping,
        top_k=int(settings["explainability"]["top_k_channels"]),
    )

    plot_score_timeline(
        timestamps=test_frame.index,
        raw_scores=test_scores,
        smoothed_scores=global_warning.smoothed_scores,
        threshold=threshold,
        labels=test_frame["is_anomaly"].to_numpy(dtype=bool),
        events=detected_events,
        output_path=model_dir / "score_timeline.png",
        title=f"SAK synthetic telemetry - {name.upper()}",
    )
    plot_error_heatmap(
        timestamps=test_frame.index,
        channel_errors=channel_errors,
        channel_names=channel_names,
        output_path=model_dir / "channel_error_heatmap.png",
        title=f"Channel reconstruction errors - {name.upper()}",
    )
    pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "raw_score": test_scores,
            "smoothed_score": global_warning.smoothed_scores,
            "global_threshold": threshold,
            "mode_aware_threshold": mode_warning.thresholds,
            "global_alarm": global_warning.alarm_mask,
            "mode_aware_alarm": mode_warning.alarm_mask,
            "is_anomaly": test_frame["is_anomaly"].to_numpy(dtype=bool),
            "operational_mode": test_frame[mode_context_column].to_numpy(),
        }
    ).to_csv(model_dir / "scores.csv", index=False)
    (model_dir / "events.json").write_text(
        json.dumps(serialized_events, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if isinstance(model, PCAAnomalyModel):
        model.save(model_dir / "model.npz")
        model_metadata = {
            "n_components": model.n_components_,
            "explained_variance_retained": float(
                np.sum(model.explained_variance_ratio_)
            ),
        }
    else:
        model.save(model_dir / "model.pt")
        model_metadata = {
            "epochs_trained": len(model.training_history_),
            "final_holdout_loss": model.training_history_[-1]["holdout_loss"],
        }

    result = {
        "threshold": threshold,
        "thresholding": {
            "strategy": "global",
            "quantile": threshold_quantile,
        },
        "point_metrics": point_result,
        "event_metrics": event_result,
        "xai_metrics": xai_result,
        "threshold_sweep": threshold_sweep,
        "model": model_metadata,
    }
    mode_result = {
        "threshold": mode_calibration.global_threshold,
        "thresholding": {
            "strategy": "mode_aware",
            **mode_calibration.to_dict(),
        },
        "point_metrics": mode_point_result,
        "event_metrics": mode_event_result,
        "xai_metrics": mode_xai_result,
        "threshold_sweep": mode_threshold_sweep,
        "model": model_metadata,
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (model_dir / "metrics_mode_aware.json").write_text(
        json.dumps(mode_result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (model_dir / "threshold_comparison.json").write_text(
        json.dumps(
            {
                "global": result,
                "mode_aware": mode_result,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return result, mode_result


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    synthetic_settings = settings["synthetic"]
    dataset = generate_synthetic_telemetry(
        SyntheticConfig(
            periods=int(synthetic_settings["periods"]),
            frequency=str(synthetic_settings["frequency"]),
            start=str(synthetic_settings["start"]),
            orbit_period_steps=int(synthetic_settings["orbit_period_steps"]),
            seed=int(settings["project"]["seed"]),
            missing_fraction=float(synthetic_settings["missing_fraction"]),
        )
    )

    data_dir = Path("data/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.frame.reset_index().to_csv(data_dir / "telemetry.csv", index=False)
    (data_dir / "injection_manifest.json").write_text(
        json.dumps([event.to_dict() for event in dataset.events], indent=2),
        encoding="utf-8",
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
    (output_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "channel_names": dataset.channel_names,
                "medians": preprocessor.medians_.tolist(),
                "scales": preprocessor.scales_.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    test_start = frames.test.index[0]
    test_end = frames.test.index[-1]
    test_events = tuple(
        event
        for event in dataset.events
        if event.start >= test_start and event.end <= test_end
    )
    subsystem_mapping = load_subsystem_mapping(
        Path(settings["explainability"]["subsystem_mapping"])
    )
    generated_report_dir = Path("reports/generated")

    pca_result, pca_mode_result = _evaluate_model(
        name="pca",
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
        generated_report_dir=generated_report_dir,
    )
    autoencoder_settings = settings["autoencoder"]
    autoencoder_result, autoencoder_mode_result = _evaluate_model(
        name="dense_autoencoder",
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
                seed=int(settings["project"]["seed"]),
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
        generated_report_dir=generated_report_dir,
    )

    summary = {
        "dataset": {
            "rows": len(dataset.frame),
            "channels": len(dataset.channel_names),
            "train_rows": len(frames.train),
            "validation_rows": len(frames.validation),
            "test_rows": len(frames.test),
            "test_events": len(test_events),
        },
        "pca_global": pca_result,
        "pca_mode_aware": pca_mode_result,
        "dense_autoencoder_global": autoencoder_result,
        "dense_autoencoder_mode_aware": autoencoder_mode_result,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    render_synthetic_dashboard(
        comparison=summary,
        artifact_dir=output_dir,
        dashboard_path=Path("dashboards/sak_synthetic_dashboard.html"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_models"),
    )
    arguments = parser.parse_args()
    summary = run(arguments.config, arguments.output)
    concise = {
        model_name: {
            "event_f1": result["event_metrics"]["f1"],
            "event_recall": result["event_metrics"]["recall"],
            "false_alarms_per_day": result["event_metrics"]["false_alarms_per_day"],
            "channel_hit_at_3": result["xai_metrics"]["channel_hit_at_3"],
        }
        for model_name, result in summary.items()
        if model_name != "dataset"
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
