"""Canonical dataset experiment runner for real telemetry benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from sak.data import NasaSmapMslAdapter, TelemetryDataset, build_data_quality_report
from sak.experiments.artifacts import (
    create_variant_artifact_paths,
    model_variant_name,
    write_csv,
    write_json,
)
from sak.experiments.comparison import write_comparison_artifacts
from sak.experiments.real_split import RealDataSplit, split_real_dataset
from sak.experiments.runner import (
    _build_partition_windows,
    _build_threshold_evaluations,
    _calibrate_model_scores,
    _model_metadata,
    _save_model,
)
from sak.features.windowing import LabelMode
from sak.models.autoencoders import DenseAutoencoderConfig, DenseAutoencoderModel
from sak.models.baselines import PCAAnomalyModel
from sak.models.temporal import (
    TCNAutoencoderConfig,
    TCNAutoencoderModel,
    aggregate_window_errors_to_timestamps,
)
from sak.models.temporal.scoring import Aggregation
from sak.preprocessing import RobustTelemetryPreprocessor
from sak.reporting import render_real_dashboard
from sak.visualization import plot_error_heatmap, plot_score_timeline
from sak.xai import load_subsystem_mapping


SUPPORTED_REAL_MODELS = ("pca", "dense_autoencoder", "tcn_autoencoder")


def run_real_dataset_experiment(
    *,
    adapter_name: str,
    data_path: Path,
    output_dir: Path,
    config_path: Path = Path("configs/synthetic_experiment.yaml"),
    models: Sequence[str] | None = None,
    channel_id: str | None = None,
    calibration: str = "constrained_event_f1",
    score_transform: str = "none",
    max_channels: int | None = None,
    render_dashboard: bool = False,
) -> dict[str, Any]:
    """Run PCA/AE/TCN baselines on a real canonical telemetry dataset."""

    if adapter_name != "nasa_smap_msl":
        raise ValueError(f"unsupported real-data adapter: {adapter_name}")
    requested_models = tuple(models or ("pca",))
    unknown_models = set(requested_models) - set(SUPPORTED_REAL_MODELS)
    if not requested_models:
        raise ValueError("at least one model must be requested")
    if unknown_models:
        raise ValueError(f"unsupported models: {', '.join(sorted(unknown_models))}")
    if calibration not in {"quantile", "constrained_event_f1"}:
        raise ValueError("calibration must be quantile or constrained_event_f1")
    if score_transform not in {"none", "log1p", "robust_zscore"}:
        raise ValueError("score_transform must be none, log1p or robust_zscore")

    repository_dir = config_path.resolve().parent.parent
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings.setdefault("threshold_selection", {})["strategy"] = calibration
    settings.setdefault("temporal_calibration", {}).update(
        {"enabled": True, "score_transform": score_transform}
    )
    settings.setdefault(
        "real_data_split",
        {
            "strategy": "source_train_test",
            "calibration_fraction_from_train": 0.20,
            "validation_fraction_from_train": 0.10,
            "use_test_for_selection": False,
        },
    )

    adapter = NasaSmapMslAdapter()
    selected_channel_id = _select_channel(adapter, data_path, channel_id, max_channels)
    dataset = adapter.load(data_path, channel_id=selected_channel_id)
    real_split = split_real_dataset(
        dataset,
        strategy=str(settings["real_data_split"].get("strategy", "source_train_test")),
        calibration_fraction_from_train=float(
            settings["real_data_split"].get("calibration_fraction_from_train", 0.20)
        ),
        validation_fraction_from_train=float(
            settings["real_data_split"].get("validation_fraction_from_train", 0.10)
        ),
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_artifacts(
        output_dir=output_dir,
        dataset=dataset,
        real_split=real_split,
        adapter_name=adapter_name,
        data_path=data_path,
        selected_channel_id=selected_channel_id,
        requested_models=requested_models,
        calibration=calibration,
        score_transform=score_transform,
    )

    preprocessor = RobustTelemetryPreprocessor(
        channel_names=dataset.channel_names,
        max_forward_fill_steps=int(settings["preprocessing"]["max_forward_fill_steps"]),
    )
    frames = real_split.frames
    train_values = preprocessor.fit_transform(frames.train)
    calibration_values = preprocessor.transform(frames.calibration)
    validation_values = preprocessor.transform(frames.validation)
    test_values = preprocessor.transform(frames.test)
    write_json(
        output_dir / "preprocessor.json",
        {
            "channel_names": dataset.channel_names,
            "medians": preprocessor.medians_,
            "scales": preprocessor.scales_,
        },
    )

    mapping_path = Path("configs/subsystems_nasa_smap_msl.yaml")
    if not mapping_path.is_absolute():
        mapping_path = repository_dir / mapping_path
    subsystem_mapping = load_subsystem_mapping(mapping_path) if mapping_path.exists() else {}

    calibration_events = real_split.partition_events("calibration")
    validation_events = real_split.partition_events("validation")
    test_events = real_split.partition_events("test")
    model_results: dict[str, dict[str, Any]] = {}
    common_payload = {
        "train_values": train_values,
        "train_frame": frames.train,
        "calibration_values": calibration_values,
        "calibration_frame": frames.calibration,
        "calibration_events": calibration_events,
        "validation_values": validation_values,
        "validation_frame": frames.validation,
        "validation_events": validation_events,
        "test_values": test_values,
        "test_frame": frames.test,
        "test_events": test_events,
        "channel_names": dataset.channel_names,
        "subsystem_mapping": subsystem_mapping,
        "settings": settings,
        "output_dir": output_dir,
        "dataset_metadata": dataset.metadata,
        "score_transform": score_transform,
    }

    if "pca" in requested_models:
        model_results.update(
            _evaluate_real_model(
                model_name="pca",
                model=PCAAnomalyModel(
                    explained_variance=float(settings["pca"]["explained_variance"])
                ),
                **common_payload,
            )
        )
    if "dense_autoencoder" in requested_models:
        autoencoder_settings = settings["autoencoder"]
        model_results.update(
            _evaluate_real_model(
                model_name="dense_autoencoder",
                model=DenseAutoencoderModel(
                    input_dim=len(dataset.channel_names),
                    config=DenseAutoencoderConfig(
                        hidden_dim=int(autoencoder_settings["hidden_dim"]),
                        latent_dim=min(
                            int(autoencoder_settings["latent_dim"]),
                            max(1, len(dataset.channel_names)),
                        ),
                        epochs=int(autoencoder_settings["epochs"]),
                        batch_size=int(autoencoder_settings["batch_size"]),
                        learning_rate=float(autoencoder_settings["learning_rate"]),
                        weight_decay=float(autoencoder_settings["weight_decay"]),
                        patience=int(autoencoder_settings["patience"]),
                        seed=int(settings["project"]["seed"]),
                    ),
                ),
                **common_payload,
            )
        )
    if "tcn_autoencoder" in requested_models:
        model_results.update(
            _evaluate_real_temporal_model(
                **common_payload,
                seed=int(settings["project"]["seed"]),
            )
        )

    summary: dict[str, Any] = {
        "dataset": {
            "name": "nasa_smap_msl",
            "source": dataset.metadata.get("source", "nasa_smap_msl"),
            "source_layout": dataset.metadata.get("source_layout", ""),
            "channel_id": dataset.metadata.get("selected_channel_id"),
            "rows": len(dataset.frame),
            "channels": len(dataset.channel_names),
            "events": len(real_split.events),
            "train_rows": len(frames.train),
            "calibration_rows": len(frames.calibration),
            "validation_rows": len(frames.validation),
            "test_rows": len(frames.test),
            "calibration_events": len(calibration_events),
            "validation_events": len(validation_events),
            "test_events": len(test_events),
            "timestamp_synthetic": bool(dataset.metadata.get("timestamp_synthetic", False)),
            "critical_region_available": bool(
                dataset.metadata.get("critical_region_available", False)
            ),
            "test_used_for_selection": False,
        },
        **model_results,
    }
    write_comparison_artifacts(output_dir, summary)
    if render_dashboard:
        render_real_dashboard(
            comparison=summary,
            artifact_dir=output_dir,
            dashboard_path=output_dir / "dashboard.html",
            dataset_metadata=dataset.metadata,
            split_manifest=real_split.manifest,
        )
    return summary


def _select_channel(
    adapter: NasaSmapMslAdapter,
    data_path: Path,
    channel_id: str | None,
    max_channels: int | None,
) -> str | None:
    if channel_id is not None:
        return channel_id
    channels = adapter.list_channels(data_path)
    if max_channels is not None and max_channels < 1:
        raise ValueError("max_channels must be positive when provided")
    if max_channels is not None and channels:
        return channels[:max_channels][0]
    return None


def _write_dataset_artifacts(
    *,
    output_dir: Path,
    dataset: TelemetryDataset,
    real_split: RealDataSplit,
    adapter_name: str,
    data_path: Path,
    selected_channel_id: str | None,
    requested_models: Sequence[str],
    calibration: str,
    score_transform: str,
) -> None:
    partition_map = {
        "train": real_split.frames.train,
        "calibration": real_split.frames.calibration,
        "validation": real_split.frames.validation,
        "test": real_split.frames.test,
    }
    write_json(
        output_dir / "data_quality_report.json",
        build_data_quality_report(
            frame=dataset.frame,
            channel_names=dataset.channel_names,
            partitions=partition_map,
            events=real_split.events,
            channel_groups={},
            strict=True,
            dataset_metadata=dataset.metadata,
        ),
    )
    write_json(output_dir / "split_manifest.json", real_split.manifest)
    write_json(
        output_dir / "run_manifest.json",
        {
            "run_id": f"sak-v3.0-real-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sak_version": "SAK-v3.0",
            "dataset_name": "nasa_smap_msl",
            "adapter": adapter_name,
            "data_path": data_path,
            "selected_channel_id": selected_channel_id,
            "models": list(requested_models),
            "calibration": {
                "strategy": calibration,
                "selection_partition": "calibration",
                "test_partition_used_for_selection": False,
            },
            "score_transform": score_transform,
            "critical_region_policy": "proxy_unavailable_unless_source_annotations_exist",
        },
    )
    write_json(output_dir / "dataset_metadata.json", dataset.metadata)


def _evaluate_real_model(
    *,
    model_name: str,
    model: Any,
    train_values: np.ndarray,
    train_frame: pd.DataFrame,
    calibration_values: np.ndarray,
    calibration_frame: pd.DataFrame,
    calibration_events: tuple[Any, ...],
    validation_values: np.ndarray,
    validation_frame: pd.DataFrame,
    validation_events: tuple[Any, ...],
    test_values: np.ndarray,
    test_frame: pd.DataFrame,
    test_events: tuple[Any, ...],
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    settings: dict[str, Any],
    output_dir: Path,
    dataset_metadata: dict[str, Any],
    score_transform: str,
) -> dict[str, dict[str, Any]]:
    del train_frame
    model.fit(train_values)
    calibration_scores, _ = model.score(calibration_values)
    validation_scores, _ = model.score(validation_values)
    test_scores, channel_errors = model.score(test_values)
    if channel_errors is None:
        raise RuntimeError(f"{model_name} must provide channel reconstruction errors")
    (
        calibration_scores,
        validation_scores,
        test_scores,
        score_calibration,
    ) = _calibrate_model_scores(
        calibration_scores=calibration_scores,
        validation_scores=validation_scores,
        test_scores=test_scores,
        calibration_frame=calibration_frame,
        method=score_transform,
    )
    evaluations = _build_threshold_evaluations(
        calibration_scores=calibration_scores,
        calibration_frame=calibration_frame,
        calibration_events=calibration_events,
        validation_scores=validation_scores,
        validation_frame=validation_frame,
        validation_events=validation_events,
        test_scores=test_scores,
        test_frame=test_frame,
        test_events=test_events,
        settings=settings,
        score_calibration=score_calibration,
    )
    results: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        model_variant, payload = _write_real_variant(
            model_name=model_name,
            model=model,
            evaluation=evaluation,
            test_scores=test_scores,
            channel_errors=channel_errors,
            test_frame=test_frame,
            test_events=test_events,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            output_dir=output_dir,
            dataset_metadata=dataset_metadata,
        )
        results[model_variant] = payload
    return results


def _evaluate_real_temporal_model(
    *,
    train_values: np.ndarray,
    train_frame: pd.DataFrame,
    calibration_values: np.ndarray,
    calibration_frame: pd.DataFrame,
    calibration_events: tuple[Any, ...],
    validation_values: np.ndarray,
    validation_frame: pd.DataFrame,
    validation_events: tuple[Any, ...],
    test_values: np.ndarray,
    test_frame: pd.DataFrame,
    test_events: tuple[Any, ...],
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    settings: dict[str, Any],
    output_dir: Path,
    dataset_metadata: dict[str, Any],
    score_transform: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    temporal_settings = settings["temporal_autoencoder"]
    shortest_partition = min(
        len(calibration_values),
        len(validation_values),
        len(test_values),
        len(train_values),
    )
    window_size = min(int(temporal_settings["window_size"]), max(2, shortest_partition // 2))
    stride = min(int(temporal_settings["stride"]), max(1, window_size))
    model = TCNAutoencoderModel(
        config=TCNAutoencoderConfig(
            window_size=window_size,
            stride=stride,
            input_channels=len(channel_names),
            hidden_channels=int(temporal_settings["hidden_channels"]),
            latent_channels=int(temporal_settings["latent_channels"]),
            kernel_size=int(temporal_settings["kernel_size"]),
            num_layers=int(temporal_settings["num_layers"]),
            dropout=float(temporal_settings["dropout"]),
            epochs=int(temporal_settings["epochs"]),
            batch_size=int(temporal_settings["batch_size"]),
            learning_rate=float(temporal_settings["learning_rate"]),
            weight_decay=float(temporal_settings["weight_decay"]),
            patience=int(temporal_settings["patience"]),
            seed=seed,
        )
    )
    label_mode = cast(LabelMode, str(temporal_settings.get("label_mode", "any")))
    train_windows = _build_partition_windows(
        values=train_values,
        frame=train_frame,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
    )
    calibration_windows = _build_partition_windows(
        values=calibration_values,
        frame=calibration_frame,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
    )
    validation_windows = _build_partition_windows(
        values=validation_values,
        frame=validation_frame,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
    )
    test_windows = _build_partition_windows(
        values=test_values,
        frame=test_frame,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
    )
    model.fit_windows(train_windows.X_windows)
    _, calibration_window_errors = model.score_windows(calibration_windows.X_windows)
    _, validation_window_errors = model.score_windows(validation_windows.X_windows)
    _, test_window_errors = model.score_windows(test_windows.X_windows)
    aggregation = str(settings.get("temporal_calibration", {}).get("aggregation", "mean"))
    calibration_scores, _ = aggregate_window_errors_to_timestamps(
        source_indices=calibration_windows.source_indices,
        window_channel_errors=calibration_window_errors,
        n_samples=len(calibration_values),
        aggregation=cast(Aggregation, aggregation),
    )
    validation_scores, _ = aggregate_window_errors_to_timestamps(
        source_indices=validation_windows.source_indices,
        window_channel_errors=validation_window_errors,
        n_samples=len(validation_values),
        aggregation=cast(Aggregation, aggregation),
    )
    test_scores, test_channel_errors = aggregate_window_errors_to_timestamps(
        source_indices=test_windows.source_indices,
        window_channel_errors=test_window_errors,
        n_samples=len(test_values),
        aggregation=cast(Aggregation, aggregation),
    )
    (
        calibration_scores,
        validation_scores,
        test_scores,
        score_calibration,
    ) = _calibrate_model_scores(
        calibration_scores=calibration_scores,
        validation_scores=validation_scores,
        test_scores=test_scores,
        calibration_frame=calibration_frame,
        method=score_transform,
    )
    evaluations = _build_threshold_evaluations(
        calibration_scores=calibration_scores,
        calibration_frame=calibration_frame,
        calibration_events=calibration_events,
        validation_scores=validation_scores,
        validation_frame=validation_frame,
        validation_events=validation_events,
        test_scores=test_scores,
        test_frame=test_frame,
        test_events=test_events,
        settings=settings,
        score_calibration=score_calibration,
    )
    results: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        model_variant, payload = _write_real_variant(
            model_name="tcn_autoencoder",
            model=model,
            evaluation=evaluation,
            test_scores=test_scores,
            channel_errors=test_channel_errors,
            test_frame=test_frame,
            test_events=test_events,
            channel_names=channel_names,
            subsystem_mapping=subsystem_mapping,
            output_dir=output_dir,
            dataset_metadata=dataset_metadata,
        )
        results[model_variant] = payload
    return results

def _write_real_variant(
    *,
    model_name: str,
    model: Any,
    evaluation: Any,
    test_scores: np.ndarray,
    channel_errors: np.ndarray,
    test_frame: pd.DataFrame,
    test_events: tuple[Any, ...],
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    output_dir: Path,
    dataset_metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_variant = model_variant_name(model_name, evaluation.strategy)
    paths = create_variant_artifact_paths(output_dir, model_variant)
    serialized_events = _serialize_real_detected_events(
        model_variant=model_variant,
        evaluation=evaluation,
        test_frame=test_frame,
        channel_errors=channel_errors,
        channel_names=channel_names,
        subsystem_mapping=subsystem_mapping,
    )
    _write_real_score_artifacts(
        paths_root=paths.root,
        evaluation=evaluation,
        test_scores=test_scores,
        test_frame=test_frame,
    )
    calibration_metadata = _real_calibration_metadata(evaluation.calibration_metadata)
    write_json(paths.root / "events.json", serialized_events)
    write_json(paths.diagnostics / "operating_point_selection.json", calibration_metadata)
    write_json(
        paths.diagnostics / "test_partition_metrics.json",
        {
            "point_metrics": evaluation.point_result,
            "event_metrics": _mark_critical_proxy(evaluation.event_result, dataset_metadata),
        },
    )
    write_json(paths.diagnostics / "calibration_partition_metrics.json", evaluation.calibration_partition_result)
    write_json(paths.diagnostics / "validation_partition_metrics.json", evaluation.validation_partition_result)
    write_csv(paths.diagnostics / "filter_sweep.csv", evaluation.filter_sweep)
    plot_score_timeline(
        timestamps=test_frame.index,
        raw_scores=test_scores,
        smoothed_scores=evaluation.smoothed_scores,
        threshold=evaluation.thresholds,
        labels=test_frame["is_anomaly"].to_numpy(dtype=bool),
        events=evaluation.detected_events,
        output_path=paths.plots / "score_timeline.png",
        title=f"SAK real telemetry - {model_variant}",
    )
    plot_error_heatmap(
        timestamps=test_frame.index,
        channel_errors=channel_errors,
        channel_names=channel_names,
        output_path=paths.plots / "channel_error_heatmap.png",
        title=f"Real telemetry channel reconstruction errors - {model_variant}",
    )
    _save_model(model, paths)
    payload = {
        "model_name": model_name,
        "model_variant": model_variant,
        "threshold": evaluation.threshold,
        "thresholding": evaluation.threshold_metadata,
        "point_metrics": evaluation.point_result,
        "event_metrics": _mark_critical_proxy(evaluation.event_result, dataset_metadata),
        "xai_metrics": _real_xai_metrics(serialized_events, evaluation.event_result, test_events),
        "threshold_sweep": evaluation.threshold_sweep,
        "calibration": calibration_metadata,
        "calibration_partition_result": evaluation.calibration_partition_result,
        "validation_partition_result": evaluation.validation_partition_result,
        "score_semantics": {
            "alarm_score_transform": calibration_metadata.get("score_transform", "none"),
            "xai_channel_errors": "raw_reconstruction_error",
            "critical_region": (
                "source_annotation"
                if dataset_metadata.get("critical_region_available", False)
                else "proxy_unavailable"
            ),
        },
        "model": _model_metadata(model),
        "fixed_quantile_test": evaluation.fixed_quantile_test_result,
    }
    write_json(paths.root / "metrics.json", payload)
    return model_variant, payload


def _real_calibration_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result["selection_partition"] = "calibration"
    result["test_partition_used_for_selection"] = False
    if int(result.get("calibration_true_events", 0)) == 0:
        result["constraints_satisfied"] = False
        result["selection_reason"] = "no_calibration_events"
    return result


def _mark_critical_proxy(
    event_result: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    result = dict(event_result)
    available = bool(dataset_metadata.get("critical_region_available", False))
    result["critical_region_available"] = available
    result["critical_region_metric_status"] = "available" if available else "proxy"
    result["lead_time_metric_status"] = "available" if available else "proxy"
    return result


def _write_real_score_artifacts(
    *,
    paths_root: Path,
    evaluation: Any,
    test_scores: np.ndarray,
    test_frame: pd.DataFrame,
) -> None:
    event_ids = np.full(len(test_frame), "", dtype=object)
    for event in evaluation.detected_events:
        event_ids[event.start_index : event.end_index + 1] = event.event_id
    pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "raw_score": test_scores,
            "smoothed_score": evaluation.smoothed_scores,
            "threshold": evaluation.thresholds,
            "alarm": evaluation.alarm_mask,
            "is_anomaly": test_frame["is_anomaly"].to_numpy(dtype=bool),
            "operational_mode": test_frame["operational_mode"].to_numpy(),
            "source_channel_id": test_frame["source_channel_id"].to_numpy(),
        }
    ).to_csv(paths_root / "scores.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": test_frame.index,
            "alarm": evaluation.alarm_mask,
            "predicted_event_id": event_ids,
            "is_anomaly": test_frame["is_anomaly"].to_numpy(dtype=bool),
        }
    ).to_csv(paths_root / "predictions.csv", index=False)


def _serialize_real_detected_events(
    *,
    model_variant: str,
    evaluation: Any,
    test_frame: pd.DataFrame,
    channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_ids = {
        str(match["predicted_event_id"]): str(match["true_event_id"])
        for match in evaluation.event_result.get("matches", [])
    }
    for event in evaluation.detected_events:
        top_channels = _top_channel_rows(
            channel_errors[event.start_index : event.end_index + 1],
            channel_names,
            subsystem_mapping,
        )
        peak_row = test_frame.iloc[event.peak_index]
        threshold = float(evaluation.thresholds[event.peak_index])
        rows.append(
            {
                "event_id": event.event_id,
                "source_event_id": source_ids.get(event.event_id),
                "model_variant": model_variant,
                "start": event.start_time.isoformat(),
                "end": event.end_time.isoformat(),
                "peak_time": event.peak_time.isoformat(),
                "peak_score": event.peak_score,
                "threshold": threshold,
                "risk_level": _risk_level(event.peak_score, threshold),
                "context": {
                    "operational_mode": str(peak_row.get("operational_mode", "unknown")),
                    "source_channel_id": str(peak_row.get("source_channel_id", "")),
                },
                "critical_window_start": event.start_time.isoformat(),
                "critical_window_end": event.end_time.isoformat(),
                "top_channels": top_channels,
                "top_subsystems": sorted(
                    {item["subsystem"] for item in top_channels if item["subsystem"]}
                ),
            }
        )
    return rows


def _top_channel_rows(
    event_channel_errors: np.ndarray,
    channel_names: tuple[str, ...],
    subsystem_mapping: dict[str, str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if event_channel_errors.size == 0:
        return []
    mean_errors = event_channel_errors.mean(axis=0)
    total = float(mean_errors.sum())
    if total <= 0.0:
        contributions = np.zeros_like(mean_errors)
    else:
        contributions = mean_errors / total
    ranking = np.argsort(contributions)[::-1][:top_k]
    return [
        {
            "channel": channel_names[int(index)],
            "contribution": float(contributions[int(index)]),
            "subsystem": subsystem_mapping.get(channel_names[int(index)], "UNKNOWN"),
            "direction": "unknown",
        }
        for index in ranking
    ]


def _real_xai_metrics(
    serialized_events: list[dict[str, Any]],
    event_result: dict[str, Any],
    true_events: tuple[Any, ...],
) -> dict[str, float]:
    true_by_id = {str(getattr(event, "event_id", "")): event for event in true_events}
    hits_at_3: list[float] = []
    for match in event_result.get("matches", []):
        true_event = true_by_id.get(str(match["true_event_id"]))
        predicted_event = next(
            (
                event
                for event in serialized_events
                if event["event_id"] == str(match["predicted_event_id"])
            ),
            None,
        )
        if true_event is None or predicted_event is None:
            continue
        expected = set(getattr(true_event, "affected_channels", ()))
        predicted = {
            item["channel"] for item in predicted_event.get("top_channels", [])[:3]
        }
        hits_at_3.append(float(bool(expected & predicted)) if expected else 0.0)
    return {
        "channel_hit_at_1": 0.0,
        "channel_hit_at_3": float(np.mean(hits_at_3)) if hits_at_3 else 0.0,
        "subsystem_hit_at_1": 0.0,
        "subsystem_hit_at_3": 0.0,
    }


def _risk_level(score: float, threshold: float) -> str:
    ratio = score / threshold if threshold > 0.0 else float("inf")
    if ratio >= 5.0:
        return "CRITICAL"
    if ratio >= 3.0:
        return "HIGH"
    if ratio >= 1.5:
        return "MEDIUM"
    return "LOW"
