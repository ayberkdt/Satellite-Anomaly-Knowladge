"""Static HTML dashboard generation for SAK experiment artefacts."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from sak.reporting.results import (
    channel_summary_rows,
    delay_rows,
    event_diagnostic_rows,
    experiment_model_rows,
    false_positive_rows,
    model_label,
    subsystem_summary_rows,
    threshold_sweep_rows,
    write_csv,
    write_results_markdown,
)
from sak.visualization.theme import (
    LABEL_TAXONOMY_COLORS,
    MODEL_COLORS,
    RISK_COLORS,
    SUBSYSTEM_COLORS,
    subsystem_color,
    subsystem_label,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_DASHBOARD_VARIANTS = (
    "pca_global",
    "dense_autoencoder_global",
    "tcn_autoencoder_global",
)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _relative(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(target, base_file.parent)).as_posix()


def _html_table(rows: list[dict[str, str | float]], *, css_class: str = "") -> str:
    if not rows:
        return "<p>No rows.</p>"
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(_fmt(row[header]))}</td>" for header in headers
        )
        body.append(f"<tr>{cells}</tr>")
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def dashboard_variant_groups(
    comparison: dict[str, Any],
    default_variants: tuple[str, ...] = DEFAULT_DASHBOARD_VARIANTS,
) -> dict[str, list[str]]:
    """Split current operating points from advanced/legacy variants."""

    model_keys = [key for key in comparison if key != "dataset"]
    default = [key for key in default_variants if key in comparison]
    advanced = [key for key in model_keys if key not in set(default)]
    return {"default": default, "advanced": advanced}


def _metric(payload: dict[str, Any], *path: str, default: float = 0.0) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value) if value is not None else default


def _operational_score(model_key: str, payload: dict[str, Any]) -> float:
    """Rank dashboard cards only; this is not a model-selection metric."""

    event = payload.get("event_metrics", {})
    xai = payload.get("xai_metrics", {})
    false_alarms = _metric(payload, "event_metrics", "false_alarms_per_day")
    lead = _metric(payload, "event_metrics", "median_lead_time_to_critical_minutes")
    normalized_fa = max(0.0, 1.0 - min(false_alarms, 5.0) / 5.0)
    normalized_lead = max(-1.0, min(lead / 60.0, 1.0))
    return (
        0.24 * float(event.get("recall", 0.0))
        + 0.22 * float(event.get("critical_region_recall", 0.0))
        + 0.18 * float(event.get("detected_before_critical_rate", 0.0))
        + 0.14 * normalized_fa
        + 0.12 * normalized_lead
        + 0.10 * float(xai.get("channel_hit_at_3", 0.0))
        + (0.001 if model_key == "tcn_autoencoder_global" else 0.0)
    )


def _filter_rows_by_models(
    rows: list[dict[str, str | float]],
    model_labels: set[str],
) -> list[dict[str, str | float]]:
    return [row for row in rows if str(row.get("model", "")) in model_labels]


def _subsystem_badge(subsystem: object) -> str:
    label = subsystem_label(subsystem)
    color = subsystem_color(subsystem)
    return (
        f'<span class="subsystem-badge" style="background:{color}">'
        f"{html.escape(label)}</span>"
    )


def _risk_badge(risk: object) -> str:
    label = str(risk or "LOW").upper()
    color = RISK_COLORS.get(label, RISK_COLORS["LOW"])
    return f'<span class="risk-badge" style="background:{color}">{html.escape(label)}</span>'


def _subsystem_legend() -> str:
    items = [
        f"<li>{_subsystem_badge(subsystem)}</li>"
        for subsystem in SUBSYSTEM_COLORS
    ]
    return f'<ul class="legend">{"".join(items)}</ul>'


def _warning_panel(warnings: list[str]) -> str:
    if not warnings:
        return "<!-- no dashboard warnings -->"
    items = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    return f'<div class="panel warning"><h2>Data Contract Warnings</h2><ul>{items}</ul></div>'


def _metric_card(title: str, value: str, note: str = "") -> str:
    return (
        f'<div class="card"><small>{html.escape(title)}</small>'
        f"<strong>{html.escape(value)}</strong>"
        f"<em>{html.escape(note)}</em></div>"
    )


def _plot_model_comparison(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_bytes(b"")
        return
    labels = [str(row["model"]) for row in rows]
    metrics = [
        ("event_recall", "Event Recall"),
        ("critical_region_recall", "Critical Recall"),
        ("detected_before_critical_rate", "Before Critical"),
        ("inverse_false_alarms", "Low False Alarms"),
        ("normalized_lead_time", "Lead Time"),
        ("channel_hit_at_3", "Channel Hit@3"),
    ]
    x = np.arange(len(labels))
    width = 0.13
    figure, axis = plt.subplots(figsize=(12, 5.4))
    colors = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#0F766E"]
    for offset, (key, label) in enumerate(metrics):
        if key == "inverse_false_alarms":
            values = [
                max(0.0, 1.0 - min(float(row["false_alarms_per_day"]), 5.0) / 5.0)
                for row in rows
            ]
        elif key == "normalized_lead_time":
            values = [
                max(0.0, min(float(row["median_lead_time_to_critical_min"]) / 60.0, 1.0))
                for row in rows
            ]
        else:
            values = [float(row[key]) for row in rows]
        axis.bar(
            x + (offset - (len(metrics) - 1) / 2.0) * width,
            values,
            width,
            label=label,
            color=colors[offset],
        )
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Score")
    axis.set_title("Model Comparison")
    axis.legend(ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_false_alarm_delay(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_bytes(b"")
        return
    labels = [str(row["model"]) for row in rows]
    false_alarms = [float(row["false_alarms_per_day"]) for row in rows]
    lead_times = [float(row["median_lead_time_to_critical_min"]) for row in rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(x, false_alarms, color="#EF4444")
    axes[0].axhline(0.75, color="#F59E0B", linestyle="--", linewidth=1.2, label="0.75/day")
    axes[0].axhline(0.50, color="#10B981", linestyle="--", linewidth=1.2, label="0.50/day")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("False alarms / day")
    axes[0].set_title("Operational Noise")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    lead_colors = ["#10B981" if value >= 0.0 else "#EF4444" for value in lead_times]
    axes[1].bar(x, lead_times, color=lead_colors)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_ylabel("Median lead time (min)")
    axes[1].set_title("Lead Time to Critical")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_threshold_sweep(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_bytes(b"")
        return
    by_model: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for model, model_rows in by_model.items():
        sorted_rows = sorted(model_rows, key=lambda row: float(row["quantile"]))
        quantiles = [float(row["quantile"]) for row in sorted_rows]
        f1_values = [float(row["event_f1"]) for row in sorted_rows]
        false_alarms = [float(row["false_alarms_per_day"]) for row in sorted_rows]
        axes[0].plot(quantiles, f1_values, marker="o", label=model)
        axes[1].plot(quantiles, false_alarms, marker="o", label=model)
    axes[0].set_title("Event F1 vs Threshold Quantile")
    axes[0].set_xlabel("Calibration quantile")
    axes[0].set_ylabel("Event F1")
    axes[0].set_ylim(0.0, 1.08)
    axes[1].set_title("False Alarms vs Threshold Quantile")
    axes[1].set_xlabel("Calibration quantile")
    axes[1].set_ylabel("False alarms / day")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_detection_delays(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted({str(row["true_event_id"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    x = np.arange(len(events))
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(12, 5.2))
    for model_index, model in enumerate(models):
        values = []
        for event in events:
            match = next(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["true_event_id"] == event
                ),
                None,
            )
            values.append(
                float(match["detection_delay_min"])
                if match is not None
                else float("nan")
            )
        axis.bar(
            x + (model_index - (len(models) - 1) / 2.0) * width,
            values,
            width,
            label=model,
        )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(events, rotation=30, ha="right")
    axis.set_ylabel("Detection delay (min)")
    axis.set_title("Matched Event Detection Delays")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_event_diagnostics(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted({str(row["true_event_id"]) for row in rows})
    labels_by_event = {
        str(row["true_event_id"]): f"{row['true_event_id']}\n{row['anomaly_type']}"
        for row in rows
    }
    models = sorted({str(row["model"]) for row in rows})
    x = np.arange(len(events))
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(13, 5.6))
    colors = {"yes": "#2f6f73", "no": "#d1495b"}

    for model_index, model in enumerate(models):
        values = []
        bar_colors = []
        for event in events:
            match = next(
                row
                for row in rows
                if row["model"] == model and row["true_event_id"] == event
            )
            values.append(float(match["detection_delay_min"]))
            bar_colors.append(colors.get(str(match["channel_hit"]), "#7a4fe0"))
        axis.bar(
            x + (model_index - (len(models) - 1) / 2.0) * width,
            values,
            width,
            label=model,
            color=bar_colors,
            alpha=0.86 if model_index == 0 else 0.58,
            edgecolor="#172033",
            linewidth=0.4,
        )

    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([labels_by_event[event] for event in events], rotation=35, ha="right")
    axis.set_ylabel("Detection delay (min)")
    axis.set_title("Event Delay by Anomaly Type (green = top-3 channel hit)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_channel_summary(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_rows = sorted(rows, key=lambda row: -float(row["top3_count"]))[:10]
    labels = [f"{row['channel']}\n{row['model']}" for row in top_rows]
    counts = [float(row["top3_count"]) for row in top_rows]
    contributions = [float(row["mean_contribution"]) for row in top_rows]
    colors = [subsystem_color(row.get("subsystem", "UNKNOWN")) for row in top_rows]
    x = np.arange(len(top_rows))
    figure, axis = plt.subplots(figsize=(13, 5.2))
    bars = axis.bar(x, counts, color=colors)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Top-3 mention count")
    axis.set_title("Most Frequent Top-3 Explanation Channels")
    axis.grid(axis="y", alpha=0.2)
    for bar, contribution in zip(bars, contributions, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{contribution:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#172033",
        )
    axis.text(
        0.01,
        0.94,
        "Labels above bars = mean contribution",
        transform=axis.transAxes,
        color="#667085",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_subsystem_summary(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows})
    subsystems = sorted({str(row["subsystem"]) for row in rows})
    x = np.arange(len(subsystems))
    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 5.2))
    for model_index, model in enumerate(models):
        values = []
        for subsystem in subsystems:
            match = next(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["subsystem"] == subsystem
                ),
                None,
            )
            values.append(float(match["total_contribution"]) if match else 0.0)
        colors = [subsystem_color(subsystem) for subsystem in subsystems]
        axis.bar(
            x + (model_index - 0.5) * width,
            values,
            width,
            label=model,
            color=colors,
            alpha=0.90 if model_index == 0 else 0.55,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(subsystems)
    axis.set_ylabel("Total top-5 contribution mass")
    axis.set_title("Subsystem-Level Explanation Mass")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_events(
    artifact_dir: Path,
    model_keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    for model_key in model_keys:
        path = artifact_dir / model_key / "events.json"
        if path.exists():
            events[model_label(model_key)] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return events


def _event_cards(events_by_model: dict[str, list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    for model, events in events_by_model.items():
        parts.append(f"<h3>{html.escape(model)} Events</h3>")
        for event in events[:6]:
            channels = ", ".join(
                f"{item['channel']} ({item['subsystem']})"
                for item in event.get("top_channels", [])[:3]
            )
            parts.append(
                f"""
<details class="event-card">
  <summary><strong>{html.escape(event["event_id"])}</strong>
  <span>{html.escape(event["risk_level"])}</span>
  <small>{html.escape(event["peak_time"])}</small></summary>
  <p><b>Interval:</b> {html.escape(event["start"])} — {html.escape(event["end"])}</p>
  <p><b>Peak score:</b> {_fmt(float(event["peak_score"]), 4)}</p>
  <p><b>Context:</b> {html.escape(str(event.get("context", {})))}</p>
  <p><b>Top channels:</b> {html.escape(channels)}</p>
</details>
"""
            )
    return "\n".join(parts)


def _load_events_themed(
    artifact_dir: Path,
    model_keys: list[str],
    warnings: list[str],
) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    for model_key in model_keys:
        path = artifact_dir / model_key / "events.json"
        if path.exists():
            events[model_label(model_key)] = json.loads(
                path.read_text(encoding="utf-8")
            )
        else:
            warnings.append(f"Missing optional events artifact: {path}")
    return events


def _dominant_subsystem(event: dict[str, Any]) -> str:
    totals: dict[str, float] = {}
    for item in event.get("top_channels", []):
        subsystem = str(item.get("subsystem", "UNKNOWN"))
        totals[subsystem] = totals.get(subsystem, 0.0) + float(
            item.get("contribution", 0.0)
        )
    return max(totals, key=totals.get) if totals else "UNKNOWN"


def _channel_badge_list(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items[:5]:
        subsystem = str(item.get("subsystem", "UNKNOWN"))
        contribution = float(item.get("contribution", 0.0))
        percent = max(0.0, min(contribution * 100.0, 100.0))
        parts.append(
            f"""
<li>
  {_subsystem_badge(subsystem)}
  <span class="channel-name">{html.escape(str(item.get("channel", "")))}</span>
  <span class="bar"><i style="width:{percent:.1f}%;background:{subsystem_color(subsystem)}"></i></span>
  <b>{percent:.0f}%</b>
</li>
"""
        )
    return f'<ul class="channel-list">{"".join(parts)}</ul>' if parts else "<p>Not available.</p>"


def _event_cards_themed(
    events_by_model: dict[str, list[dict[str, Any]]],
    *,
    max_cards: int = 20,
) -> str:
    parts: list[str] = []
    emitted = 0
    for model, events in events_by_model.items():
        parts.append(f"<h3>{html.escape(model)} Events</h3>")
        for event in events:
            if emitted >= max_cards:
                break
            emitted += 1
            dominant = _dominant_subsystem(event)
            color = subsystem_color(dominant)
            parts.append(
                f"""
<details class="event-card" style="border-left: 6px solid {color}">
  <summary><strong>{html.escape(str(event.get("event_id", "")))}</strong>
  {_risk_badge(event.get("risk_level", "LOW"))}
  {_subsystem_badge(dominant)}
  <small>{html.escape(str(event.get("peak_time", "")))}</small></summary>
  <p><b>Interval:</b> {html.escape(str(event.get("start", "")))} - {html.escape(str(event.get("end", "")))}</p>
  <p><b>Peak score:</b> {_fmt(float(event.get("peak_score", 0.0)), 4)}</p>
  <p><b>Context:</b> {html.escape(str(event.get("context", {})))}</p>
  <p><b>Top channels:</b></p>
  {_channel_badge_list(event.get("top_channels", []))}
</details>
"""
            )
    return "\n".join(parts) if parts else "<p>No event cards available.</p>"


def _top_takeaways(
    event_rows: list[dict[str, str | float]],
    false_positives: list[dict[str, str | float]],
) -> str:
    hit_rate = 0.0
    if event_rows:
        hit_rate = sum(row["channel_hit"] == "yes" for row in event_rows) / len(event_rows)
    fp_modes = sorted({str(row["operational_mode"]) for row in false_positives})
    fp_text = ", ".join(fp_modes) if fp_modes else "none"
    return f"""
<ul class="takeaways">
  <li><b>Channel explanation hit rate:</b> {_fmt(hit_rate)} across model-event matches.</li>
  <li><b>False-positive operational modes:</b> {html.escape(fp_text)}.</li>
  <li><b>Reading tip:</b> green delay bars mean the expected injected channel appeared in the model's top-3 explanation.</li>
</ul>
"""


def render_synthetic_dashboard(
    *,
    comparison: dict[str, Any],
    artifact_dir: Path,
    dashboard_path: Path,
    manifest_path: Path | None = None,
) -> Path:
    """Generate CSV tables, plots and a static HTML dashboard."""

    dashboard_artifacts = artifact_dir / "dashboard"
    warnings: list[str] = []
    variant_groups = dashboard_variant_groups(comparison)
    default_keys = variant_groups["default"]
    advanced_keys = variant_groups["advanced"]
    default_labels = {model_label(key) for key in default_keys}
    default_comparison = {
        "dataset": comparison.get("dataset", {}),
        **{key: comparison[key] for key in default_keys},
    }
    advanced_comparison = {
        "dataset": comparison.get("dataset", {}),
        **{key: comparison[key] for key in advanced_keys},
    }
    model_rows = experiment_model_rows(default_comparison)
    advanced_model_rows = experiment_model_rows(advanced_comparison)
    all_model_rows = experiment_model_rows(comparison)
    sweep_rows = threshold_sweep_rows(comparison)
    delays = delay_rows(default_comparison)
    model_keys = [key for key in comparison if key != "dataset"]
    resolved_manifest_path = manifest_path or Path(
        "data/synthetic/injection_manifest.json"
    )
    if not resolved_manifest_path.exists():
        warnings.append(f"Missing optional injection manifest: {resolved_manifest_path}")
    event_rows = event_diagnostic_rows(
        default_comparison,
        artifact_dir,
        resolved_manifest_path,
    )
    false_positives = false_positive_rows(default_comparison, artifact_dir)
    channel_rows = channel_summary_rows(artifact_dir, default_keys)
    subsystem_rows = subsystem_summary_rows(artifact_dir, default_keys)
    advanced_event_rows = event_diagnostic_rows(
        advanced_comparison,
        artifact_dir,
        resolved_manifest_path,
    ) if advanced_keys else []
    advanced_false_positives = (
        false_positive_rows(advanced_comparison, artifact_dir) if advanced_keys else []
    )

    write_csv(dashboard_artifacts / "model_comparison.csv", model_rows)
    write_csv(dashboard_artifacts / "model_comparison_all.csv", all_model_rows)
    write_csv(dashboard_artifacts / "threshold_sweep.csv", sweep_rows)
    write_csv(dashboard_artifacts / "detection_delays.csv", delays)
    write_csv(dashboard_artifacts / "event_diagnostics.csv", event_rows)
    write_csv(dashboard_artifacts / "false_positives.csv", false_positives)
    write_csv(dashboard_artifacts / "channel_summary.csv", channel_rows)
    write_csv(dashboard_artifacts / "subsystem_summary.csv", subsystem_rows)
    write_results_markdown(dashboard_artifacts / "results_summary.md", comparison)

    comparison_plot = dashboard_artifacts / "model_comparison.png"
    operational_plot = dashboard_artifacts / "false_alarm_delay.png"
    threshold_plot = dashboard_artifacts / "threshold_sweep.png"
    delay_plot = dashboard_artifacts / "detection_delays.png"
    event_plot = dashboard_artifacts / "event_diagnostics.png"
    channel_plot = dashboard_artifacts / "channel_summary.png"
    subsystem_plot = dashboard_artifacts / "subsystem_summary.png"
    _plot_model_comparison(model_rows, comparison_plot)
    _plot_false_alarm_delay(model_rows, operational_plot)
    _plot_threshold_sweep(sweep_rows, threshold_plot)
    _plot_detection_delays(delays, delay_plot)
    _plot_event_diagnostics(event_rows, event_plot)
    _plot_channel_summary(channel_rows, channel_plot)
    _plot_subsystem_summary(subsystem_rows, subsystem_plot)

    dataset = comparison["dataset"]
    best_key = max(
        default_keys or model_keys,
        key=lambda key: _operational_score(key, comparison[key]),
    )
    best_payload = comparison[best_key]
    best_label = model_label(best_key)
    best_row = next(
        row for row in all_model_rows if str(row["model"]) == best_label
    )
    best_channel = max(all_model_rows, key=lambda row: float(row["channel_hit_at_3"]))
    best_event = max(all_model_rows, key=lambda row: float(row["event_f1"]))
    low_noise = min(all_model_rows, key=lambda row: float(row["false_alarms_per_day"]))
    score_variant = (
        "pca_global"
        if "pca_global" in comparison
        else model_keys[0]
    )
    heatmap_variant = (
        "dense_autoencoder_global"
        if "dense_autoencoder_global" in comparison
        else (
            "tcn_autoencoder_global"
            if "tcn_autoencoder_global" in comparison
            else model_keys[0]
        )
    )
    pca_score = artifact_dir / score_variant / "plots" / "score_timeline.png"
    ae_heatmap = (
        artifact_dir / heatmap_variant / "plots" / "channel_error_heatmap.png"
    )
    if not pca_score.exists():
        pca_score = artifact_dir / score_variant / "score_timeline.png"
    if not ae_heatmap.exists():
        ae_heatmap = artifact_dir / heatmap_variant / "channel_error_heatmap.png"
    default_events = _load_events_themed(artifact_dir, default_keys, warnings)
    advanced_events = _load_events_themed(artifact_dir, advanced_keys, warnings)

    html_text = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAK Synthetic Experiment Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --blue: #235789;
      --green: #2f6f73;
      --amber: #f2a541;
      --red: #d1495b;
      --eps: {SUBSYSTEM_COLORS["EPS"]};
      --thermal: {SUBSYSTEM_COLORS["THERMAL"]};
      --aocs: {SUBSYSTEM_COLORS["AOCS"]};
      --comm: {SUBSYSTEM_COLORS["COMM"]};
      --payload: {SUBSYSTEM_COLORS["PAYLOAD"]};
      --unknown: {SUBSYSTEM_COLORS["UNKNOWN"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 32px 40px;
      color: white;
      background: linear-gradient(135deg, #172033 0%, #235789 55%, #2f6f73 100%);
    }}
    header p {{ max-width: 920px; color: #dbeafe; line-height: 1.55; }}
    main {{ padding: 24px 40px 48px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
    .tab-button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 700;
    }}
    .tab-button.active {{ color: white; background: var(--blue); border-color: var(--blue); }}
    .section {{ display: none; }}
    .section.active {{ display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
    .card, .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(23, 32, 51, 0.06);
    }}
    .card {{ padding: 18px; }}
    .card small, .muted {{ color: var(--muted); }}
    .card strong {{ display: block; font-size: 28px; margin-top: 8px; }}
    .card em {{ display: block; color: var(--muted); font-size: 12px; font-style: normal; margin-top: 6px; }}
    .panel {{ padding: 22px; margin: 18px 0; }}
    .panel h2 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    img {{ width: 100%; border-radius: 14px; border: 1px solid var(--line); background: white; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .event-card {{ padding: 12px 14px; margin: 10px 0; border: 1px solid var(--line); border-radius: 14px; background: #fbfdff; }}
    .event-card summary {{ cursor: pointer; display: flex; align-items: center; gap: 12px; }}
    .subsystem-badge, .risk-badge {{
      display: inline-block;
      color: white;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .02em;
    }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 0; list-style: none; }}
    .channel-list {{ list-style: none; padding: 0; margin: 8px 0; }}
    .channel-list li {{ display: grid; grid-template-columns: 92px 1fr 130px 48px; gap: 10px; align-items: center; margin: 8px 0; }}
    .channel-name {{ font-family: Consolas, monospace; }}
    .bar {{ display: block; height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; border-radius: 999px; }}
    .warning {{ border-color: #F59E0B; background: #FFFBEB; }}
    details.advanced {{ margin: 18px 0; }}
    details.advanced > summary {{ cursor: pointer; font-size: 18px; font-weight: 800; }}
    .takeaways {{ padding-left: 20px; line-height: 1.6; }}
    .artifact-list a {{ color: var(--blue); font-weight: 700; text-decoration: none; }}
    .artifact-list li {{ margin: 8px 0; }}
    @media (max-width: 960px) {{
      main, header {{ padding-left: 20px; padding-right: 20px; }}
      .grid, .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SAK — Synthetic Experiment Dashboard</h1>
    <p>Uydu telemetri anomali tespiti için ilk PCA ve Dense Autoencoder sonuçları.
    Bu dashboard model performansını, erken uyarı kalitesini, açıklanabilirlik
    metriklerini ve ilk görselleri tek ekranda kontrol etmek için üretildi.</p>
  </header>
  <main>
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab-button active" data-tab="overview">Overview</button>
      <button class="tab-button" data-tab="models">Model Tables</button>
      <button class="tab-button" data-tab="events">Events & XAI</button>
      <button class="tab-button" data-tab="diagnostics">Diagnostics</button>
      <button class="tab-button" data-tab="advanced">Advanced / Legacy</button>
      <button class="tab-button" data-tab="artifacts">Artifacts</button>
    </nav>

    <section id="overview" class="section active">
      {_warning_panel(warnings)}
      <div class="grid">
        {_metric_card("Rows", f"{dataset['rows']:,}", "synthetic telemetry")}
        {_metric_card("Telemetry channels", str(dataset["channels"]), "scaled train-only")}
        {_metric_card("Test events", str(dataset["test_events"]), "controlled injections")}
        {_metric_card("Best XAI Hit@3", str(best_channel["model"]), "top channel attribution")}
      </div>
      <div class="panel">
        <h2>SAK-v2.5 Executive Summary</h2>
        <p><b>Best current operating point:</b> {html.escape(best_label)}.
        Event recall {_fmt(float(best_row["event_recall"]))},
        critical-region recall {_fmt(float(best_row["critical_region_recall"]))},
        before-critical rate {_fmt(float(best_row["detected_before_critical_rate"]))},
        false alarms/day {_fmt(float(best_row["false_alarms_per_day"]))},
        median lead time {_fmt(float(best_row["median_lead_time_to_critical_min"]))} min,
        Channel Hit@3 {_fmt(float(best_row["channel_hit_at_3"]))}.</p>
        <p class="muted">Operational score is only used for dashboard ranking;
        threshold/filter selection remains calibration-only.</p>
        <h3>Subsystem Legend</h3>
        {_subsystem_legend()}
      </div>
      <div class="panel">
        <h2>Executive Readout</h2>
        <p><b>Best event F1:</b> {html.escape(str(best_event["model"]))}
        ({_fmt(float(best_event["event_f1"]))}). <b>Lowest false alarm rate:</b>
        {html.escape(str(low_noise["model"]))}
        ({_fmt(float(low_noise["false_alarms_per_day"]))}/day).</p>
        {_top_takeaways(event_rows, false_positives)}
        <p class="muted">Not: Bu sonuçlar sentetik veri üzerindedir; gerçek görev
        verisi için yalnız metodoloji ve UI altyapısı olarak değerlendirilmelidir.</p>
      </div>
      <div class="two-col">
        <div class="panel"><h2>Model Comparison</h2><img src="{_relative(comparison_plot, dashboard_path)}" alt="Model comparison chart"></div>
        <div class="panel"><h2>False Alarm & Delay</h2><img src="{_relative(operational_plot, dashboard_path)}" alt="False alarm and delay chart"></div>
      </div>
    </section>

    <section id="models" class="section">
      <div class="panel"><h2>Model Metrics</h2>{_html_table(model_rows)}</div>
      <div class="panel"><h2>Threshold Sweep</h2><img src="{_relative(threshold_plot, dashboard_path)}" alt="Threshold sweep chart">{_html_table(sweep_rows)}</div>
      <div class="panel"><h2>Detection Delays</h2><img src="{_relative(delay_plot, dashboard_path)}" alt="Detection delay chart">{_html_table(delays)}</div>
    </section>

    <section id="events" class="section">
      <div class="two-col">
        <div class="panel"><h2>PCA Score Timeline</h2><img src="{_relative(pca_score, dashboard_path)}" alt="PCA score timeline"></div>
        <div class="panel"><h2>Autoencoder Heatmap</h2><img src="{_relative(ae_heatmap, dashboard_path)}" alt="Autoencoder channel heatmap"></div>
      </div>
      <div class="panel"><h2>Event Diagnostics</h2><img src="{_relative(event_plot, dashboard_path)}" alt="Event diagnostics chart">{_html_table(event_rows)}</div>
      <div class="panel"><h2>Predicted Events</h2>{_event_cards_themed(default_events, max_cards=20)}</div>
    </section>

    <section id="diagnostics" class="section">
      <div class="two-col">
        <div class="panel"><h2>Top Explanation Channels</h2><img src="{_relative(channel_plot, dashboard_path)}" alt="Channel summary chart"></div>
        <div class="panel"><h2>Subsystem Explanation Mass</h2><img src="{_relative(subsystem_plot, dashboard_path)}" alt="Subsystem contribution chart"></div>
      </div>
      <div class="panel"><h2>False Positive Events</h2>{_html_table(false_positives)}</div>
      <div class="panel"><h2>Channel Summary Table</h2>{_html_table(channel_rows)}</div>
      <div class="panel"><h2>Subsystem Summary Table</h2>{_html_table(subsystem_rows)}</div>
    </section>

    <section id="advanced" class="section">
      <details class="advanced" open>
        <summary>Mode-aware, fixed-quantile and sweep diagnostics</summary>
        <div class="panel"><h2>All Model Metrics</h2>{_html_table(all_model_rows)}</div>
        <div class="panel"><h2>Advanced Model Metrics</h2>{_html_table(advanced_model_rows)}</div>
        <div class="panel"><h2>Threshold / Filter Sweep</h2>{_html_table(sweep_rows)}</div>
        <div class="panel"><h2>Advanced Event Diagnostics</h2>{_html_table(advanced_event_rows)}</div>
        <div class="panel"><h2>Advanced False Positives</h2>{_html_table(advanced_false_positives)}</div>
        <div class="panel"><h2>Advanced Predicted Events</h2>{_event_cards_themed(advanced_events, max_cards=20)}</div>
      </details>
    </section>

    <section id="artifacts" class="section">
      <div class="panel artifact-list">
        <h2>Generated Files</h2>
        <ul>
          <li><a href="{_relative(dashboard_artifacts / "model_comparison.csv", dashboard_path)}">model_comparison.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "threshold_sweep.csv", dashboard_path)}">threshold_sweep.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "detection_delays.csv", dashboard_path)}">detection_delays.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "event_diagnostics.csv", dashboard_path)}">event_diagnostics.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "false_positives.csv", dashboard_path)}">false_positives.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "channel_summary.csv", dashboard_path)}">channel_summary.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "subsystem_summary.csv", dashboard_path)}">subsystem_summary.csv</a></li>
          <li><a href="{_relative(dashboard_artifacts / "results_summary.md", dashboard_path)}">results_summary.md</a></li>
          <li><a href="../docs/experiment-001-synthetic-baselines.md">experiment-001-synthetic-baselines.md</a></li>
        </ul>
      </div>
    </section>
  </main>
  <script>
    const buttons = document.querySelectorAll('.tab-button');
    const sections = document.querySelectorAll('.section');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        sections.forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def render_real_dashboard(
    *,
    comparison: dict[str, Any],
    artifact_dir: Path,
    dashboard_path: Path,
    dataset_metadata: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> Path:
    """Generate a static HTML dashboard for real-data benchmark runs."""

    dashboard_artifacts = artifact_dir / "dashboard"
    dashboard_artifacts.mkdir(parents=True, exist_ok=True)
    dataset = comparison.get("dataset", {})
    metadata = dataset_metadata or {}
    split = split_manifest or {}
    model_keys = [key for key in comparison if key != "dataset"]
    model_rows = _real_model_rows(comparison)
    write_csv(dashboard_artifacts / "model_comparison.csv", model_rows)
    events_by_model = _load_events_themed(artifact_dir, model_keys, warnings=[])
    event_cards = _event_cards_themed(events_by_model, max_cards=24)
    limitations = (
        "Critical-region lead-time metrics are unavailable or proxy unless the "
        "source dataset provides critical/failure annotations."
    )
    critical_available = bool(
        dataset.get(
            "critical_region_available",
            metadata.get("critical_region_available", False),
        )
    )
    partition_rows = [
        {
            "partition": name,
            "rows": values.get("rows", 0),
            "anomaly_rows": values.get("anomaly_rows", 0),
            "events": values.get("event_count", 0),
        }
        for name, values in split.get("partitions", {}).items()
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAK Real Dataset Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --blue: #235789;
      --green: #2f6f73;
      --amber: #f2a541;
      --unknown: {SUBSYSTEM_COLORS["UNKNOWN"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 28px 36px 24px;
      color: white;
      background: #172033;
      border-bottom: 5px solid var(--green);
    }}
    header p {{ max-width: 940px; color: #dbeafe; line-height: 1.5; }}
    main {{ padding: 24px 36px 44px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(23, 32, 51, 0.05);
    }}
    .panel {{ padding: 20px; margin: 16px 0; }}
    .card {{ padding: 16px; }}
    .card small, .muted {{ color: var(--muted); }}
    .card strong {{ display: block; font-size: 26px; margin-top: 7px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .banner {{
      border: 1px solid #F59E0B;
      background: #FFFBEB;
      border-radius: 8px;
      padding: 14px 16px;
      margin: 16px 0;
      font-weight: 700;
    }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .event-card {{ padding: 12px 14px; margin: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; }}
    .event-card summary {{ cursor: pointer; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .subsystem-badge, .risk-badge {{
      display: inline-block;
      color: white;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
    }}
    .channel-list {{ list-style: none; padding: 0; margin: 8px 0; }}
    .channel-list li {{ display: grid; grid-template-columns: 92px 1fr 120px 46px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar {{ display: block; height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; border-radius: 999px; }}
    @media (max-width: 960px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .grid, .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SAK-v3.0 Real Dataset Benchmark</h1>
    <p>NASA SMAP/MSL real-data adapter output. Test labels are held out for final
    evaluation and are never used for threshold selection.</p>
  </header>
  <main>
    <div class="banner">{html.escape(limitations)}</div>
    <div class="grid">
      {_metric_card("Source", str(dataset.get("source", "nasa_smap_msl")), "real telemetry")}
      {_metric_card("Channel ID", str(dataset.get("channel_id", "all")), "source channel")}
      {_metric_card("Rows", f"{int(dataset.get('rows', 0)):,}", "canonical samples")}
      {_metric_card("Critical Region", "available" if critical_available else "not available / proxy", "lead-time status")}
    </div>
    <div class="panel">
      <h2>Dataset Summary</h2>
      <p><b>Layout:</b> {html.escape(str(dataset.get("source_layout", metadata.get("source_layout", ""))))}
      &nbsp; <b>Channels:</b> {html.escape(str(dataset.get("channels", 0)))}
      &nbsp; <b>Events:</b> {html.escape(str(dataset.get("events", 0)))}
      &nbsp; <b>Timestamp synthetic:</b> {html.escape(str(dataset.get("timestamp_synthetic", metadata.get("timestamp_synthetic", False))))}
      &nbsp; <b>Subsystem fallback:</b> {_subsystem_badge("UNKNOWN")}</p>
      {_html_table(partition_rows)}
    </div>
    <div class="panel">
      <h2>Model Comparison</h2>
      {_html_table(model_rows)}
    </div>
    <div class="two-col">
      <div class="panel">
        <h2>Metric Availability</h2>
        <table><tbody>
          <tr><th>Available</th><td>Point precision/recall/F1, event precision/recall/F1, false alarms/day, detection delay, reconstruction channel ranking.</td></tr>
          <tr><th>Proxy</th><td>Critical-region recall and lead-time fields when source critical/failure annotations are absent.</td></tr>
          <tr><th>Unavailable</th><td>Subsystem hit metrics unless a trusted source-channel mapping is provided.</td></tr>
        </tbody></table>
      </div>
      <div class="panel">
        <h2>Leakage Guard</h2>
        <p>Selection partition: calibration. Test used for selection:
        <b>{html.escape(str(dataset.get("test_used_for_selection", False)))}</b>.</p>
        <p class="muted">If calibration contains no labeled anomaly events, operating
        point metadata reports <code>no_calibration_events</code> and falls back to
        nominal false-alarm suppression.</p>
      </div>
    </div>
    <div class="panel">
      <h2>Real Event Timeline</h2>
      <p class="muted">Cards show predicted alarm intervals, source event IDs when
      matched, false positives, and UNKNOWN subsystem fallback where no mapping exists.</p>
      {event_cards}
    </div>
  </main>
</body>
</html>
"""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(html_text, encoding="utf-8")
    return dashboard_path


def _real_model_rows(comparison: dict[str, Any]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for model_key, payload in comparison.items():
        if model_key == "dataset":
            continue
        event = payload.get("event_metrics", {})
        point = payload.get("point_metrics", {})
        xai = payload.get("xai_metrics", {})
        rows.append(
            {
                "model": model_label(model_key),
                "event_recall": event.get("recall", 0.0),
                "event_f1": event.get("f1", 0.0),
                "false_alarms_per_day": event.get("false_alarms_per_day", 0.0) or 0.0,
                "detection_delay_min": event.get("median_detection_delay_minutes", 0.0)
                or 0.0,
                "point_f1": point.get("f1", 0.0),
                "channel_hit_at_3": xai.get("channel_hit_at_3", 0.0),
                "critical_region_status": str(
                    event.get("critical_region_metric_status", "proxy")
                ),
            }
        )
    return rows
