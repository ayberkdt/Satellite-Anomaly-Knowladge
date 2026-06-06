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

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


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


def _metric_card(title: str, value: str, note: str = "") -> str:
    return (
        f'<div class="card"><small>{html.escape(title)}</small>'
        f"<strong>{html.escape(value)}</strong>"
        f"<em>{html.escape(note)}</em></div>"
    )


def _plot_model_comparison(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in rows]
    metrics = [
        ("event_f1", "Event F1"),
        ("point_f1", "Point F1"),
        ("channel_hit_at_3", "Channel Hit@3"),
        ("subsystem_hit_at_2", "Subsystem Hit@2"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5.2))
    colors = ["#2f6f73", "#7a4fe0", "#f2a541", "#d1495b"]
    for offset, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        axis.bar(x + (offset - 1.5) * width, values, width, label=label, color=colors[offset])
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
    labels = [str(row["model"]) for row in rows]
    false_alarms = [float(row["false_alarms_per_day"]) for row in rows]
    delays = [float(row["median_delay_min"]) for row in rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(x, false_alarms, color="#d1495b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("False alarms / day")
    axes[0].set_title("Operational Noise")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, delays, color="#235789")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_ylabel("Median delay (min)")
    axes[1].set_title("Early Warning Delay")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_threshold_sweep(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    axes[0].set_xlabel("Validation quantile")
    axes[0].set_ylabel("Event F1")
    axes[0].set_ylim(0.0, 1.08)
    axes[1].set_title("False Alarms vs Threshold Quantile")
    axes[1].set_xlabel("Validation quantile")
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
    x = np.arange(len(top_rows))
    figure, axis = plt.subplots(figsize=(13, 5.2))
    bars = axis.bar(x, counts, color="#235789")
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
        axis.bar(x + (model_index - 0.5) * width, values, width, label=model)
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
    model_rows = experiment_model_rows(comparison)
    sweep_rows = threshold_sweep_rows(comparison)
    delays = delay_rows(comparison)
    model_keys = [key for key in comparison if key != "dataset"]
    resolved_manifest_path = manifest_path or Path(
        "data/synthetic/injection_manifest.json"
    )
    event_rows = event_diagnostic_rows(
        comparison,
        artifact_dir,
        resolved_manifest_path,
    )
    false_positives = false_positive_rows(comparison, artifact_dir)
    channel_rows = channel_summary_rows(artifact_dir, model_keys)
    subsystem_rows = subsystem_summary_rows(artifact_dir, model_keys)

    write_csv(dashboard_artifacts / "model_comparison.csv", model_rows)
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
    best_channel = max(model_rows, key=lambda row: float(row["channel_hit_at_3"]))
    best_event = max(model_rows, key=lambda row: float(row["event_f1"]))
    low_noise = min(model_rows, key=lambda row: float(row["false_alarms_per_day"]))
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
    .event-card span {{ color: white; background: var(--red); border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
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
      <button class="tab-button" data-tab="artifacts">Artifacts</button>
    </nav>

    <section id="overview" class="section active">
      <div class="grid">
        {_metric_card("Rows", f"{dataset['rows']:,}", "synthetic telemetry")}
        {_metric_card("Telemetry channels", str(dataset["channels"]), "scaled train-only")}
        {_metric_card("Test events", str(dataset["test_events"]), "controlled injections")}
        {_metric_card("Best XAI Hit@3", str(best_channel["model"]), "top channel attribution")}
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
      <div class="panel"><h2>Predicted Events</h2>{_event_cards(_load_events(artifact_dir, model_keys))}</div>
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
