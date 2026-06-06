"""Engineering report generation."""

from sak.reporting.dashboard import render_real_dashboard, render_synthetic_dashboard
from sak.reporting.markdown import (
    build_early_warning_report_payload,
    render_early_warning_report,
    render_early_warning_report_payload,
)
from sak.reporting.results import (
    channel_summary_rows,
    delay_rows,
    event_diagnostic_rows,
    experiment_model_rows,
    false_positive_rows,
    markdown_table,
    subsystem_summary_rows,
    threshold_sweep_rows,
    write_results_markdown,
)

__all__ = [
    "channel_summary_rows",
    "build_early_warning_report_payload",
    "delay_rows",
    "event_diagnostic_rows",
    "experiment_model_rows",
    "false_positive_rows",
    "markdown_table",
    "render_early_warning_report",
    "render_early_warning_report_payload",
    "render_real_dashboard",
    "render_synthetic_dashboard",
    "subsystem_summary_rows",
    "threshold_sweep_rows",
    "write_results_markdown",
]
