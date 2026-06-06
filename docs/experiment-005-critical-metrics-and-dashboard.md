# Experiment 005 - Critical Metrics And Dashboard Cleanup

## Motivation

SAK-v2.4 introduced anomalous calibration and held-out test evaluation, but the
headline critical metric could still behave like ordinary event recall. For an
early-warning system, detecting an anomaly after the critical region is over is
not equivalent to warning before or during the critical window.

SAK-v2.5 fixes that distinction and makes the dashboard easier to read for
engineering triage.

## Metric Definitions

- `event_recall`: fraction of true anomaly events with any matched predicted
  event inside the early-warning-to-end-plus-tolerance match window.
- `detected_before_critical_rate`: fraction of true anomaly events whose first
  matched alarm starts before `failure_region_start`.
- `critical_region_recall`: fraction of true anomaly events whose matched alarm
  starts before critical start or overlaps the critical region while active.
- `precursor_detection_rate`: fraction of true anomaly events detected after
  `early_warning_region_start` and before anomaly onset.
- `missed_critical_count`: events that were not warned before or during the
  critical region.
- `median_lead_time_to_critical_minutes`: median time from alarm start to
  critical-region start. Positive is early; negative is late.
- `p10_lead_time_to_critical_minutes`: lower-tail lead time for worst-case
  behavior.
- `late_detection_rate`: fraction of true events whose matched alarm starts
  after the critical region begins.

## Before-Critical Vs Critical-Covered

Before-critical is stricter for early warning: the alarm begins before the
critical region starts. Critical-covered means the alarm overlaps the critical
region. A late alarm can cover the critical region but still fail the
before-critical objective.

An event can now have `event_recall = 1` and `critical_region_recall = 0` when
the prediction matches only after the event window and does not cover the
critical region.

## Operating Point Selection

Calibration-only selection now enforces:

- event recall >= configured minimum;
- critical-region recall >= configured minimum;
- before-critical rate >= configured minimum;
- false alarms/day <= configured maximum.

Feasible candidates are ranked by event F1, critical recall, before-critical
rate, lead time, false alarms, delay, point F1 and Channel Hit@3. Test metrics
are not used for selection.

## Dashboard Cleanup

The default dashboard view now focuses on selected global operating points:

- `pca_global`;
- `dense_autoencoder_global`;
- `tcn_autoencoder_global`.

Mode-aware variants, fixed-quantile comparisons, threshold sweeps and detailed
diagnostics remain available in the Advanced / Legacy section. They are hidden
by default to reduce visual noise, not deleted.

## Subsystem Theme

Subsystem colors are stable across the UI:

- EPS: amber `#F59E0B`;
- THERMAL: red `#EF4444`;
- AOCS: blue `#3B82F6`;
- COMM: purple `#8B5CF6`;
- PAYLOAD: green `#10B981`;
- UNKNOWN: gray `#6B7280`.

Event cards use the dominant subsystem as a left border. Top-channel lists show
subsystem badges and contribution bars.

## NASA SMAP/MSL Adapter

The NASA adapter can now inspect candidate roots, detect normalized
`telemetry.csv` or `telemetry.parquet` staging files, report label availability
and load normalized tabular telemetry. Raw Telemanom `.npy` arrays are
inspectable but still require sequence and channel-name mapping before full
benchmark evaluation.

## Remaining Limitations

Dashboard operational score is a UI ranking only. It is not used for threshold
selection. Synthetic results still cannot support real mission performance
claims. The next validation step is to stage NASA SMAP/MSL sequences into the
canonical telemetry contract and run the same leakage-safe evaluation.
