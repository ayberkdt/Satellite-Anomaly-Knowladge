# Experiment 006 - NASA SMAP/MSL Real-Data Adapter

## Why Real Data Now?

SAK-v2.5 made the synthetic pipeline useful for controlled evidence: known
injections, leakage-safe splits, calibration-only operating-point selection and
dashboard artifacts. SAK-v3.0 adds the first bridge to open real spacecraft
telemetry so the same benchmark protocol can be exercised outside synthetic
assumptions.

Synthetic results remain method validation, not mission-performance claims.
NASA SMAP/MSL results are benchmark evidence for an open dataset, not evidence
about TUSAS or any private mission.

## NASA Adapter Contract

`NasaSmapMslAdapter` exposes:

```python
inspect(path: Path) -> DatasetInspection
list_channels(path: Path) -> list[str]
load(path: Path, *, channel_id: str | None = None) -> TelemetryDataset
```

`DatasetInspection` reports existence, support status, detected layout, channel
count, label availability, anomaly interval availability, warnings and errors.
Unsupported layouts fail with `UnsupportedDatasetLayoutError`. Missing paths
fail with `AdapterDataNotFoundError`.

## Dataset Inspection

The inspection CLI prints JSON and can optionally write the same payload:

```powershell
python experiments/inspect_dataset_adapter.py --adapter nasa_smap_msl --path data/raw/smap_msl
python experiments/inspect_dataset_adapter.py --adapter nasa_smap_msl --path data/raw/smap_msl --output artifacts/real/nasa_smap_msl/adapter_inspection.json
```

Supported layouts are:

- normalized `telemetry.csv` or `telemetry.parquet` staging folders;
- Telemanom-style `train/*.npy` and `test/*.npy` source-channel arrays;
- optional `labeled_anomalies.csv`, `labels.csv` or `anomaly_intervals.csv`.

## Canonical Mapping

Every loaded source returns `TelemetryDataset`:

- `frame`: timestamp-indexed canonical telemetry frame;
- `channel_names`: numeric telemetry channel columns;
- `context_columns`: available context such as `operational_mode`,
  `source_channel_id` and `partition`;
- `events`: canonical real event records;
- `metadata`: adapter inspection and source limitations.

Required frame columns are filled conservatively:

- `is_anomaly`: boolean;
- `anomaly_event_id`: empty string when nominal;
- `anomaly_type`: empty string or source anomaly class;
- `label_taxonomy`: `nominal` or `anomaly`;
- `operational_mode`: `unknown` when absent;
- `source_channel_id`: selected source channel or `all`.

If timestamps are absent, SAK creates a deterministic one-minute synthetic
timestamp index and records `timestamp_synthetic: true` and
`sample_period_unknown: true`.

## Split Strategy

Real-data mode supports `source_train_test` first:

- source train -> train, calibration and validation;
- source test -> final held-out test;
- train must be nominal-only;
- test is never used for threshold or filter selection.

If a source does not expose train/test partitions, SAK can fall back to
`chronological_train_calibration_validation_test`.

The default real split settings are:

```yaml
real_data_split:
  strategy: source_train_test
  calibration_fraction_from_train: 0.20
  validation_fraction_from_train: 0.10
  use_test_for_selection: false
```

## Calibration Limitation

NASA SMAP/MSL source train data is commonly nominal. If the derived calibration
partition contains no true anomaly events, constrained event-F1 selection cannot
tune event constraints. SAK records:

```json
{
  "selection_partition": "calibration",
  "calibration_true_events": 0,
  "constraints_satisfied": false,
  "selection_reason": "no_calibration_events",
  "test_partition_used_for_selection": false
}
```

Thresholds still come from nominal calibration scores. Test labels remain final
evaluation only.

## First Benchmark Protocol

Run:

```powershell
python experiments/run_real_dataset_models.py --adapter nasa_smap_msl --data data/raw/smap_msl --models pca dense_autoencoder
python experiments/run_real_dataset_models.py --adapter nasa_smap_msl --data data/raw/smap_msl --channel-id P-1 --models pca dense_autoencoder tcn_autoencoder --render-dashboard
```

The runner writes:

- `comparison.json` and `comparison.csv`;
- `data_quality_report.json`;
- `split_manifest.json`;
- `run_manifest.json`;
- per-model scores, predictions, metrics and diagnostics;
- `dashboard.html` when `--render-dashboard` is supplied.

## Dashboard Real-Data Mode

The real dashboard is independent of the synthetic injection manifest. It shows:

- dataset source, channel ID, row count and critical-region availability;
- train/calibration/validation/test rows and anomaly rows;
- PCA, Dense AE and TCN comparison rows;
- real predicted event cards and source event IDs when matched;
- an explicit limitations banner.

UNKNOWN subsystem badges are used when no trusted source mapping exists.

## Metrics Available

Available on real data when labels exist:

- point precision, recall and F1;
- event precision, recall and F1;
- false alarms per day;
- median detection delay;
- channel reconstruction-error ranking.

## Metrics Proxy Or Unavailable

Critical-region and lead-time metrics are proxy/unavailable unless the source
dataset provides critical/failure annotations. NASA SMAP/MSL interval labels do
not provide those regions, so SAK maps early-warning and failure starts to the
event start and marks `critical_region_metric_status: proxy`.

Subsystem hit metrics are unavailable unless a trusted
`configs/subsystems_nasa_smap_msl.yaml` mapping is populated. The default file
does not invent subsystem assignments.

## Remaining Limitations

- Only the first real-data bridge is implemented.
- Source-channel arrays are evaluated one selected channel at a time.
- NASA interval labels do not identify critical/failure regions.
- Orbit, mode and subsystem context are often unavailable.
- Dense AE and TCN behavior depends on source-channel dimensionality and sample
  count.

## Next Step

The next data step should be ESA-ADB adapter completion or richer NASA channel
batching. GNN/GAT should remain deferred until real-data baseline behavior and
critical-region annotation limitations are well understood.
