# Experiment 004 - Anomalous Calibration And Data Realism

## Why V2.3 Was Not Enough

SAK-v2.3 could suppress nominal validation alarms, but its selection
partition contained no anomaly events. Event recall, critical recall and
delay therefore could not constrain operating-point selection. V2.4 fixes
the evaluation design before adding another model family.

## Data Suitability

The original synthetic data was sufficient for pipeline and artifact
validation, leakage checks, simple model comparison and controlled injection
tests. It was not sufficient for mission-performance claims, reliable
event-constrained calibration, industrial acceptance, proof of temporal
superiority or graph-topology learning.

V2.4 expands the frame to 29 correlated channels grouped into EPS, thermal,
AOCS, communications and payload subsystems. Context includes orbit phase,
eclipse, sunlight, beta angle, nominal/payload/maneuver/safe modes, maneuver
flags and safe-mode flags.

This remains synthetic evidence. Real performance claims require an open or
mission-specific dataset such as NASA SMAP/MSL or ESA-ADB.

## Four-Partition Split

| Partition | Fraction | Purpose |
|---|---:|---|
| Train | 0.50 | Nominal-only preprocessing and model fitting |
| Calibration | 0.20 | Threshold/filter operating-point selection |
| Validation | 0.10 | Independent post-selection sanity check |
| Test | 0.20 | Held-out final evaluation |

The generator schedules independent event instances in calibration,
validation and test. Event IDs are unique, no event crosses a boundary, and a
train anomaly raises an error.

## Event Taxonomy And Early Warning

Rows use `nominal`, `benign_transient`, `precursor`, `anomaly` and `critical`
labels. Manifest records include partition, severity, precursor start,
anomaly onset, critical-region start, affected channels and expected
subsystem.

Benign mode, eclipse and safe-mode transients remain nominal for detection.
They are intentional false-positive challenges.

Evaluation retains anomaly-onset delay and adds median lead time to critical,
critical-region recall, detection-before-critical rate and precursor
detection rate.

## Operating-Point Selection

Quantile and EWMA/persistence candidates are evaluated only on calibration.
Feasible candidates must satisfy event recall, critical-region recall and
false-alarms/day constraints. Ranking then prefers event F1, critical recall,
lead time, fewer false alarms, lower delay and point F1.

Validation and test metrics are absent from the selector API. Validation is
written as a sanity artifact; test is evaluated only after selection. Fixed
quantile and selected test results are both exported.

## Data Quality And Reproducibility

`data_quality_report.json` records missingness, partition lengths, event
counts, label ratios, channel groups, operational modes and leakage checks.
`split_manifest.json` records exact boundaries and states that test was not
used for selection.

Each model variant writes operating-point selection plus calibration,
validation and test partition metrics. Alarm score transforms affect only the
scalar alarm score; XAI uses raw channel reconstruction errors. Robust
z-score is not presented as an automatic quantile-performance improvement.

## TCN Ablation

`experiments/run_temporal_ablation.py` explores a bounded subset of the
configured window, stride, latent/hidden channel, kernel and depth grid.
Results include event/critical recall, F1, false alarms, delay, lead time,
point F1, Channel Hit@3, runtime and trained epochs. Ablation remains separate
from routine runs.

## Measured Results

Three-seed global-threshold means:

| Model / operating point | Recall | Critical recall | Event F1 | FA/day | Delay | Lead time | Hit@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense AE fixed | 0.367 | 0.367 | 0.536 | 0.000 | 33.50 | 7.17 | - |
| Dense AE selected | 0.950 | 0.950 | 0.839 | 2.262 | 2.33 | 35.33 | 0.842 |
| TCN AE fixed | 0.233 | 0.233 | 0.378 | 0.000 | 32.67 | 9.33 | - |
| TCN AE selected | 0.900 | 0.900 | 0.915 | 0.476 | 8.83 | 32.83 | 0.906 |

Selection materially improves both model families over fixed quantile. Dense
AE reaches higher recall and lower delay, but does so with 2.262 false
alarms/day. TCN provides the strongest operational trade-off: it satisfies the
recall constraints while holding false alarms below 0.50/day and preserving
positive critical-region lead time.

Calibration and validation behavior supports the selection design. Across
seeds, TCN global calibration recall was `1.0, 0.9, 1.0`, validation recall
was consistently `0.9`, and all three calibration choices satisfied the
configured constraints. PCA and Dense AE reached calibration recall 1.0 but
failed the false-alarm constraint, so their metadata correctly records
`constraints_satisfied: false`.

## Acceptance Criteria

Data acceptance requires nominal train, anomalous calibration, held-out
anomalous test, split/data-quality manifests and passing leakage checks.

Selected model targets are event recall and critical recall at least 0.90,
false alarms/day no more than 0.75, and test F1 no worse than fixed quantile.
Ideal targets add false alarms/day no more than 0.50, positive median lead
time and Channel Hit@3 at least 0.90.

The selected TCN global operating point passes both minimum and ideal
three-seed mean targets:

- event recall: `0.900`;
- critical-region recall: `0.900`;
- false alarms/day: `0.476`;
- median lead time: `+32.83` minutes;
- Channel Hit@3: `0.906`;
- selected event F1: `0.915`, above fixed quantile `0.378`.

Dense AE does not pass the false-alarm target despite recall `0.950`. TCN
global is therefore the most reliable current operating point. This is a
synthetic result, not a mission-performance claim.

## Remaining Limitations And Next Step

Synthetic equations are controlled approximations, not spacecraft failure
physics. Annotation noise, mission-specific modes and long-term degradation
remain underrepresented. The next step is to implement one real adapter,
preferably NASA SMAP/MSL for initial contract validation, then repeat
operating-point selection without changing the held-out-test rule.

GNN/GAT should begin only after real-data ingestion, stable baseline
operating points, channel-topology provenance and cross-mission evaluation.
