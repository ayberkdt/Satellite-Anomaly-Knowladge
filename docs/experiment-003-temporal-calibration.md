# Experiment 003 - Temporal Score Calibration

## Purpose

SAK-v2.2 showed that the TCN Autoencoder improved point F1 and median
detection delay, but did not reduce false alarms relative to Dense AE.
SAK-v2.3 therefore focuses on score calibration, threshold objectives,
persistence filtering and false-positive diagnosis rather than adding a new
model family.

## Leakage Control

Score transforms are fitted only on nominal validation scores. Threshold
quantiles, mode thresholds and alarm-filter candidates are also calibrated
only on validation. Test labels and test event metrics are used exclusively
for final evaluation.

The default synthetic validation partition contains no injected anomalies.
Consequently, validation event recall and event F1 are not identifiable.
`constrained_event_f1` still minimizes nominal validation false alarms, but
records:

```text
constraints_satisfied: false
selection_reason: no_validation_events
```

This limitation must not be hidden by selecting a default from test results.
A representative anomalous validation or calibration partition is required
before constrained event optimization can be considered fully validated.

## Score Distribution Diagnosis

Each model variant exports `diagnostics/score_distribution.json` with raw and
EWMA score quantiles, nominal/anomaly means, threshold margins, alarm score
ratios and false-positive context counts. Point-level context is written to
`diagnostics/false_positive_context.json`.

TCN supports:

- `none`: preserve the timestamp score scale;
- `log1p`: compress a positively skewed score tail;
- `robust_zscore`: normalize with nominal validation median and IQR;
- `mean` or `max` window-error aggregation;
- neutral filling of uncovered or explicitly trimmed edge timestamps.

## Threshold Objective And Filter Sweep

The default `quantile` strategy preserves the configured quantile and filter.
The optional `constrained_event_f1` strategy evaluates the candidate
quantiles and filter grid on validation.

Feasible candidates must satisfy minimum event recall and maximum false
alarms/day. They are ordered by:

1. higher event F1;
2. lower median detection delay;
3. lower false alarms/day;
4. higher point F1.

If no candidate is feasible, the fallback first preserves the highest recall
and then applies deterministic false-alarm, F1, delay and point-F1 ordering.
All evaluated rows and the single selected row are written to
`diagnostics/filter_sweep.csv`.

## False-Positive Context Analysis

`false_positive_diagnostics.csv` includes event duration, peak score,
threshold ratio, mode, eclipse state, orbit phase, top channels/subsystems,
nearest true event and temporal distance. `likely_reason` can be
`mode_transition`, `eclipse_boundary`, `threshold_margin_low`,
`isolated_spike`, `long_low_confidence_alarm` or `unknown`.

These labels are diagnostic hints. They support investigation but do not
claim physical root cause.

## Multi-Seed Results

Seeds 1, 2 and 3 use the same synthetic generator and model settings as
Experiment 002. V2.3 uses `constrained_event_f1`, the configured filter grid,
`mean` aggregation and no nonlinear score transform.

| Global model | Event recall | Event F1 | False alarms/day | Median delay | Point F1 | Channel Hit@3 |
|---|---:|---:|---:|---:|---:|---:|
| Dense AE, v2.2 quantile | 1.000 | 0.811 | 1.667 | 14.67 | 0.721 | 1.000 |
| TCN AE, v2.2 quantile | 1.000 | 0.812 | 1.667 | 11.00 | 0.796 | 0.900 |
| Dense AE, v2.3 constrained | 1.000 | 0.924 | 0.595 | 16.00 | 0.697 | 1.000 |
| TCN AE, v2.3 constrained | 1.000 | 0.864 | 1.190 | 15.17 | 0.747 | 0.867 |
| TCN AE, v2.3 constrained + log1p | 1.000 | 0.864 | 1.190 | 15.17 | 0.751 | 0.867 |

The constrained filter reduced TCN false alarms/day from 1.667 to 1.190 and
increased event F1 from 0.812 to 0.864 without reducing event recall. However,
median delay worsened from 11.00 to 15.17 minutes. Dense AE benefited more:
its false alarms/day fell to 0.595 and event F1 rose to 0.924.

`log1p` slightly improved TCN point F1 but did not change event decisions.
`robust_zscore` is a positive affine transformation fitted on validation; with
quantile thresholds and linear EWMA it is not expected to change decisions,
so it was not promoted as a better operating point.

## Anomaly-Type Findings

The v2.3 TCN filter delayed every anomaly type on average. The largest changes
were slow degradation (`79.67 -> 98.00` minutes), thermal runaway
(`59.00 -> 80.00`) and drift (`23.00 -> 31.00`). This explains why aggregate
delay worsened even though event recall stayed at 1.00.

Across the three v2.3 TCN global runs, unmatched events were most often
associated with nominal-mode eclipse boundaries (four events) or low threshold
margins in payload mode (three events). Remaining unmatched events were marked
`unknown`; the labels remain diagnostic hints.

## Acceptance Criteria

Minimum acceptance requires TCN global mean recall of at least 0.90, no more
false alarms/day than Dense AE global, and lower median delay than Dense AE
global. Ideal acceptance additionally requires no more than 0.50 false
alarms/day, delay no more than 10 minutes, point F1 at least as high as Dense
AE, and Channel Hit@3 of at least 0.90.

The final assessment must reject any apparent false-alarm gain that reduces
recall below the minimum.

Measured acceptance:

- recall at least 0.90: passed (`1.00`);
- no more false alarms/day than Dense AE: failed (`1.190 > 0.595`);
- lower delay than Dense AE: passed (`15.17 < 16.00`);
- ideal false alarms/day no more than 0.50: failed;
- ideal delay no more than 10 minutes: failed;
- point F1 at least Dense AE: passed (`0.747 > 0.697`);
- Channel Hit@3 at least 0.90: failed (`0.867`).

Minimum and ideal acceptance are therefore not satisfied. The main blocker is
not recall loss but the absence of anomalous validation events: the selector
can suppress nominal alarms, but cannot protect event delay or attribution
quality while choosing persistence settings.

## Remaining Limitations

- Nominal-only validation cannot tune an event-recall constraint.
- Synthetic operational modes are simpler than mission operations.
- Filter candidates are intentionally small to keep routine runs tractable.
- Heuristic false-positive reasons require engineering review.
- GNN/GAT remains deferred until temporal selection is validated without this
  validation-label limitation.
