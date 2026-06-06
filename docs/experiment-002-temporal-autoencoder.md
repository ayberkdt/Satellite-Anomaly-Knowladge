# Experiment 002 - Window-based TCN Autoencoder

## Purpose

SAK-v2.2 introduces a temporal reconstruction baseline before graph models.
The experiment compares timestamp-based PCA and Dense Autoencoder models with
an optional window-based TCN Autoencoder under the same threshold, event,
diagnostic and XAI contracts.

## Model

`TCNAutoencoderModel` accepts arrays shaped
`[num_windows, window_size, num_channels]`. Internally, PyTorch uses
`[num_windows, num_channels, window_size]`. Dilated depthwise temporal
convolutions provide a growing receptive field, while channel projections
create a compact latent representation and reconstruct the original window.

Training uses CPU, MSE reconstruction loss, deterministic seeds, shuffled
training windows and the final chronological 10% of train windows as an early
stopping holdout.

## Windowing Settings

The default experiment uses:

- window size: 60 timestamps;
- stride: 1 timestamp;
- label mode: `any`;
- timestamp aggregation: `mean`;
- hidden channels: 32;
- latent channels: 16;
- kernel size: 5;
- temporal layers: 3.

## Leakage Control

Raw telemetry is first split chronologically. Preprocessing is fitted only on
the train partition. `build_windows` is then called independently for train,
validation and test. No window can contain rows from two partitions, and
threshold calibration still uses nominal validation data only.

## Scoring

The TCN returns squared reconstruction errors shaped
`[num_windows, window_size, num_channels]`. A timestamp may occur in several
overlapping windows. `aggregate_window_errors_to_timestamps` combines those
errors with `mean` or `max`, then averages channel errors into one timestamp
anomaly score. Uncovered timestamps are explicitly warned about and assigned
zero rather than silently producing NaN.

## Temporal XAI

Timestamp-aligned channel errors feed the existing reconstruction attribution,
subsystem mapping and critical-window logic. TCN variants additionally export
a window/channel heatmap and a JSON summary containing window counts,
coverage, score statistics and mean timestamp channel errors.

## First Expected Comparison

| Model variant | Event F1 | False alarms/day | Median delay | Point F1 | Channel Hit@3 | Critical-window hit |
|---|---:|---:|---:|---:|---:|---:|
| pca_global | 0.952 | 0.357 | 13.5 | 0.655 | 0.900 | 0.900 |
| dense_autoencoder_global | 0.952 | 0.357 | 13.5 | 0.714 | 1.000 | 1.000 |
| tcn_autoencoder_global | 0.741 | 2.500 | 9.0 | 0.787 | 0.900 | 1.000 |

TCN is not required to dominate every aggregate metric. The main hypothesis is
that drift, slow degradation, correlation break and thermal runaway events may
show improved delay or critical-window localization.

The first seed supports only part of that hypothesis. TCN reduced drift delay
from 51-53 minutes to 16 minutes and slow-degradation delay from 80-98 minutes
to 74 minutes. It also produced earlier spike and voltage/current-rise alarms.
Correlation-break delay stayed at 2 minutes, while thermal-runaway delay
worsened from 56 to 66 minutes. The current threshold calibration creates too
many TCN false alarms, so multi-seed validation and temporal-score calibration
are required before claiming an operational improvement.

## Three-Seed Check

Global-threshold mean and population standard deviation for seeds 1, 2 and 3:

| Model | Event F1 | False alarms/day | Median delay | Point F1 | Channel Hit@3 | Critical-window hit |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 0.831 +/- 0.018 | 0.952 +/- 0.168 | 25.00 +/- 0.00 | 0.649 +/- 0.004 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Dense AE | 0.811 +/- 0.016 | 1.667 +/- 0.168 | 14.67 +/- 1.31 | 0.721 +/- 0.011 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| TCN AE | 0.812 +/- 0.030 | 1.667 +/- 0.337 | 11.00 +/- 1.47 | 0.796 +/- 0.003 | 0.900 +/- 0.000 | 1.000 +/- 0.000 |

Across these seeds, TCN improves point coverage and median delay relative to
Dense AE, but it does not reduce false alarms and loses some channel Hit@3.
The next experiment should calibrate temporal scores and thresholding before
increasing model complexity.

## Risks

- Highly overlapping windows increase CPU training cost and sample redundancy.
- Large stride values can leave timestamps uncovered.
- Reconstruction quality may improve without improving operational alarm
  quality.
- Temporal smoothing and EWMA can jointly increase detection delay.
- Synthetic temporal gains may not transfer to real mission telemetry.
- GNN/GAT remains deferred until temporal performance is validated across
  seeds and anomaly types.
