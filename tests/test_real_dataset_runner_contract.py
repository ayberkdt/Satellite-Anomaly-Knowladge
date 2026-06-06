from pathlib import Path

import numpy as np
import pandas as pd

from sak.experiments.dataset_runner import run_real_dataset_experiment


def _write_source_dataset(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    train_step = np.linspace(0.0, 4.0 * np.pi, 120)
    test_step = np.linspace(0.0, 2.0 * np.pi, 60)
    train = np.column_stack([np.sin(train_step), np.cos(train_step)])
    test = np.column_stack([np.sin(test_step), np.cos(test_step)])
    test[20:26, 0] += 5.0
    np.save(root / "train" / "P-1.npy", train)
    np.save(root / "test" / "P-1.npy", test)
    pd.DataFrame({"chan_id": ["P-1"], "anomaly_sequences": ["[[20, 26]]"]}).to_csv(
        root / "labeled_anomalies.csv",
        index=False,
    )


def test_real_runner_pca_writes_contract_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"
    _write_source_dataset(data_dir)

    summary = run_real_dataset_experiment(
        adapter_name="nasa_smap_msl",
        data_path=data_dir,
        output_dir=output_dir,
        models=("pca",),
        channel_id="P-1",
        calibration="constrained_event_f1",
    )

    assert "pca_global" in summary
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "data_quality_report.json").exists()
    assert (output_dir / "split_manifest.json").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert summary["pca_global"]["calibration"]["test_partition_used_for_selection"] is False
    assert summary["pca_global"]["calibration"]["selection_reason"] == "no_calibration_events"
