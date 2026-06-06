from pathlib import Path

import numpy as np
import pandas as pd

from sak.experiments.dataset_runner import run_real_dataset_experiment


def test_real_dashboard_handles_unknown_critical_region(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "test").mkdir()
    train_step = np.linspace(0.0, 2.0 * np.pi, 90)
    test_step = np.linspace(0.0, 2.0 * np.pi, 40)
    np.save(data_dir / "train" / "P-1.npy", np.column_stack([np.sin(train_step)]))
    test = np.column_stack([np.sin(test_step)])
    test[12:17, 0] += 4.0
    np.save(data_dir / "test" / "P-1.npy", test)
    pd.DataFrame({"chan_id": ["P-1"], "anomaly_sequences": ["[[12, 17]]"]}).to_csv(
        data_dir / "labeled_anomalies.csv",
        index=False,
    )

    run_real_dataset_experiment(
        adapter_name="nasa_smap_msl",
        data_path=data_dir,
        output_dir=output_dir,
        models=("pca",),
        channel_id="P-1",
        render_dashboard=True,
    )

    html = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Critical-region lead-time metrics are unavailable or proxy" in html
    assert "not available / proxy" in html
    assert "Unknown" in html
