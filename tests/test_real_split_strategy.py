from pathlib import Path

import numpy as np
import pandas as pd

from sak.data import NasaSmapMslAdapter
from sak.experiments.real_split import split_real_dataset


def _write_source_dataset(root: Path) -> None:
    (root / "train").mkdir()
    (root / "test").mkdir()
    step_train = np.arange(100, dtype=float)
    step_test = np.arange(40, dtype=float)
    np.save(root / "train" / "P-1.npy", np.column_stack([step_train, step_train**0.5]))
    np.save(root / "test" / "P-1.npy", np.column_stack([step_test, step_test**0.5]))
    pd.DataFrame(
        {
            "chan_id": ["P-1"],
            "spacecraft": ["SMAP"],
            "anomaly_sequences": ["[[10, 15]]"],
            "class": ["contextual"],
        }
    ).to_csv(root / "labeled_anomalies.csv", index=False)


def test_source_train_test_keeps_test_out_of_selection(tmp_path: Path) -> None:
    _write_source_dataset(tmp_path)
    dataset = NasaSmapMslAdapter().load(tmp_path, channel_id="P-1")

    split = split_real_dataset(
        dataset,
        strategy="source_train_test",
        calibration_fraction_from_train=0.20,
        validation_fraction_from_train=0.10,
    )

    assert split.manifest["test_used_for_selection"] is False
    assert len(split.frames.train) == 70
    assert len(split.frames.calibration) == 20
    assert len(split.frames.validation) == 10
    assert len(split.frames.test) == 40
    assert not split.frames.train["is_anomaly"].any()
    assert len(split.partition_events("test")) == 1
    assert len(split.partition_events("calibration")) == 0
