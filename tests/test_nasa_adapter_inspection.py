import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sak.data import (
    AdapterDataNotFoundError,
    NasaSmapMslAdapter,
    UnsupportedDatasetLayoutError,
)


def test_missing_path_raises_adapter_data_not_found(tmp_path: Path) -> None:
    with pytest.raises(AdapterDataNotFoundError, match="does not exist"):
        NasaSmapMslAdapter().inspect(tmp_path / "missing")


def test_empty_path_is_unsupported_with_explanatory_error(tmp_path: Path) -> None:
    adapter = NasaSmapMslAdapter()
    inspection = adapter.inspect(tmp_path)

    assert inspection.supported is False
    assert inspection.errors
    with pytest.raises(UnsupportedDatasetLayoutError, match="Unsupported NASA SMAP/MSL"):
        adapter.load(tmp_path)


def test_source_array_inspection_is_json_serializable(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    np.save(tmp_path / "train" / "P-1.npy", np.ones((12, 1)))
    np.save(tmp_path / "test" / "P-1.npy", np.ones((8, 1)))
    pd.DataFrame({"chan_id": ["P-1"], "anomaly_sequences": ["[[2, 4]]"]}).to_csv(
        tmp_path / "labeled_anomalies.csv",
        index=False,
    )

    inspection = NasaSmapMslAdapter().inspect(tmp_path)

    assert inspection.supported is True
    assert inspection.has_anomaly_intervals is True
    assert json.loads(json.dumps(inspection.to_dict()))["channel_count"] == 1
