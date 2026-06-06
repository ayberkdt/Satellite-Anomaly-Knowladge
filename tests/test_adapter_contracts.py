import json
from pathlib import Path

import pytest

from sak.data import (
    AdapterDataNotFoundError,
    EsaAdbAdapter,
    NasaSmapMslAdapter,
    SyntheticTelemetryAdapter,
    TelemetryDataset,
)
from sak.synthetic import SyntheticConfig, generate_synthetic_telemetry


def test_synthetic_adapter_returns_canonical_dataset(tmp_path: Path) -> None:
    generated = generate_synthetic_telemetry(
        SyntheticConfig(periods=12000, missing_fraction=0.0)
    )
    telemetry_path = tmp_path / "telemetry.csv"
    generated.frame.reset_index().to_csv(telemetry_path, index=False)
    (tmp_path / "injection_manifest.json").write_text(
        json.dumps([event.to_dict() for event in generated.events]),
        encoding="utf-8",
    )

    loaded = SyntheticTelemetryAdapter().load(telemetry_path)

    assert isinstance(loaded, TelemetryDataset)
    assert loaded.channel_names
    assert loaded.events


def test_esa_dataset_skeleton_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError):
        EsaAdbAdapter().load(Path("unused"))


def test_nasa_adapter_reports_missing_data_explicitly() -> None:
    with pytest.raises(AdapterDataNotFoundError):
        NasaSmapMslAdapter().load(Path("unused"))
