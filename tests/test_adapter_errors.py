from pathlib import Path

import pytest

from sak.data import NasaSmapMslAdapter, UnsupportedDatasetLayoutError


def test_unsupported_layout_error_is_explanatory(tmp_path: Path) -> None:
    (tmp_path / "random.txt").write_text("not telemetry", encoding="utf-8")

    with pytest.raises(UnsupportedDatasetLayoutError) as error:
        NasaSmapMslAdapter().load(tmp_path)

    message = str(error.value)
    assert "Unsupported NASA SMAP/MSL dataset layout" in message
    assert "telemetry.csv/parquet" in message
