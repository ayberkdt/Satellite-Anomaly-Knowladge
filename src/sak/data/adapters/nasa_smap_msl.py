"""NASA SMAP/MSL adapter placeholder with a stable public API."""

from pathlib import Path

from sak.data.adapters.base import TelemetryDataset


class NasaSmapMslAdapter:
    """Future adapter for the Telemanom SMAP/MSL dataset layout."""

    def load(self, path: Path) -> TelemetryDataset:
        raise NotImplementedError(
            "NASA SMAP/MSL loading is not implemented; use the adapter contract "
            "to add mission-specific channel and label mapping."
        )
