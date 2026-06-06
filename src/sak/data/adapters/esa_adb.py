"""ESA Anomaly Detection Benchmark adapter placeholder."""

from pathlib import Path

from sak.data.adapters.base import TelemetryDataset


class EsaAdbAdapter:
    """Future adapter for ESA-ADB telemetry and anomaly annotations."""

    def load(self, path: Path) -> TelemetryDataset:
        raise NotImplementedError(
            "ESA-ADB loading is not implemented; dataset-specific schema "
            "normalization is required."
        )
