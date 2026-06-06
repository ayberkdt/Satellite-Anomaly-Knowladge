"""Canonical telemetry dataset adapter contracts."""

from sak.data.adapters.base import TelemetryDataset, TelemetryDatasetAdapter
from sak.data.adapters.esa_adb import EsaAdbAdapter
from sak.data.adapters.nasa_smap_msl import AdapterDataNotFoundError, NasaSmapMslAdapter
from sak.data.adapters.synthetic import SyntheticTelemetryAdapter

__all__ = [
    "EsaAdbAdapter",
    "AdapterDataNotFoundError",
    "NasaSmapMslAdapter",
    "SyntheticTelemetryAdapter",
    "TelemetryDataset",
    "TelemetryDatasetAdapter",
]
