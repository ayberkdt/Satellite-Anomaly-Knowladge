"""Canonical telemetry dataset adapter contracts."""

from sak.data.adapters.base import TelemetryDataset, TelemetryDatasetAdapter
from sak.data.adapters.esa_adb import EsaAdbAdapter
from sak.data.adapters.nasa_smap_msl import (
    AdapterDataNotFoundError,
    DatasetInspection,
    NasaSmapMslAdapter,
    RealTelemetryEvent,
    UnsupportedDatasetLayoutError,
)
from sak.data.adapters.synthetic import SyntheticTelemetryAdapter

__all__ = [
    "EsaAdbAdapter",
    "AdapterDataNotFoundError",
    "DatasetInspection",
    "NasaSmapMslAdapter",
    "RealTelemetryEvent",
    "SyntheticTelemetryAdapter",
    "TelemetryDataset",
    "TelemetryDatasetAdapter",
    "UnsupportedDatasetLayoutError",
]
