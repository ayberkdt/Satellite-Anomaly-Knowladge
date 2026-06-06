"""Common anomaly model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyModel(ABC):
    """Interface implemented by PCA, autoencoder, temporal and graph models."""

    @abstractmethod
    def fit(self, values: np.ndarray) -> "AnomalyModel":
        """Fit only on the intended training partition."""

    @abstractmethod
    def score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return aggregate and optional channel-level anomaly errors."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state and preprocessing metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "AnomalyModel":
        """Restore model state."""

