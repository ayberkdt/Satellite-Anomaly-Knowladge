"""NumPy PCA reconstruction baseline with channel-level errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sak.models.base import AnomalyModel


class PCAAnomalyModel(AnomalyModel):
    """Principal-component reconstruction anomaly detector."""

    def __init__(self, explained_variance: float = 0.95) -> None:
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError("explained_variance must be in (0, 1]")
        self.explained_variance = explained_variance
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "PCAAnomalyModel":
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("values must be a two-dimensional matrix with at least two rows")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must contain only finite numbers")

        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        variances = singular_values**2
        ratio = variances / variances.sum()
        cumulative = np.cumsum(ratio)
        component_count = int(np.searchsorted(cumulative, self.explained_variance) + 1)
        component_count = min(component_count, matrix.shape[1])
        self.components_ = right_vectors[:component_count]
        self.n_components_ = component_count
        self.explained_variance_ratio_ = ratio[:component_count]
        return self

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("model must be fitted before reconstruction")
        matrix = np.asarray(values, dtype=float)
        centered = matrix - self.mean_
        latent = centered @ self.components_.T
        return latent @ self.components_ + self.mean_

    def score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(values, dtype=float)
        residual = matrix - self.reconstruct(matrix)
        channel_errors = residual**2
        return channel_errors.mean(axis=1), channel_errors

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            explained_variance=np.array(self.explained_variance),
            mean=self.mean_,
            components=self.components_,
        )

    @classmethod
    def load(cls, path: Path) -> "PCAAnomalyModel":
        payload = np.load(path)
        model = cls(float(payload["explained_variance"]))
        model.mean_ = payload["mean"]
        model.components_ = payload["components"]
        model.n_components_ = model.components_.shape[0]
        return model

