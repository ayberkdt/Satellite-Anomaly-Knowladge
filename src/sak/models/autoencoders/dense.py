"""Small PyTorch dense autoencoder for nonlinear telemetry reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sak.models.base import AnomalyModel


@dataclass(frozen=True)
class DenseAutoencoderConfig:
    """Training and architecture settings for the dense autoencoder."""

    hidden_dim: int = 24
    latent_dim: int = 6
    epochs: int = 35
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    seed: int = 42


class _DenseNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DenseAutoencoderModel(AnomalyModel):
    """Dense reconstruction model with an internal chronological holdout."""

    def __init__(
        self,
        input_dim: int,
        config: DenseAutoencoderConfig | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.config = config or DenseAutoencoderConfig()
        self.device = torch.device("cpu")
        torch.manual_seed(self.config.seed)
        self.network = _DenseNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        self.training_history_: list[dict[str, float]] = []

    def fit(self, values: np.ndarray) -> "DenseAutoencoderModel":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dim:
            raise ValueError(f"values must have shape [n, {self.input_dim}]")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must contain only finite numbers")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        holdout_start = max(1, int(len(matrix) * 0.90))
        train_values = torch.from_numpy(matrix[:holdout_start])
        holdout_values = torch.from_numpy(matrix[holdout_start:]).to(self.device)
        loader_generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(train_values),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.MSELoss()
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            self.network.train()
            total_loss = 0.0
            total_samples = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self.network(batch)
                loss = loss_function(reconstruction, batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch)
                total_samples += len(batch)

            self.network.eval()
            with torch.no_grad():
                holdout_loss = float(
                    loss_function(self.network(holdout_values), holdout_values).item()
                )
            train_loss = total_loss / max(total_samples, 1)
            self.training_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "holdout_loss": holdout_loss,
                }
            )

            if holdout_loss < best_loss - 1e-6:
                best_loss = holdout_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            result = self.network(torch.from_numpy(matrix).to(self.device))
        return result.cpu().numpy()

    def score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(values, dtype=np.float32)
        residual = matrix - self.reconstruct(matrix)
        channel_errors = residual.astype(float) ** 2
        return channel_errors.mean(axis=1), channel_errors

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "config": asdict(self.config),
                "state_dict": self.network.state_dict(),
                "history": self.training_history_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DenseAutoencoderModel":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            input_dim=int(payload["input_dim"]),
            config=DenseAutoencoderConfig(**payload["config"]),
        )
        model.network.load_state_dict(payload["state_dict"])
        model.training_history_ = payload["history"]
        return model
