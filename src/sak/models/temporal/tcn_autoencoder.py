"""Deterministic CPU temporal convolutional autoencoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TCNAutoencoderConfig:
    """Architecture, windowing and optimization settings for the TCN model."""

    window_size: int = 60
    stride: int = 1
    input_channels: int = 13
    hidden_channels: int = 32
    latent_channels: int = 16
    kernel_size: int = 5
    num_layers: int = 3
    dropout: float = 0.0
    epochs: int = 35
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    seed: int = 42

    def __post_init__(self) -> None:
        positive_fields = {
            "window_size": self.window_size,
            "stride": self.stride,
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve window length")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")


class _TemporalResidualBlock(nn.Module):
    """Efficient dilated depthwise temporal residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class _TCNAutoencoderNetwork(nn.Module):
    def __init__(self, config: TCNAutoencoderConfig) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(
            config.input_channels,
            config.hidden_channels,
            kernel_size=1,
        )
        self.temporal_blocks = nn.Sequential(
            *(
                _TemporalResidualBlock(
                    channels=config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=2**layer,
                    dropout=config.dropout,
                )
                for layer in range(config.num_layers)
            )
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(
                config.hidden_channels,
                config.latent_channels,
                kernel_size=1,
            ),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(
                config.latent_channels,
                config.hidden_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_channels,
                config.input_channels,
                kernel_size=1,
            ),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values)
        temporal = self.temporal_blocks(hidden)
        latent = self.encoder(temporal)
        return self.decoder(latent)


class TCNAutoencoderModel:
    """Window-based reconstruction model using dilated temporal convolutions."""

    def __init__(self, config: TCNAutoencoderConfig | None = None) -> None:
        self.config = config or TCNAutoencoderConfig()
        self.device = torch.device("cpu")
        self._seed_all()
        self.network = _TCNAutoencoderNetwork(self.config).to(self.device)
        self.training_history_: list[dict[str, float]] = []

    def _seed_all(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.use_deterministic_algorithms(True)

    def _validate_windows(
        self,
        windows: np.ndarray,
        *,
        minimum_windows: int = 0,
    ) -> np.ndarray:
        matrix = np.asarray(windows, dtype=np.float32)
        expected_tail = (self.config.window_size, self.config.input_channels)
        if matrix.ndim != 3 or matrix.shape[1:] != expected_tail:
            raise ValueError(
                "windows must have shape "
                f"[n, {self.config.window_size}, {self.config.input_channels}]"
            )
        if len(matrix) < minimum_windows:
            raise ValueError(f"at least {minimum_windows} windows are required")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("windows must contain only finite values")
        return matrix

    @staticmethod
    def _to_torch(matrix: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(matrix).permute(0, 2, 1).contiguous()

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.permute(0, 2, 1).contiguous().cpu().numpy()

    def fit_windows(self, windows: np.ndarray) -> "TCNAutoencoderModel":
        """Fit on windows with the final chronological 10% as a holdout."""

        matrix = self._validate_windows(windows, minimum_windows=2)
        self._seed_all()
        holdout_start = max(1, int(len(matrix) * 0.90))
        train_values = self._to_torch(matrix[:holdout_start])
        holdout_values = self._to_torch(matrix[holdout_start:]).to(self.device)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(train_values),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
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
        self.training_history_ = []

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
                    loss_function(
                        self.network(holdout_values),
                        holdout_values,
                    ).item()
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

    def reconstruct_windows(self, windows: np.ndarray) -> np.ndarray:
        """Reconstruct windows in their public [window, time, channel] layout."""

        matrix = self._validate_windows(windows)
        if len(matrix) == 0:
            return matrix.copy()
        tensor = self._to_torch(matrix).to(self.device)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        reconstructions: list[torch.Tensor] = []
        self.network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                reconstructions.append(self.network(batch))
        return self._to_numpy(torch.cat(reconstructions, dim=0))

    def score_windows(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean window scores and position/channel squared errors."""

        matrix = self._validate_windows(windows)
        reconstruction = self.reconstruct_windows(matrix)
        channel_errors = (matrix.astype(float) - reconstruction.astype(float)) ** 2
        scores = channel_errors.mean(axis=(1, 2))
        return scores, channel_errors

    def save(self, path: Path) -> None:
        """Save architecture, weights and training history."""

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.network.state_dict(),
                "history": self.training_history_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TCNAutoencoderModel":
        """Load a CPU TCN checkpoint."""

        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(config=TCNAutoencoderConfig(**payload["config"]))
        model.network.load_state_dict(payload["state_dict"])
        model.training_history_ = payload["history"]
        return model
