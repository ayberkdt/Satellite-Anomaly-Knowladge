from pathlib import Path

import numpy as np

from sak.models.temporal import TCNAutoencoderConfig, TCNAutoencoderModel


def _config(seed: int = 7) -> TCNAutoencoderConfig:
    return TCNAutoencoderConfig(
        window_size=8,
        stride=2,
        input_channels=3,
        hidden_channels=4,
        latent_channels=2,
        kernel_size=3,
        num_layers=1,
        epochs=2,
        batch_size=4,
        patience=2,
        seed=seed,
    )


def _windows() -> np.ndarray:
    return np.random.default_rng(11).normal(size=(16, 8, 3))


def test_tcn_fit_reconstruction_and_score_shapes() -> None:
    windows = _windows()
    model = TCNAutoencoderModel(_config()).fit_windows(windows)

    reconstruction = model.reconstruct_windows(windows)
    scores, channel_errors = model.score_windows(windows)

    assert reconstruction.shape == windows.shape
    assert scores.shape == (len(windows),)
    assert channel_errors.shape == windows.shape
    assert len(model.training_history_) == 2


def test_tcn_save_and_load_preserve_reconstruction(tmp_path: Path) -> None:
    windows = _windows()
    model = TCNAutoencoderModel(_config()).fit_windows(windows)
    checkpoint = tmp_path / "model.pt"

    model.save(checkpoint)
    loaded = TCNAutoencoderModel.load(checkpoint)

    assert checkpoint.exists()
    assert loaded.config == model.config
    assert np.allclose(
        loaded.reconstruct_windows(windows),
        model.reconstruct_windows(windows),
    )


def test_tcn_seed_is_deterministic() -> None:
    windows = _windows()

    first = TCNAutoencoderModel(_config()).fit_windows(windows)
    second = TCNAutoencoderModel(_config()).fit_windows(windows)

    assert np.allclose(
        first.reconstruct_windows(windows),
        second.reconstruct_windows(windows),
    )
