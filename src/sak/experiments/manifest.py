"""Run manifest generation for reproducible SAK experiments."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sak.experiments.artifacts import write_json


def file_checksum(path: Path) -> str:
    """Return a SHA-256 checksum for a dataset artefact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def optional_git_hash(repository_dir: Path) -> str | None:
    """Return the current Git hash without making manifest generation fragile."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_run_manifest(
    *,
    config_path: Path,
    dataset_path: Path,
    seed: int,
    train_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    models: list[str],
    repository_dir: Path,
) -> dict[str, Any]:
    """Build the stable SAK-v2.3 run metadata payload."""

    created_at = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "run_id": f"sak-v2.3-{created_at:%Y%m%dT%H%M%SZ}-seed-{seed}",
        "created_at": created_at.isoformat(),
        "sak_version": "SAK-v2.3",
        "seed": seed,
        "config_path": str(config_path),
        "dataset_name": "synthetic",
        "dataset_checksum": file_checksum(dataset_path),
        "train_start": train_index[0].isoformat(),
        "train_end": train_index[-1].isoformat(),
        "validation_start": validation_index[0].isoformat(),
        "validation_end": validation_index[-1].isoformat(),
        "test_start": test_index[0].isoformat(),
        "test_end": test_index[-1].isoformat(),
        "models": models,
        "notes": "Temporal score calibration and false-alarm suppression run",
    }
    git_hash = optional_git_hash(repository_dir)
    if git_hash is not None:
        payload["git_hash"] = git_hash
    return payload


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write a run manifest to disk."""

    return write_json(path, manifest)
