"""Filesystem helpers for deterministic experiment artefacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def model_variant_name(model_name: str, threshold_strategy: str) -> str:
    """Return the canonical artefact and comparison key for one model variant."""

    normalized_model = model_name.strip().lower()
    normalized_strategy = threshold_strategy.strip().lower()
    if not normalized_model or not normalized_strategy:
        raise ValueError("model_name and threshold_strategy must be non-empty")
    return f"{normalized_model}_{normalized_strategy}"


@dataclass(frozen=True)
class VariantArtifactPaths:
    """Canonical paths owned by one model and threshold strategy."""

    root: Path
    reports: Path
    xai: Path
    plots: Path
    diagnostics: Path


def create_variant_artifact_paths(
    output_dir: Path,
    model_variant: str,
) -> VariantArtifactPaths:
    """Create and return the directory tree for a model variant."""

    root = output_dir / model_variant
    paths = VariantArtifactPaths(
        root=root,
        reports=root / "reports",
        xai=root / "xai",
        plots=root / "plots",
        diagnostics=root / "diagnostics",
    )
    for path in (
        paths.root,
        paths.reports,
        paths.xai,
        paths.plots,
        paths.diagnostics,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def json_default(value: object) -> object:
    """Convert common scientific Python values into JSON-compatible values."""

    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> Path:
    """Write a stable, human-readable JSON artefact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default, allow_nan=False),
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write mapping rows to CSV while preserving the first row's column order."""

    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    return path
