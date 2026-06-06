"""Inspect external telemetry dataset adapter readiness as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sak.data import AdapterDataNotFoundError, NasaSmapMslAdapter  # noqa: E402
from sak.experiments.artifacts import write_json  # noqa: E402


def _adapter(name: str) -> object:
    if name == "nasa_smap_msl":
        return NasaSmapMslAdapter()
    raise ValueError(f"unsupported adapter: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        choices=["nasa_smap_msl"],
        required=True,
        help="Dataset adapter to inspect.",
    )
    parser.add_argument("--path", type=Path, required=True, help="Dataset root path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for adapter_inspection.json.",
    )
    return parser.parse_args()


def _missing_payload(adapter_name: str, path: Path, error: Exception) -> dict[str, Any]:
    return {
        "source": adapter_name,
        "adapter": adapter_name,
        "path": str(path),
        "exists": False,
        "supported": False,
        "recognized_layout": False,
        "load_supported": False,
        "detected_layout": "missing",
        "channel_count": 0,
        "has_train_data": False,
        "has_test_data": False,
        "has_labels": False,
        "has_anomaly_intervals": False,
        "warnings": [],
        "errors": [str(error)],
        "notes": "Dataset path is missing; no telemetry was inspected.",
    }


def main() -> None:
    arguments = parse_args()
    adapter = _adapter(arguments.adapter)
    try:
        inspection = adapter.inspect(arguments.path)  # type: ignore[attr-defined]
        payload = inspection.to_dict()
    except AdapterDataNotFoundError as error:
        payload = _missing_payload(arguments.adapter, arguments.path, error)
    print(json.dumps(payload, indent=2))
    if arguments.output is not None:
        write_json(arguments.output, payload)


if __name__ == "__main__":
    main()
