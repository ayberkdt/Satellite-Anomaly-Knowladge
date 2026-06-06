"""Inspect external telemetry dataset adapter readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sak.data import AdapterDataNotFoundError, NasaSmapMslAdapter


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
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    adapter = _adapter(arguments.adapter)
    try:
        report = adapter.inspect(arguments.path)  # type: ignore[attr-defined]
    except AdapterDataNotFoundError as error:
        report = {
            "adapter": arguments.adapter,
            "path": str(arguments.path),
            "exists": False,
            "recognized_layout": False,
            "load_supported": False,
            "channel_count": None,
            "labels_available": False,
            "event_count": None,
            "error": str(error),
            "notes": "Dataset path is missing; no telemetry was inspected.",
        }
    if arguments.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Adapter: {report['adapter']}")
    print(f"Path: {report['path']}")
    print(f"Recognized layout: {report['recognized_layout']}")
    print(f"Load supported: {report['load_supported']}")
    print(f"Channel count: {report['channel_count']}")
    print(f"Labels available: {report['labels_available']}")
    print(f"Event count: {report['event_count']}")
    print(f"Notes: {report['notes']}")


if __name__ == "__main__":
    main()
