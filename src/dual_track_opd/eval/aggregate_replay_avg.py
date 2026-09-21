"""Aggregate per-repeat replay scorer results into an avg@N summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def aggregate(paths: list[Path], metric: str) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --result-json is required")
    values: list[float] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get(metric), (int, float)):
            raise ValueError(f"{path}: missing numeric metric {metric!r}")
        values.append(float(value[metric]))
    return {
        "metric": metric,
        "aggregation": "mean",
        "repeat_count": len(values),
        "values": values,
        "mean": fmean(values),
        "result_jsons": [str(path) for path in paths],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", action="append", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = aggregate([Path(value).expanduser() for value in args.result_json], args.metric)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
