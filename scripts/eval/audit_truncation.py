#!/usr/bin/env python3
"""Audit lmms-eval sample files for responses at the generation cap.

The tool is read-only. It accepts sample JSONL files and reports:
  - number of samples
  - output-token mean/p50/p95/max
  - number and percentage at the requested cap

Usage (one cap for every file):
  python scripts/eval/audit_truncation.py --cap 4096 path/to/*samples*.jsonl

Usage (one cap per file, options before all paths):
  python scripts/eval/audit_truncation.py \
    --cap 128 --cap 256 path/to/*samples_gqa.jsonl path/to/*samples_viewspatial.jsonl

The output includes one row per file and a total row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _token_count(row: dict[str, Any]) -> int:
    counts = row.get("token_counts") or []
    if not counts or not isinstance(counts, list):
        return 0
    first = counts[0] or {}
    return int(first.get("output_tokens", 0) or 0)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.floor(percentile * len(ordered))))
    return ordered[index]


def audit_file(path: Path, cap: int) -> dict[str, Any]:
    tokens: list[int] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens.append(_token_count(row))

    at_cap = sum(value >= cap for value in tokens)
    return {
        "file": str(path),
        "cap": cap,
        "samples": len(tokens),
        "output_tokens_mean": round(sum(tokens) / len(tokens), 2) if tokens else 0,
        "output_tokens_p50": _percentile(tokens, 0.50),
        "output_tokens_p95": _percentile(tokens, 0.95),
        "output_tokens_max": max(tokens) if tokens else 0,
        "at_cap": at_cap,
        "at_cap_percent": round(100.0 * at_cap / len(tokens), 2) if tokens else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more lmms-eval samples_*.jsonl files",
    )
    parser.add_argument(
        "--cap",
        action="append",
        required=True,
        type=int,
        help="Generation cap. May be repeated; applied in order to the paths",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per line instead of a Markdown table",
    )
    return parser.parse_args()


def _iter_specs(paths: Iterable[Path], caps: list[int]):
    paths = list(paths)
    if len(caps) == 1:
        for path in paths:
            yield path, caps[0]
        return
    if len(caps) != len(paths):
        raise SystemExit(
            "Either provide one --cap for all files, or one --cap per input file "
            f"(got {len(caps)} caps and {len(paths)} files)"
        )
    yield from zip(paths, caps, strict=True)


def main() -> None:
    args = parse_args()
    rows = [audit_file(path, cap) for path, cap in _iter_specs(args.paths, args.cap)]

    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return

    header = (
        "file",
        "cap",
        "samples",
        "mean",
        "p50",
        "p95",
        "max",
        "at_cap",
        "at_cap_percent",
    )
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                str(value)
                for value in (
                    row["file"],
                    row["cap"],
                    row["samples"],
                    row["output_tokens_mean"],
                    row["output_tokens_p50"],
                    row["output_tokens_p95"],
                    row["output_tokens_max"],
                    row["at_cap"],
                    row["at_cap_percent"],
                )
            )
            + " |"
        )


if __name__ == "__main__":
    main()
