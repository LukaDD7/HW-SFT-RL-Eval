"""Summarize raw JSONL evaluation outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_jsonl(path: str | Path) -> dict[str, Any]:
    """Count rows, errors, finish reasons, and datasets in a JSONL file."""

    path = Path(path)
    finish_reasons: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    rows = 0
    errors = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            if item.get("error"):
                errors += 1
            finish_reasons[str(item.get("finish_reason", "unknown"))] += 1
            datasets[str(item.get("dataset", "unknown"))] += 1

    return {
        "path": str(path),
        "rows": rows,
        "errors": errors,
        "finish_reason": dict(finish_reasons),
        "datasets": dict(datasets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", help="Path to raw JSONL evaluation output.")
    args = parser.parse_args()
    print(json.dumps(summarize_jsonl(args.jsonl_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
