"""Collect primary metrics from one project benchmark run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .benchmark_suite import BenchmarkSpec, load_suite
from .score_open import extract_normalized_answer, normalize_answer


def _latest_result_json(directory: Path) -> Path | None:
    candidates: list[Path] = []
    for path in directory.rglob("*.json") if directory.is_dir() else ():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and ("results" in value or "groups" in value):
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _metric_values(value: Any, primary_metric: str) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                key == primary_metric
                or key.startswith(primary_metric + ",")
                or key.endswith("/" + primary_metric)
            ) and isinstance(nested, (int, float)):
                found.append(float(nested))
            found.extend(_metric_values(nested, primary_metric))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_metric_values(nested, primary_metric))
    return found


def _summarize_lmms(run_dir: Path, spec: BenchmarkSpec) -> tuple[float | None, str]:
    result_path = _latest_result_json(run_dir / "lmms" / spec.benchmark_id)
    if result_path is None or spec.primary_metric is None:
        return None, "result_missing"
    value = json.loads(result_path.read_text(encoding="utf-8"))
    # Group summaries are already aggregated over their child tasks. Prefer them
    # when present so suites such as BLINK and MathVista are not counted twice.
    values = _metric_values(value.get("groups", {}), spec.primary_metric)
    if not values:
        values = _metric_values(value.get("results", {}), spec.primary_metric)
    if not values:
        return None, f"metric_missing:{spec.primary_metric}"
    return fmean(values), str(result_path)


def _summarize_replay(run_dir: Path, spec: BenchmarkSpec) -> tuple[float | None, str]:
    path = run_dir / "replay" / f"{spec.benchmark_id}.jsonl"
    if not path.is_file():
        return None, "result_missing"
    if spec.primary_metric != "normalized_exact":
        return None, "official_evaluator_pending"
    correct = total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("error"):
                continue
            prediction = extract_normalized_answer(str(item.get("prediction", "")))
            truth_value = item.get("ground_truth")
            truths = truth_value if isinstance(truth_value, list) else [truth_value]
            normalized_truths = {
                normalize_answer(str(truth)) for truth in truths if truth not in (None, "")
            }
            if prediction is None:
                full_prediction = normalize_answer(str(item.get("prediction", "")))
                if full_prediction in normalized_truths:
                    prediction = full_prediction
            if prediction and normalized_truths:
                total += 1
                correct += int(prediction is not None and prediction in normalized_truths)
    return (correct / total if total else None), str(path)


def summarize_run(run_dir: str | Path, config: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    suite = load_suite(config)
    manifest_path = run_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {
        record["benchmark_id"]: record for record in manifest.get("runs", [])
    }
    rows: list[dict[str, Any]] = []
    for benchmark_id, record in selected.items():
        spec = suite.benchmarks[benchmark_id]
        if record.get("status") == "deferred":
            metric, source = None, record.get("reason", "deferred")
        elif spec.runner == "lmms_eval":
            metric, source = _summarize_lmms(run_path, spec)
        else:
            metric, source = _summarize_replay(run_path, spec)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "contract_name": spec.contract_name,
                "category": spec.category,
                "metric_tier": spec.metric_tier,
                "scoring": spec.scoring,
                "primary_metric": spec.primary_metric,
                "value": metric,
                "source": source,
            }
        )

    category_macro: dict[str, float] = {}
    for category in sorted({row["category"] for row in rows}):
        values = [
            float(row["value"])
            for row in rows
            if row["category"] == category
            and row["value"] is not None
            and row["metric_tier"] != "internal_diagnostic"
        ]
        if values:
            category_macro[category] = fmean(values)
    return {
        "run_dir": str(run_path),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "dataset_manifest_hash": manifest.get("dataset_manifest_hash"),
        "rows": rows,
        "category_macro_excluding_internal_diagnostics": category_macro,
    }


def write_summary(summary: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = (
        "benchmark_id",
        "contract_name",
        "category",
        "metric_tier",
        "scoring",
        "primary_metric",
        "value",
        "source",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary["rows"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="configs/eval/project_vision_opd.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = summarize_run(run_dir, args.config)
    output_json = run_dir / "summary.json"
    output_csv = run_dir / "summary.csv"
    write_summary(summary, output_json, output_csv)
    print(json.dumps({"summary_json": str(output_json), "summary_csv": str(output_csv)}))


if __name__ == "__main__":
    main()
