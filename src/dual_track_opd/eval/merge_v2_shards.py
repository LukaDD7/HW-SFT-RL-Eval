"""Merge parallel B6-v2 shard runs into one canonical run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


V2_BENCHMARK_ORDER = ("gqa", "dynamath", "viewspatial", "mmmu_pro")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _benchmark_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(record.get("benchmark_id"))
        for record in manifest.get("runs", [])
        if record.get("status") == "completed"
    }


def _validate_shards(
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    benchmark_ids: Sequence[str],
) -> None:
    if len(manifests) != 2:
        raise ValueError("B6-v2 parallel merge expects exactly two shard runs")

    first = manifests[0]
    identity_keys = (
        "suite",
        "backend",
        "checkpoint_path",
        "thinking_protocol",
        "generation_budgets",
        "repeat_count",
        "sampling_temperature",
        "protocol",
    )
    for key in identity_keys:
        if first.get(key) != manifests[1].get(key):
            raise ValueError(f"shard protocol mismatch for {key}; use new shard run names")
    if any(manifest.get("returncode") != 0 for manifest in manifests):
        raise ValueError("cannot merge a shard with a nonzero returncode")

    expected = set(benchmark_ids)
    if not expected or len(expected) != len(benchmark_ids) or not expected <= set(config["benchmarks"]):
        raise ValueError(f"invalid v2 merge benchmark selection: {list(benchmark_ids)}")
    ids_a = _benchmark_ids(first)
    ids_b = _benchmark_ids(manifests[1])
    if ids_a & ids_b:
        raise ValueError(f"overlapping shard benchmarks: {sorted(ids_a & ids_b)}")
    if (ids_a | ids_b) != expected:
        raise ValueError(
            "shard benchmark set mismatch: "
            f"expected={sorted(expected)}, shard_a={sorted(ids_a)}, shard_b={sorted(ids_b)}"
        )


def _link_benchmark(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            destination.unlink()
        elif destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            raise FileExistsError(
                f"refusing to replace non-empty benchmark output: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def merge_shards(
    output_run: Path,
    shard_runs: list[Path],
    *,
    config_path: Path,
    benchmark_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a canonical run from completed v2 shard directories."""
    output_run = output_run.expanduser().resolve()
    shard_runs = [path.expanduser().resolve() for path in shard_runs]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("benchmarks"), dict):
        raise ValueError(f"invalid v2 config: {config_path}")

    ordered_benchmarks = (
        tuple(benchmark_ids) if benchmark_ids is not None else V2_BENCHMARK_ORDER
    )
    manifests = []
    for shard_run in shard_runs:
        manifest_path = shard_run / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"shard manifest not found: {manifest_path}")
        manifests.append(_load_json(manifest_path))
    _validate_shards(config, manifests, ordered_benchmarks)

    output_run.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_run / "run_manifest.json"
    existing_manifest = (
        _load_json(existing_manifest_path) if existing_manifest_path.is_file() else None
    )
    if existing_manifest is not None and not existing_manifest.get("v2_parallel_shards"):
        raise FileExistsError(
            f"refusing to overwrite a non-parallel run: {existing_manifest_path}"
        )

    if existing_manifest is not None:
        existing_ids = existing_manifest.get("requested_benchmarks")
        if existing_ids is None:
            existing_ids = sorted(_benchmark_ids(existing_manifest))
        if set(existing_ids) != set(ordered_benchmarks) or len(existing_ids) != len(ordered_benchmarks):
            raise ValueError(
                "merged benchmark set mismatch: "
                f"expected={list(ordered_benchmarks)}, prior={existing_ids}. Use a new run name."
            )

    benchmark_to_shard = {
        benchmark_id: shard_runs[index]
        for index, manifest in enumerate(manifests)
        for benchmark_id in _benchmark_ids(manifest)
    }
    for benchmark_id in ordered_benchmarks:
        shard_run = benchmark_to_shard[benchmark_id]
        _link_benchmark(
            shard_run / "lmms" / benchmark_id,
            output_run / "lmms" / benchmark_id,
        )

    records_by_id = {
        str(record.get("benchmark_id")): record
        for manifest in manifests
        for record in manifest.get("runs", [])
        if record.get("status") == "completed"
    }
    merged_manifest = dict(manifests[0])
    merged_manifest["requested_benchmarks"] = list(ordered_benchmarks)
    merged_manifest["runs"] = [records_by_id[key] for key in ordered_benchmarks]
    merged_manifest["run_name"] = output_run.name
    merged_manifest["raw_output_path"] = str(output_run)
    merged_manifest["v2_parallel_shards"] = [str(path) for path in shard_runs]
    merged_manifest["returncode"] = max(
        int(manifest.get("returncode", 0)) for manifest in manifests
    )
    merged_manifest["notes"] = (
        "Merged from two parallel v2 shard runs; benchmark outputs are symlinks to shard raw files."
    )
    _save_json(existing_manifest_path, merged_manifest)

    prompt_checks = []
    for shard_run in shard_runs:
        path = shard_run / "server_prompt_check.json"
        if path.is_file():
            prompt_checks.append(_load_json(path))
    if prompt_checks:
        prompt_check = dict(prompt_checks[0])
        prompt_check["passed"] = all(check.get("passed") for check in prompt_checks)
        prompt_check["shards"] = prompt_checks
        _save_json(output_run / "server_prompt_check.json", prompt_check)

    return merged_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--shard-run", required=True, action="append", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated expected benchmark IDs; defaults to the historical four-task v2 set.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    merge_shards(
        args.output_run,
        args.shard_run,
        config_path=args.config,
        benchmark_ids=(
            [part for part in args.benchmarks.split(",") if part]
            if args.benchmarks is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
