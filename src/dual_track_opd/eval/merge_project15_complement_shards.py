"""Merge four Project15-complement output shards into one canonical run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT15_COMPLEMENT_ORDER = (
    "vqav2",
    "mindcube",
    "scienceqa",
    "mmsi_bench",
    "blink",
    "mathverse",
    "mathvista",
    "mmvet",
    "mv_math",
)


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


def _validate_shards(config: dict[str, Any], manifests: list[dict[str, Any]]) -> None:
    if len(manifests) != 4:
        raise ValueError("Project15-complement 4-GPU merge expects exactly four shard runs")

    first = manifests[0]
    identity_keys = (
        "suite",
        "backend",
        "checkpoint_path",
        "thinking_protocol",
        "repeat_count",
        "sampling_temperature",
        "protocol",
    )
    for key in identity_keys:
        if any(first.get(key) != manifest.get(key) for manifest in manifests[1:]):
            raise ValueError(f"shard protocol mismatch for {key}; use new shard run names")
    if any(manifest.get("returncode") != 0 for manifest in manifests):
        raise ValueError("cannot merge a shard with a nonzero returncode")

    expected = {
        benchmark_id
        for benchmark_id, spec in config["benchmarks"].items()
        if benchmark_id in PROJECT15_COMPLEMENT_ORDER
    }
    ids = [_benchmark_ids(manifest) for manifest in manifests]
    overlap = set()
    for index in range(len(ids)):
        for other in range(index + 1, len(ids)):
            overlap |= ids[index] & ids[other]
    if overlap:
        raise ValueError(f"overlapping shard benchmarks: {sorted(overlap)}")
    union = set.union(*ids)
    if union != expected:
        raise ValueError(
            "shard benchmark set mismatch: "
            f"expected={sorted(expected)}, "
            f"shard_a={sorted(ids[0])}, "
            f"shard_b={sorted(ids[1])}, "
            f"shard_c={sorted(ids[2])}, "
            f"shard_d={sorted(ids[3])}"
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
) -> dict[str, Any]:
    """Create a canonical Project15-complement run from four shard directories."""
    output_run = output_run.expanduser().resolve()
    shard_runs = [path.expanduser().resolve() for path in shard_runs]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("benchmarks"), dict):
        raise ValueError(f"invalid Project15 config: {config_path}")

    manifests = []
    for shard_run in shard_runs:
        manifest_path = shard_run / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"shard manifest not found: {manifest_path}")
        manifests.append(_load_json(manifest_path))
    _validate_shards(config, manifests)

    output_run.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_run / "run_manifest.json"
    existing_manifest = (
        _load_json(existing_manifest_path) if existing_manifest_path.is_file() else None
    )
    if existing_manifest is not None and not existing_manifest.get("project15_complement_4gpu_shards"):
        raise FileExistsError(
            f"refusing to overwrite a non-parallel run: {existing_manifest_path}"
        )

    benchmark_to_shard = {
        benchmark_id: shard_runs[index]
        for index, manifest in enumerate(manifests)
        for benchmark_id in _benchmark_ids(manifest)
    }
    for benchmark_id in PROJECT15_COMPLEMENT_ORDER:
        shard_run = benchmark_to_shard[benchmark_id]
        _link_benchmark(
            shard_run / "lmms" / benchmark_id,
            output_run / "lmms" / benchmark_id,
        )

    # MV-MATH is the only replay benchmark in the complement set.
    mv_math_shard = benchmark_to_shard["mv_math"]
    _link_benchmark(
        mv_math_shard / "replay",
        output_run / "replay",
    )

    records_by_id = {
        str(record.get("benchmark_id")): record
        for manifest in manifests
        for record in manifest.get("runs", [])
        if record.get("status") == "completed"
    }
    merged_manifest = dict(manifests[0])
    merged_manifest["runs"] = [records_by_id[key] for key in PROJECT15_COMPLEMENT_ORDER]
    merged_manifest["run_name"] = output_run.name
    merged_manifest["raw_output_path"] = str(output_run)
    merged_manifest["project15_complement_4gpu_shards"] = [str(path) for path in shard_runs]
    merged_manifest["returncode"] = max(
        int(manifest.get("returncode", 0)) for manifest in manifests
    )
    merged_manifest["notes"] = (
        "Merged from three parallel Project15-complement shard runs; "
        "benchmark outputs are symlinks to shard raw files."
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    merge_shards(
        args.output_run,
        args.shard_run,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
