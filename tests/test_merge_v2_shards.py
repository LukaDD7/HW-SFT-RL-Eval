from pathlib import Path
import json

import pytest

from dual_track_opd.eval.merge_v2_shards import merge_shards


CONFIG = Path(__file__).resolve().parents[1] / "configs/eval/project_vision_opd_v2.yaml"


def make_shard(root: Path, name: str, benchmarks: list[str], returncode: int = 0) -> Path:
    run = root / name
    run.mkdir(parents=True)
    records = [
        {"benchmark_id": benchmark, "status": "completed", "repeat_count": 4}
        for benchmark in benchmarks
    ]
    manifest = {
        "suite": {"name": "test"},
        "backend": {"name": "test"},
        "checkpoint_path": "/checkpoint",
        "thinking_protocol": {"mode": "auto"},
        "generation_budgets": {
            "gqa": 8192,
            "dynamath": 16384,
            "viewspatial": 8192,
            "mmmu_pro": 16384,
        },
        "repeat_count": 4,
        "sampling_temperature": 1.0,
        "protocol": "avg@4",
        "runs": records,
        "returncode": returncode,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for benchmark in benchmarks:
        directory = run / "lmms" / benchmark
        directory.mkdir(parents=True)
        (directory / "result.txt").write_text(benchmark, encoding="utf-8")
    return run


def test_merges_two_shards_with_symlinks_and_canonical_order(tmp_path: Path) -> None:
    shard_a = make_shard(tmp_path, "a", ["gqa", "dynamath"])
    shard_b = make_shard(tmp_path, "b", ["viewspatial", "mmmu_pro"])
    output = tmp_path / "merged"

    merge_shards(output, [shard_a, shard_b], config_path=CONFIG)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert [record["benchmark_id"] for record in manifest["runs"]] == [
        "gqa",
        "dynamath",
        "viewspatial",
        "mmmu_pro",
    ]
    assert manifest["v2_parallel_shards"] == [str(shard_a), str(shard_b)]
    assert (output / "lmms" / "gqa").is_symlink()
    assert (output / "lmms" / "gqa" / "result.txt").read_text(encoding="utf-8") == "gqa"


def test_rejects_failed_or_overlapping_shards(tmp_path: Path) -> None:
    shard_a = make_shard(tmp_path, "a", ["gqa", "dynamath"], returncode=1)
    shard_b = make_shard(tmp_path, "b", ["gqa", "viewspatial", "mmmu_pro"])
    with pytest.raises(ValueError, match="nonzero returncode"):
        merge_shards(tmp_path / "merged", [shard_a, shard_b], config_path=CONFIG)

    shard_a = make_shard(tmp_path, "c", ["gqa", "dynamath"])
    shard_b = make_shard(tmp_path, "d", ["gqa", "viewspatial", "mmmu_pro"])
    with pytest.raises(ValueError, match="overlapping shard benchmarks"):
        merge_shards(tmp_path / "merged2", [shard_a, shard_b], config_path=CONFIG)


def test_refuses_to_replace_non_parallel_run(tmp_path: Path) -> None:
    shard_a = make_shard(tmp_path, "a", ["gqa", "dynamath"])
    shard_b = make_shard(tmp_path, "b", ["viewspatial", "mmmu_pro"])
    output = tmp_path / "merged"
    output.mkdir()
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-parallel run"):
        merge_shards(output, [shard_a, shard_b], config_path=CONFIG)
