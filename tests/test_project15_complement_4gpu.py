import json
import subprocess
from pathlib import Path

import yaml

from dual_track_opd.eval.merge_project15_complement_shards import merge_shards


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/project_vision_opd.yaml"
SCRIPT = ROOT / "scripts/eval/run_project15_complement_4gpu.sh"


def _make_manifest(path: Path, *, benchmarks: list[str], run_name: str) -> None:
    manifest = {
        "suite": {"name": "project_multimodal_understanding_vision_opd"},
        "backend": {"name": "lmms-eval"},
        "checkpoint_path": "/checkpoint",
        "thinking_protocol": {"mode": "think"},
        "generation_budgets": {benchmark: 1024 for benchmark in benchmarks},
        "repeat_count": 1,
        "sampling_temperature": 1.0,
        "protocol": "single-generation",
        "run_name": run_name,
        "raw_output_path": str(path),
        "returncode": 0,
        "runs": [
            {"benchmark_id": benchmark, "status": "completed"}
            for benchmark in benchmarks
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_4gpu_script_syntax_and_shard_partition() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'SFT_RL_BENCHMARKS="vqav2,scienceqa"' in source
    assert 'SFT_RL_BENCHMARKS="mindcube,mmsi_bench,blink"' in source
    assert 'SFT_RL_BENCHMARKS="mv_math"' in source
    assert 'SFT_RL_JUDGE_BENCHMARKS="mathverse,mathvista,mmvet"' in source
    assert '_nojudge' in source
    assert '_judged' in source
    assert "RUN_SUFFIX=" in source
    assert "JUDGED_RUN_SUFFIX=" in source
    assert "${SHARD_A_RUN}${RUN_SUFFIX}" in source
    assert "${SHARD_C_RUN}${JUDGED_RUN_SUFFIX}" in source
    assert "MODEL_GPU_A" in source
    assert "MODEL_GPU_B" in source
    assert "MODEL_GPU_C" in source
    assert "JUDGE_GPU" in source
    assert "project_vision_opd_v1_offline_complement.yaml" in source
    assert "merge_project15_complement_shards" in source


def test_merge_project15_complement_shards(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmarks": {
                    benchmark_id: {}
                    for benchmark_id in (
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
                }
            }
        ),
        encoding="utf-8",
    )

    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    shard_c_nojudge = tmp_path / "shard_c_nojudge"
    shard_c_judged = tmp_path / "shard_c_judged"
    for shard, benchmarks in (
        (shard_a, ["vqav2"]),
        (shard_b, ["mindcube", "scienceqa", "mmsi_bench", "blink"]),
        (shard_c_nojudge, ["mv_math"]),
        (shard_c_judged, ["mathverse", "mathvista", "mmvet"]),
    ):
        shard.mkdir()
        _make_manifest(shard / "run_manifest.json", benchmarks=benchmarks, run_name=shard.name)
        for benchmark in benchmarks:
            benchmark_dir = shard / "lmms" / benchmark
            benchmark_dir.mkdir(parents=True)
            (benchmark_dir / "result.json").write_text("{}", encoding="utf-8")
    (shard_c_nojudge / "replay").mkdir()
    (shard_c_nojudge / "replay" / "mv_math.jsonl").write_text("", encoding="utf-8")

    output_run = tmp_path / "merged"
    merged = merge_shards(
        output_run,
        [shard_a, shard_b, shard_c_nojudge, shard_c_judged],
        config_path=config_path,
    )

    assert [record["benchmark_id"] for record in merged["runs"]] == [
        "vqav2",
        "mindcube",
        "scienceqa",
        "mmsi_bench",
        "blink",
        "mathverse",
        "mathvista",
        "mmvet",
        "mv_math",
    ]
    assert (output_run / "lmms" / "vqav2").is_symlink()
    assert (output_run / "lmms" / "mindcube").is_symlink()
    assert (output_run / "lmms" / "mathverse").is_symlink()
    assert (output_run / "replay").is_symlink()
