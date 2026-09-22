from pathlib import Path


def test_parallel_runner_does_not_double_append_smoke_suffix() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/eval/run_target_benchmarks_v2_parallel.sh"
    source = script.read_text(encoding="utf-8")
    assert 'SHARD_A_BASE="${BASE_RUN_NAME}_v2a"' in source
    assert 'SHARD_B_BASE="${BASE_RUN_NAME}_v2b"' in source
    assert 'export EVAL_RUN_NAME="${SHARD_A_BASE}"' in source
    assert 'export EVAL_RUN_NAME="${SHARD_B_BASE}"' in source
    assert 'SMOKE_SUFFIX="${SMOKE_SUFFIX}"' not in source
