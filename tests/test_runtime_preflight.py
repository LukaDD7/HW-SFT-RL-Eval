from pathlib import Path

import pytest
import subprocess

from dual_track_opd.eval.runtime_preflight import (
    REMI_REPLAY,
    MV_MATH_METADATA,
    MV_MATH_REPLAY,
    validate_cuda_toolchain,
    validate_environment,
    validate_protocol_assets,
)


def make_environment(root: Path, *, complete: bool = True) -> Path:
    root.mkdir(parents=True)
    (root / "bin").mkdir()
    python = root / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    if complete:
        (root / "bin" / "vllm").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (root / "bin" / "vllm").chmod(0o755)
    return root


def test_environment_requires_python_and_vllm(tmp_path: Path) -> None:
    environment = make_environment(tmp_path / "env", complete=False)
    with pytest.raises(FileNotFoundError, match="Missing executable"):
        validate_environment(environment)


def test_complete_environment_passes(tmp_path: Path) -> None:
    environment = make_environment(tmp_path / "env")
    validate_environment(environment)


def test_environment_must_import_eval_runtime(tmp_path: Path) -> None:
    environment = make_environment(tmp_path / "env")
    python = environment / "bin" / "python"
    python.write_text("#!/bin/sh\necho 'ModuleNotFoundError' >&2\nexit 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lmms_eval, jinja2"):
        validate_environment(environment)


def test_cuda_toolchain_requires_nvcc(tmp_path: Path) -> None:
    toolchain = tmp_path / "cuda"
    toolchain.mkdir()
    with pytest.raises(FileNotFoundError, match="EVAL_CUDA_TOOLCHAIN and SFT_RL_CUDA_TOOLCHAIN"):
        validate_cuda_toolchain(toolchain)


def test_cuda_toolchain_passes_with_nvcc(tmp_path: Path) -> None:
    toolchain = tmp_path / "cuda"
    (toolchain / "bin").mkdir(parents=True)
    (toolchain / "bin" / "nvcc").touch()
    validate_cuda_toolchain(toolchain)


def test_b6_requires_remi_replay_and_parquet(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="benchmark dataset root not found"):
        validate_protocol_assets(repo, protocol="b6-mixed")

    dataset_root = tmp_path / "datasets"
    (dataset_root / "ReMI").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="ReMI replay source not found"):
        validate_protocol_assets(repo, protocol="b6-mixed", dataset_root=dataset_root)

    remi = repo.joinpath(REMI_REPLAY)
    remi.parent.mkdir(parents=True)
    remi.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ReMI parquet files not found"):
        validate_protocol_assets(repo, protocol="b6-mixed", dataset_root=dataset_root)

    (dataset_root / "ReMI" / "test.parquet").write_bytes(b"parquet")
    validate_protocol_assets(repo, protocol="b6-mixed", dataset_root=dataset_root)


def test_project15_requires_remi_and_mv_math_assets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(FileNotFoundError, match="benchmark dataset root not found"):
        validate_protocol_assets(repo, protocol="project15")

    dataset_root = tmp_path / "datasets"
    (dataset_root / "ReMI").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="ReMI replay source not found"):
        validate_protocol_assets(repo, protocol="project15", dataset_root=dataset_root)

    remi = repo.joinpath(REMI_REPLAY)
    remi.parent.mkdir(parents=True)
    remi.write_text("{}", encoding="utf-8")
    (dataset_root / "ReMI" / "test.parquet").write_bytes(b"parquet")
    with pytest.raises(FileNotFoundError, match="MV-MATH replay source not found"):
        validate_protocol_assets(repo, protocol="project15", dataset_root=dataset_root)

    mv_math = repo.joinpath(MV_MATH_REPLAY)
    mv_math.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="MV-MATH metadata not found"):
        validate_protocol_assets(repo, protocol="project15", dataset_root=dataset_root)

    metadata = dataset_root.joinpath(MV_MATH_METADATA)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("[]", encoding="utf-8")
    validate_protocol_assets(repo, protocol="project15", dataset_root=dataset_root)


def test_project15_complement_requires_mv_math_but_not_remi(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    with pytest.raises(FileNotFoundError, match="MV-MATH replay source not found"):
        validate_protocol_assets(
            repo,
            protocol="project15-complement",
            dataset_root=dataset_root,
        )

    mv_math = repo.joinpath(MV_MATH_REPLAY)
    mv_math.parent.mkdir(parents=True)
    mv_math.write_text("{}", encoding="utf-8")
    metadata = dataset_root.joinpath(MV_MATH_METADATA)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("[]", encoding="utf-8")
    validate_protocol_assets(
        repo,
        protocol="project15-complement",
        dataset_root=dataset_root,
    )


def test_canonical_entrypoint_normalizes_runner_variables() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/eval/run_pinned_eval.sh"
    source = script.read_text(encoding="utf-8")
    assert 'CUDA_TOOLCHAIN="${EVAL_CUDA_TOOLCHAIN:-${SFT_RL_CUDA_TOOLCHAIN' in source
    assert "export EVAL_CUDA_TOOLCHAIN" in source
    assert "export SFT_RL_CUDA_TOOLCHAIN" in source
    assert "dual_track_opd.eval.runtime_preflight" in source
    assert "--dataset-root" in source


def test_canonical_entrypoint_help_does_not_require_environment() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/eval/run_pinned_eval.sh"
    result = subprocess.run(
        ["bash", str(script), "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "b6-mixed" in result.stdout


def test_canonical_entrypoint_preflight_uses_repository_pythonpath(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/eval/run_pinned_eval.sh"
    source = script.read_text(encoding="utf-8")
    assert 'export PYTHONPATH="${REPO_ROOT}/src' in source


def test_canonical_entrypoint_propagates_hf_cache_to_v1_runner() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/eval/run_pinned_eval.sh"
    source = script.read_text(encoding="utf-8")
    assert 'export SFT_RL_HF_CACHE="${SFT_RL_HF_CACHE:-${EVAL_HF_CACHE}}"' in source
