"""Fail-fast checks shared by canonical evaluation entrypoints."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REMI_REPLAY = Path("assets/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl")
MV_MATH_REPLAY = Path("assets/remi_replay/raw_responses/qwen3vl8b_MV_MATH_len65536_maxtok1024_raw.jsonl")
MV_MATH_METADATA = Path("MV-MATH/MV-MATH.json")


def validate_environment(environment: Path) -> None:
    """Require the conda-style executables used by both v1 and v2 runners."""
    python = environment / "bin" / "python"
    vllm = environment / "bin" / "vllm"
    missing = [str(path) for path in (python, vllm) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"HW_EVAL_ENV is not a complete evaluation environment: {environment}. "
            f"Missing executable(s): {', '.join(missing)}"
        )
    probe = subprocess.run(
        [str(python), "-c", "import lmms_eval, jinja2"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"HW_EVAL_ENV cannot import the evaluation runtime: {environment}. "
            f"Required modules: lmms_eval, jinja2. stderr: {probe.stderr.strip()}"
        )


def validate_cuda_toolchain(toolchain: Path) -> None:
    nvcc = toolchain / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(
            "CUDA toolchain not found. Set EVAL_CUDA_TOOLCHAIN and SFT_RL_CUDA_TOOLCHAIN "
            f"to a directory containing bin/nvcc; checked {nvcc}"
        )


def validate_protocol_assets(
    repo_root: Path,
    *,
    protocol: str,
    dataset_root: Path | None = None,
) -> None:
    """Check ignored/local assets before vLLM startup.

    Git worktrees do not contain ignored benchmark assets. A canonical run must
    fail before loading models if those assets are not linked or configured.
    """
    if protocol not in {"b6-mixed", "project15", "project15-complement"}:
        raise ValueError(f"unknown protocol: {protocol}")

    resolved_root = dataset_root or repo_root / "assets" / "datasets"
    if not resolved_root.is_dir():
        raise FileNotFoundError(
            "benchmark dataset root not found. Git worktrees do not include ignored assets. "
            f"Set DTOPD_DATASET_ROOT or link {repo_root / 'assets' / 'datasets'}; checked {resolved_root}"
        )

    if protocol in {"b6-mixed", "project15"}:
        remi_replay = repo_root / REMI_REPLAY
        if not remi_replay.is_file():
            raise FileNotFoundError(
                "ReMI replay source not found. Git worktrees do not include ignored assets. "
                f"Link assets/remi_replay or use the main checkout; checked {remi_replay}"
            )
        remi_dataset = resolved_root / "ReMI"
        if not remi_dataset.is_dir() or not any(remi_dataset.glob("*.parquet")):
            raise FileNotFoundError(
                f"ReMI parquet files not found under {remi_dataset}"
            )

    if protocol in {"project15", "project15-complement"}:
        mv_math_replay = repo_root / MV_MATH_REPLAY
        if not mv_math_replay.is_file():
            raise FileNotFoundError(
                "MV-MATH replay source not found. Git worktrees do not include ignored assets. "
                f"Link assets/remi_replay or use the main checkout; checked {mv_math_replay}"
            )
        mv_math_metadata = resolved_root / "MV-MATH" / "MV-MATH.json"
        if not mv_math_metadata.is_file():
            raise FileNotFoundError(f"MV-MATH metadata not found: {mv_math_metadata}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--cuda-toolchain", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        choices=("b6-mixed", "project15", "project15-complement"),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    validate_environment(args.environment.expanduser())
    validate_cuda_toolchain(args.cuda_toolchain.expanduser())
    validate_protocol_assets(
        args.repo_root.expanduser(),
        protocol=args.protocol,
        dataset_root=args.dataset_root.expanduser() if args.dataset_root else None,
    )


if __name__ == "__main__":
    main()
