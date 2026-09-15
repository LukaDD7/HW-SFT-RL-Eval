"""Plan and run the contract benchmark suite against a Vision-OPD checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# ---------------------------------------------------------------------------
# Per-process signal flag – set by the SIGTERM/SIGINT handler so the main
# loop can interrupt gracefully after the current benchmark finishes.
# ---------------------------------------------------------------------------
_interrupted = False


def _handle_interrupt(signum: int, _frame: Any) -> None:
    global _interrupted
    _interrupted = True
    # Re-install default handler so a second signal kills hard.
    signal.signal(signum, signal.SIG_DFL)


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        value = _ENV_DEFAULT_RE.sub(
            lambda match: os.environ.get(match.group(1), match.group(2)), value
        )
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    contract_name: str
    category: str
    runner: str
    task: str | None
    source_file: str | None
    metric_tier: str
    scoring: str
    primary_metric: str | None
    max_new_tokens: int
    judge_required: bool
    note: str = ""

    @classmethod
    def from_mapping(cls, benchmark_id: str, value: Mapping[str, Any]) -> "BenchmarkSpec":
        required = ("contract_name", "category", "runner", "metric_tier", "scoring")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"benchmark {benchmark_id!r} missing fields: {', '.join(missing)}")
        runner = str(value["runner"])
        task = value.get("task")
        source_file = value.get("source_file")
        if runner == "lmms_eval" and not task:
            raise ValueError(f"benchmark {benchmark_id!r} requires task for lmms_eval")
        if runner == "replay_openai" and not source_file:
            raise ValueError(f"benchmark {benchmark_id!r} requires source_file for replay")
        return cls(
            benchmark_id=benchmark_id,
            contract_name=str(value["contract_name"]),
            category=str(value["category"]),
            runner=runner,
            task=str(task) if task else None,
            source_file=str(source_file) if source_file else None,
            metric_tier=str(value["metric_tier"]),
            scoring=str(value["scoring"]),
            primary_metric=(str(value["primary_metric"]) if value.get("primary_metric") else None),
            max_new_tokens=int(value.get("max_new_tokens", 1024)),
            judge_required=bool(value.get("judge_required", False)),
            note=str(value.get("note", "")),
        )


@dataclass(frozen=True)
class SuiteConfig:
    path: Path
    raw: dict[str, Any]
    defaults: dict[str, Any]
    backend: dict[str, Any]
    profiles: dict[str, list[str]]
    benchmarks: dict[str, BenchmarkSpec]


def load_suite(path: str | Path) -> SuiteConfig:
    config_path = Path(path).expanduser().resolve()
    raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("suite config must be a mapping")
    raw = _expand_env(raw_value)
    defaults = dict(raw.get("defaults", {}))
    backend = dict(raw.get("backend", {}))
    profiles = {str(k): [str(item) for item in v] for k, v in raw.get("profiles", {}).items()}
    benchmarks = {
        str(key): BenchmarkSpec.from_mapping(str(key), value)
        for key, value in raw.get("benchmarks", {}).items()
    }
    for profile, ids in profiles.items():
        unknown = [item for item in ids if item not in benchmarks]
        if unknown:
            raise ValueError(f"profile {profile!r} has unknown benchmarks: {unknown}")
    return SuiteConfig(config_path, raw, defaults, backend, profiles, benchmarks)


def select_benchmarks(
    suite: SuiteConfig, profile: str, explicit: Sequence[str] | None = None
) -> list[BenchmarkSpec]:
    ids = list(explicit or ())
    if not ids:
        if profile not in suite.profiles:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(suite.profiles)}")
        ids = suite.profiles[profile]
    unknown = [item for item in ids if item not in suite.benchmarks]
    if unknown:
        raise ValueError(f"unknown benchmark ids: {unknown}")
    return [suite.benchmarks[item] for item in ids]


def _model_args(
    *, backend: str, defaults: Mapping[str, Any], checkpoint: str, api_base: str
) -> str:
    if backend == "openai":
        values = {
            "model": defaults["served_model_name"],
            "base_url": api_base,
            "api_key": defaults.get("api_key", "EMPTY"),
            "timeout": 600,
            "num_concurrent": defaults.get("workers", 8),
            "httpx_trust_env": "false",
        }
    elif backend == "async_openai":
        values = {
            "model": defaults["served_model_name"],
            "base_url": api_base,
            "api_key": defaults.get("api_key", "EMPTY"),
            "timeout": 600,
            "is_qwen3_vl": "true",
        }
        if system_instruction := os.environ.get("SFT_RL_SYSTEM_INSTRUCTION"):
            values["system_prompt"] = system_instruction
    elif backend == "vllm":
        values = {
            "model": checkpoint,
            "tensor_parallel_size": defaults.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": defaults.get("gpu_memory_utilization", 0.85),
            "max_model_len": defaults.get("max_model_len", 65536),
            "reasoning_parser": "qwen3",
            "trust_remote_code": "true",
        }
    else:
        raise ValueError(f"unsupported inference backend: {backend}")
    return ",".join(f"{key}={value}" for key, value in values.items())


def build_command(
    spec: BenchmarkSpec,
    *,
    suite: SuiteConfig,
    run_dir: Path,
    python: str,
    inference_backend: str,
    checkpoint: str,
    api_base: str,
    limit: float | None,
    judge_policy: str,
) -> list[str] | None:
    if spec.judge_required and judge_policy == "defer":
        return None
    defaults = suite.defaults
    if spec.runner == "lmms_eval":
        inference_backend = (
            "async_openai"
            if os.environ.get("SFT_RL_SYSTEM_INSTRUCTION")
            else inference_backend
        )
        max_new_tokens = spec.max_new_tokens
        if override := os.environ.get("SFT_RL_MAX_NEW_TOKENS_OVERRIDE"):
            max_new_tokens = int(override)
        command = [
            python,
            "-m",
            "lmms_eval",
            "--model",
            inference_backend,
            "--model_args",
            _model_args(
                backend=inference_backend,
                defaults=defaults,
                checkpoint=checkpoint,
                api_base=api_base,
            ),
            "--tasks",
            str(spec.task),
            "--batch_size",
            str(defaults.get("batch_size", 1)),
            "--seed",
            str(defaults.get("seed", 42)),
            "--gen_kwargs",
            f"temperature=0,max_new_tokens={max_new_tokens}",
            "--log_samples",
            "--log_samples_suffix",
            f"vision_opd_{spec.benchmark_id}",
            "--output_path",
            str(run_dir / "lmms" / spec.benchmark_id),
            "--use_cache",
            str(run_dir / "cache"),
            "--trust_remote_code",
            "--show_config",
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        include_task_path = defaults.get("include_task_path")
        if include_task_path:
            # Extra task directories (e.g. eval_tasks/opd_v2 overrides) are
            # indexed after the bundled lmms-eval tasks, so override tasks
            # must use fresh task names to survive conflict resolution.
            command.extend(["--include_path", str(include_task_path)])
        if spec.judge_required and judge_policy == "predict":
            command.append("--predict_only")
        return command

    if spec.runner == "replay_openai":
        if inference_backend != "openai":
            return None
        source = Path(str(defaults["prior_raw_root"])) / str(spec.source_file)
        command = [
            python,
            "-m",
            "dual_track_opd.eval.run_vlm_eval",
            "--input-jsonl",
            str(source),
            "--output-jsonl",
            str(run_dir / "replay" / f"{spec.benchmark_id}.jsonl"),
            "--dataset",
            spec.contract_name,
            "--dataset-root",
            str(defaults["dataset_root"]),
            "--api-base",
            api_base,
            "--api-key",
            str(defaults.get("api_key", "EMPTY")),
            "--model",
            str(defaults["served_model_name"]),
            "--max-tokens",
            str(spec.max_new_tokens),
            "--workers",
            str(defaults.get("workers", 8)),
            "--resume",
        ]
        if limit is not None:
            command.extend(["--limit", str(int(limit))])
        return command
    raise ValueError(f"unsupported runner {spec.runner!r} for {spec.benchmark_id}")


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo, check=False, text=True, capture_output=True
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def _dataset_manifest_hash(specs: Sequence[BenchmarkSpec], backend: Mapping[str, Any]) -> str:
    payload = {
        "backend_commit": backend.get("commit"),
        "datasets": [
            {
                "id": spec.benchmark_id,
                "task": spec.task,
                "source_file": spec.source_file,
                "scoring": spec.scoring,
            }
            for spec in specs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _judge_environment() -> dict[str, str]:
    required = ("JUDGE_API_KEY", "JUDGE_API_URL", "JUDGE_MODEL")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(
            "judge scoring requires environment variables: " + ", ".join(missing)
        )
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": os.environ["JUDGE_API_KEY"],
            "OPENAI_API_URL": os.environ["JUDGE_API_URL"],
            "MODEL_VERSION": os.environ["JUDGE_MODEL"],
            "API_TYPE": os.environ.get("JUDGE_API_TYPE", "openai"),
        }
    )
    return env


def _check_openai_endpoint(api_base: str, api_key: str) -> None:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Vision-OPD endpoint is not ready at {api_base}; "
            "start scripts/eval/start_vision_opd_server.sh first"
        ) from exc


def _print_plan(
    rows: Sequence[tuple[BenchmarkSpec, list[str] | None]],
    judge_policy: str,
    skip_ids: set[str] | None = None,
) -> None:
    skip_ids = skip_ids or set()
    print("id\tcategory\ttier\tscoring\tstatus")
    for spec, command in rows:
        if spec.benchmark_id in skip_ids:
            status = "skipped:resume"
        elif command is None:
            if spec.judge_required and judge_policy == "defer":
                status = "deferred:judge"
            else:
                status = "deferred:requires_openai_endpoint"
        elif spec.judge_required and judge_policy == "predict":
            status = "predict_only"
        else:
            status = "ready"
        print(
            f"{spec.benchmark_id}\t{spec.category}\t{spec.metric_tier}\t"
            f"{spec.scoring}\t{status}"
        )
    print("\nCommands:")
    for spec, command in rows:
        if command:
            print(f"\n# {spec.contract_name}\n{shlex.join(command)}")


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the run manifest so crashes never produce a partial file."""
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)


def run_suite(args: argparse.Namespace) -> int:
    global _interrupted

    suite = load_suite(args.config)
    explicit = [part for value in args.benchmarks for part in value.split(",") if part]
    specs = select_benchmarks(suite, args.profile, explicit or None)
    checkpoint = str(Path(args.checkpoint or suite.defaults["checkpoint"]).expanduser())
    api_base = str(args.api_base or suite.defaults["api_base"])
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")

    # ---- resume logic ---------------------------------------------------
    existing_manifest: dict[str, Any] | None = None
    completed_ids: set[str] = set()
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"resume run directory not found: {resume_path}")
        manifest_candidate = resume_path / "run_manifest.json"
        if not manifest_candidate.is_file():
            raise FileNotFoundError(f"manifest not found in resume directory: {manifest_candidate}")
        existing_manifest = json.loads(manifest_candidate.read_text(encoding="utf-8"))
        for run_rec in existing_manifest.get("runs", []):
            if run_rec.get("status") == "completed":
                completed_ids.add(run_rec["benchmark_id"])
        if completed_ids:
            print(f"[resume] {len(completed_ids)} benchmark(s) already completed: {sorted(completed_ids)}")

        # Re-use the existing run directory so cache and intermediate outputs
        # are picked up automatically.
        run_dir = resume_path
        run_name = existing_manifest.get("run_name", run_dir.name)
        output_root = run_dir.parent
    else:
        run_name = args.run_name or f"{args.profile}_{timestamp}"
        output_root = Path(args.output_root or suite.defaults["output_root"]).expanduser()
        run_dir = output_root / run_name

    rows = [
        (
            spec,
            build_command(
                spec,
                suite=suite,
                run_dir=run_dir,
                python=args.python,
                inference_backend=args.inference_backend,
                checkpoint=checkpoint,
                api_base=api_base,
                limit=args.limit,
                judge_policy=args.judge_policy,
            ),
        )
        for spec in specs
    ]
    _print_plan(rows, args.judge_policy, completed_ids)
    if not args.execute:
        return 0

    if args.inference_backend == "vllm" and not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"Vision-OPD checkpoint not found: {checkpoint}")

    # Only check the endpoint if there are actually benchmarks to run.
    pending = [
        (spec, cmd) for spec, cmd in rows
        if cmd is not None and spec.benchmark_id not in completed_ids
    ]
    if pending and args.inference_backend == "openai":
        _check_openai_endpoint(api_base, str(suite.defaults.get("api_key", "EMPTY")))
    for spec, command in rows:
        if command is not None and spec.runner == "replay_openai" and spec.benchmark_id not in completed_ids:
            source = Path(str(suite.defaults["prior_raw_root"])) / str(spec.source_file)
            if not source.is_file():
                raise FileNotFoundError(f"prior raw replay source not found: {source}")
            dataset_root = Path(str(suite.defaults["dataset_root"]))
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    # ---- new run or resume ----------------------------------------------
    if existing_manifest is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        repo_root = Path(__file__).resolve().parents[3]
        manifest: dict[str, Any] = {
            "suite": suite.raw.get("suite", {}),
            "profile": args.profile,
            "run_name": run_name,
            "started_at": datetime.now().astimezone().isoformat(),
            "repo": _git_state(repo_root),
            "backend": suite.backend,
            "resolved_config": suite.raw,
            "dataset_manifest_hash": _dataset_manifest_hash(specs, suite.backend),
            "dataset_manifest_kind": "logical_task_registry",
            "checkpoint_path": checkpoint,
            "raw_output_path": str(run_dir),
            "inference_backend": args.inference_backend,
            "api_base": api_base,
            "judge_policy": args.judge_policy,
            "runs": [],
            "notes": "Raw outputs remain outside Git; judge-dependent metrics are explicitly classified.",
        }
    else:
        manifest = existing_manifest
        manifest.setdefault("notes", "")
        manifest["notes"] += (
            f" Resumed at {datetime.now().astimezone().isoformat()}; "
            f"{len(completed_ids)} benchmarks already complete."
        )

    manifest_path = run_dir / "run_manifest.json"
    _save_manifest(manifest_path, manifest)

    # ---- signal handling ------------------------------------------------
    # Install handlers so the run can be interrupted gracefully.
    # After the current benchmark finishes the loop exits and the manifest
    # is flushed – no partial state is lost (cache preserves LM responses).
    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    env = _judge_environment() if args.judge_policy == "score" else os.environ.copy()
    overall_rc = 0
    for spec, command in rows:
        if _interrupted:
            print(f"\n[interrupted] stopping before {spec.contract_name}", flush=True)
            break

        # ---- skip already-completed benchmarks on resume ----------------
        if spec.benchmark_id in completed_ids:
            print(f"\n[skip] {spec.contract_name} (already completed in prior run)", flush=True)
            continue

        record: dict[str, Any] = {
            "benchmark_id": spec.benchmark_id,
            "contract_name": spec.contract_name,
            "category": spec.category,
            "metric_tier": spec.metric_tier,
            "scoring": spec.scoring,
            "primary_metric": spec.primary_metric,
            "command": command,
        }
        if command is None:
            record["status"] = "deferred"
            record["reason"] = (
                "judge required" if spec.judge_required else "OpenAI-compatible endpoint required"
            )
        else:
            # ---- deterministic cache run-id  ---------------------------
            # Use a stable cache key so that the same benchmark re-using
            # the same --use_cache directory will hit previous responses.
            run_env = env.copy()
            run_env["LMMS_CACHE_RUN_ID"] = f"{manifest['run_name']}__{spec.benchmark_id}"

            print(f"\n[run] {spec.contract_name}", flush=True)
            completed = subprocess.run(command, check=False, env=run_env)
            record["returncode"] = completed.returncode
            output_dir = run_dir / "lmms" / spec.benchmark_id
            has_results = any(path.is_file() for path in output_dir.rglob("*.json"))
            succeeded = completed.returncode == 0 and has_results
            record["status"] = "completed" if succeeded else "failed"
            if not succeeded:
                record["failure_reason"] = (
                    "lmms_eval_returned_nonzero"
                    if completed.returncode != 0
                    else "lmms_eval_returned_no_result_files"
                )
                overall_rc = completed.returncode if completed.returncode != 0 else 1
                if not args.keep_going:
                    manifest["runs"].append(record)
                    _save_manifest(manifest_path, manifest)
                    break
        manifest["runs"].append(record)
        _save_manifest(manifest_path, manifest)

    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["returncode"] = overall_rc
    _save_manifest(manifest_path, manifest)

    if _interrupted:
        print("[interrupted] manifest saved — re-run with --resume-from to continue.", flush=True)
    return overall_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/eval/project_vision_opd.yaml", help="Suite YAML path."
    )
    parser.add_argument("--profile", default="acceptance_core")
    parser.add_argument(
        "--benchmarks", action="append", default=[], help="Comma-separated benchmark ids."
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--api-base")
    parser.add_argument("--output-root")
    parser.add_argument("--run-name")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inference-backend", choices=("openai", "vllm"), default="openai")
    parser.add_argument("--judge-policy", choices=("defer", "predict", "score"), default="defer")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--resume-from", help="Resume from a previous run directory (skip completed benchmarks).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run_suite(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
