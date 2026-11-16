"""One expert's B6 stages on isolated model and judge endpoints."""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
STAGES = ("v2a", "v2b", "remi", "mmbench")
_children: dict[str, subprocess.Popen] = {}


def value(name: str) -> str:
    result = os.environ.get(name, "")
    if not result:
        raise ValueError(f"missing {name}")
    return result


def port(name: str) -> int:
    result = int(value(name))
    if not 1024 <= result <= 65535:
        raise ValueError(f"invalid {name}: {result}")
    return result


def run(name: str, arguments: list[str], env: dict[str, str]) -> None:
    log_path = Path(value("B6_STAGE_LOG_ROOT")) / f"{name}.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(arguments, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        _children[name] = child
        print(f"[scoped-b6] {name}: pid={child.pid}; log={log_path}", flush=True)
        try:
            status = child.wait()
        finally:
            _children.pop(name, None)
    if status:
        raise RuntimeError(f"{name} failed with status {status}; see {log_path}")


def background(name: str, arguments: list[str], env: dict[str, str]) -> None:
    log_path = Path(value("B6_STAGE_LOG_ROOT")) / f"{name}.log"
    log = log_path.open("w")
    child = subprocess.Popen(arguments, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    _children[name] = child
    print(f"[scoped-b6] {name}: pid={child.pid}; log={log_path}", flush=True)


def terminate(*_args) -> None:
    for child in list(_children.values()):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def endpoint(port_number: int) -> str:
    return f"http://127.0.0.1:{port_number}/v1"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "EVAL_LIMIT", "SFT_RL_LIMIT", "EVAL_SMOKE", "SFT_RL_SMOKE",
        "EVAL_BENCHMARKS", "EVAL_JUDGE_BENCHMARKS", "SFT_RL_BENCHMARKS",
        "SFT_RL_JUDGE_BENCHMARKS", "SFT_RL_EVAL_CONFIG", "EVAL_CONFIG",
        "EVAL_V2_SECOND_GPU", "EVAL_V2_SECOND_PORT", "EVAL_V2_PARALLEL",
        "EVAL_MAX_NEW_TOKENS_OVERRIDE", "SFT_RL_MAX_NEW_TOKENS_OVERRIDE",
        "EVAL_KEEP_GOING", "SFT_RL_KEEP_GOING", "SFT_RL_SYSTEM_INSTRUCTION",
        "VISION_OPD_THINK_SYSTEM_PROMPT", "VISION_OPD_THINK_MATH_INSTRUCTION",
        "VISION_OPD_THINK_MATH_TASKS", "VISION_OPD_THINK_PROTOCOL_VERSION",
        "VISION_OPD_MAX_NEW_TOKENS_CAP", "CUDA_VISIBLE_DEVICES",
    ):
        env.pop(name, None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO / "src"),
        "EVAL_SEED": "42", "EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0",
        "EVAL_THINK_MODE": "auto", "SFT_RL_THINK_MODE": "auto",
        "EVAL_CKPT": value("EVAL_CKPT"),
        "EVAL_SERVED_MODEL": value("EVAL_SERVED_MODEL"),
        # The v1 runner reads SFT_RL_MODEL_HF/HW_MODEL_CKPT, SFT_RL_SERVED_MODEL,
        # and SFT_RL_RUN_NAME instead of the v2 EVAL_* names; keep both in sync
        # so one expert identity drives every lane.
        "SFT_RL_MODEL_HF": value("EVAL_CKPT"), "HW_MODEL_CKPT": value("EVAL_CKPT"),
        "SFT_RL_SERVED_MODEL": value("EVAL_SERVED_MODEL"),
        "SFT_RL_RUN_NAME": value("EVAL_RUN_NAME"),
        "EVAL_MAX_LEN": value("B6_MAX_MODEL_LEN"),
        "SFT_RL_MAX_LEN": value("B6_MAX_MODEL_LEN"),
        "EVAL_GPUS": value("B6_V2A_GPUS"), "EVAL_TP": value("B6_MODEL_TP"),
        "SFT_RL_EVAL_GPUS": value("B6_V2A_GPUS"), "SFT_RL_EVAL_TP": value("B6_MODEL_TP"),
        "SFT_RL_JUDGE_GPUS": value("B6_JUDGE_GPUS"), "SFT_RL_JUDGE_TP": value("B6_JUDGE_TP"),
        "SFT_RL_JUDGE_GPU": value("B6_JUDGE_GPUS").split(",")[0],
        "EVAL_JUDGE_GPU": value("B6_JUDGE_GPUS").split(",")[0],
        "EVAL_JUDGE_PORT": str(port("B6_JUDGE_PORT")),
        "SFT_RL_JUDGE_PORT": str(port("B6_JUDGE_PORT")),
        "SFT_RL_JUDGE_MAX_LEN": value("B6_JUDGE_MAX_MODEL_LEN"),
        "SFT_RL_JUDGE_HF": value("SFT_RL_JUDGE_HF"),
        "no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost",
    })
    if os.environ.get("B6_SMOKE") == "1":
        env["EVAL_SMOKE"] = "1"
        # The v1 runner keys its 8-row limit AND its _smoke run-name suffix
        # off SFT_RL_SMOKE (it appends the suffix itself, so never append it
        # here — that produced _smoke_smoke). v2 keys off EVAL_SMOKE.
        env["SFT_RL_SMOKE"] = "1"
    # GPU utilization: the 18h instance reclaim makes wall time the binding
    # constraint. Raise the client concurrency (workers -> num_concurrent /
    # --workers via B6_WORKERS_OVERRIDE) and the server-side batch width
    # (max_num_seqs) together. Each TP=2 shard owns two dedicated GPUs, so
    # there is no cross-lane contention; the judge is likewise isolated on 4,5.
    workers = os.environ.get("B6_WORKERS", "32")
    env["B6_WORKERS_OVERRIDE"] = workers
    # Observed on the first full run: v2 KV cache peaked at ~4.5% with 64, so
    # the server batch width was the throughput ceiling, not memory. 256 is
    # vLLM's default and still far below KV capacity; lower via B6_MAX_NUM_SEQS
    # only if a server becomes unstable.
    env["EVAL_MAX_NUM_SEQS"] = os.environ.get("B6_MAX_NUM_SEQS", "256")
    return env


def run_v2(env: dict[str, str]) -> None:
    base = value("EVAL_RUN_NAME")
    for name, gpu_name, port_name, benchmarks in (
        ("v2a", "B6_V2A_GPUS", "B6_V2A_PORT", "gqa,dynamath"),
        ("v2b", "B6_V2B_GPUS", "B6_V2B_PORT", "viewspatial,mmmu_pro"),
    ):
        child = env | {
            "EVAL_RUN_NAME": f"{base}_{name}",
            "V2_BENCHMARKS": benchmarks,
            "EVAL_GPU": value(gpu_name).split(",")[0],
            "EVAL_GPUS": value(gpu_name),
            "EVAL_PORT": str(port(port_name)),
        }
        background(name, ["bash", str(REPO / "scripts/eval/run_target_benchmarks_v2.sh")], child)
    failures = []
    for name in ("v2a", "v2b"):
        child = _children.pop(name)
        status = child.wait()
        print(f"[scoped-b6] {name}: exit={status}", flush=True)
        if status:
            failures.append(name)
    if failures:
        raise RuntimeError(f"v2 failed: {','.join(failures)}; shard outputs preserved")
    suffix = "_smoke" if env.get("EVAL_SMOKE") == "1" else ""
    subprocess.run([
        value("B6_PYTHON"), "-m", "dual_track_opd.eval.merge_v2_shards",
        "--output-run", str(Path(value("DTOPD_EVAL_ROOT")) / f"{base}{suffix}"),
        "--shard-run", str(Path(value("DTOPD_EVAL_ROOT")) / f"{base}_v2a{suffix}"),
        "--shard-run", str(Path(value("DTOPD_EVAL_ROOT")) / f"{base}_v2b{suffix}"),
        "--config", str(REPO / "configs/eval/project_vision_opd_v2.yaml"),
    ], cwd=REPO, env=env, check=True)


def run_v1(env: dict[str, str]) -> None:
    base = value("EVAL_RUN_NAME")
    for name, gpu_name, port_name, benchmark, judged in (
        ("remi", "B6_REMI_GPUS", "B6_REMI_PORT", "remi", False),
        ("mmbench", "B6_MMBENCH_GPUS", "B6_MMBENCH_PORT", "mmbench", True),
    ):
        child = env | {
            "SFT_RL_EVAL_GPU": value(gpu_name).split(",")[0],
            "SFT_RL_EVAL_GPUS": value(gpu_name),
            "SFT_RL_EVAL_PORT": str(port(port_name)),
            "SFT_RL_BENCHMARKS": "" if judged else benchmark,
            "SFT_RL_JUDGE_BENCHMARKS": benchmark if judged else "",
            "SFT_RL_EVAL_CONFIG": "configs/eval/project_vision_opd.yaml",
        }
        # Call the pinned v1 runtime directly: its outer wrapper would apply the
        # MMBench patch concurrently in both lanes before either one starts.
        background(name, ["bash", str(REPO / "scripts/sft_rl/run_sftrl_benchmarks.sh")], child)
    failures = []
    for name in ("remi", "mmbench"):
        child = _children.pop(name)
        status = child.wait()
        print(f"[scoped-b6] {name}: exit={status}", flush=True)
        if status:
            failures.append(name)
    if failures:
        raise RuntimeError(f"v1 failed: {','.join(failures)}; shard outputs preserved")
    suffix = "_smoke" if env.get("EVAL_SMOKE") == "1" else ""
    replay = Path(value("DTOPD_EVAL_ROOT")) / f"{base}{suffix}_nojudge/replay/remi.jsonl"
    if not replay.is_file() or not replay.stat().st_size:
        raise RuntimeError(f"missing ReMI replay: {replay}")
    arguments = [
        value("B6_PYTHON"), "-m", "dual_track_opd.eval.score_remi",
        "--replay", str(replay), "--out", str(replay.parents[1] / "remi_result.json"),
    ]
    if suffix:
        arguments.append("--smoke")
    subprocess.run(arguments, cwd=REPO, env=env, check=True)


def main() -> None:
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    Path(value("B6_STAGE_LOG_ROOT")).mkdir(parents=True, exist_ok=True)
    endpoints = [port(name) for name in (
        "B6_V2A_PORT", "B6_V2B_PORT", "B6_REMI_PORT", "B6_MMBENCH_PORT", "B6_JUDGE_PORT")]
    if endpoints != [8000, 8001, 8000, 8002, 8004]:
        raise ValueError("SCOPED B6 endpoints must be 8000,8001,8000,8002,8004")
    env = environment()
    if not re.fullmatch(r"SCOPED-C[1-4]", env["EVAL_SERVED_MODEL"]):
        raise ValueError("SCOPED B6 requires an explicit SCOPED-C1..C4 served alias")
    try:
        run_v2(env)
        run_v1(env)
    finally:
        terminate()
        for child in list(_children.values()):
            child.wait()
    print(f"[scoped-b6] complete: {env['EVAL_RUN_NAME']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
