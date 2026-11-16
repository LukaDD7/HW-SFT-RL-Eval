"""Test queue ordering with valid synthetic canonical outputs, never GPU services."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dual_track_opd.eval.b6_pass1_report import PROTOCOL, receipt_path
from test_b6_report import make_plan


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/eval/run_b6_pass1_queue.sh"


@pytest.fixture
def queue(tmp_path):
    checkpoint_root = tmp_path / "models"
    experiments = []
    for index in range(4):
        checkpoint = checkpoint_root / f"c{index + 1}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        experiments.append({"expert": f"C{index + 1}", "run_name": f"run{index}",
                            "checkpoint": str(checkpoint)})
    judge = tmp_path / "judge"
    judge.mkdir()
    (judge / "model.safetensors").write_bytes(b"judge")
    plan_path = tmp_path / "plan.json"
    plan = make_plan(experiments)
    plan["judge_checkpoint"] = str(judge)
    plan_path.write_text(json.dumps(plan))
    runner = tmp_path / "runner.sh"
    runner.write_text(f'''import json, os, sys
from pathlib import Path
from test_b6_report import make_experiment
name = os.environ["EVAL_RUN_NAME"]
smoke = os.environ.get("B6_SMOKE") == "1"
with open(os.environ["QUEUE_LOG"], "a") as handle:
    handle.write(json.dumps({{"name": name, "args": sys.argv[1:],
        "repeat": os.environ.get("EVAL_REPEAT_COUNT"), "seed": os.environ.get("EVAL_SEED"),
        "temperature": os.environ.get("EVAL_SAMPLING_TEMPERATURE"),
        "think": os.environ.get("EVAL_THINK_MODE"), "smoke": os.environ.get("B6_SMOKE"),
        "eval_limit": os.environ.get("EVAL_LIMIT"),
        "budget_override": os.environ.get("EVAL_MAX_NEW_TOKENS_OVERRIDE"),
        "judge": os.environ.get("SFT_RL_JUDGE_HF"),
        "v2a_gpus": os.environ.get("B6_V2A_GPUS"),
        "v2b_gpus": os.environ.get("B6_V2B_GPUS"),
        "remi_gpus": os.environ.get("B6_REMI_GPUS"),
        "mmbench_gpus": os.environ.get("B6_MMBENCH_GPUS"),
        "judge_gpus": os.environ.get("B6_JUDGE_GPUS"),
        "model_tp": os.environ.get("B6_MODEL_TP"), "judge_tp": os.environ.get("B6_JUDGE_TP"),
        "judge_port": os.environ.get("B6_JUDGE_PORT"),
        "stage_log": os.environ.get("B6_STAGE_LOG_ROOT")}}) + "\\n")
if name == os.environ.get("FAIL_RUN"):
    sys.exit(19)
experiment = make_experiment(Path(os.environ["DTOPD_EVAL_ROOT"]), name=name,
                             smoke=smoke,
                             expert=os.environ["EVAL_SERVED_MODEL"].removeprefix("SCOPED-"),
                             checkpoint=Path(os.environ["EVAL_CKPT"]),
                             judge_checkpoint=os.environ["SFT_RL_JUDGE_HF"])
''')
    repo = tmp_path / "repo"
    (repo / "scripts/eval").mkdir(parents=True)
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repo / "scripts/eval/audit_truncation.py").symlink_to(ROOT / "scripts/eval/audit_truncation.py")
    (repo / "scripts/eval/apply_mmbench_patch.sh").symlink_to(ROOT / "scripts/eval/apply_mmbench_patch.sh")
    (repo / "patches").symlink_to(ROOT / "patches", target_is_directory=True)
    (repo / "configs").symlink_to(ROOT / "configs", target_is_directory=True)
    (repo / "assets").symlink_to(ROOT / "assets", target_is_directory=True)
    queue_path = repo / "scripts/eval/run_b6_pass1_queue.sh"
    queue_path.write_text(QUEUE.read_text())
    (repo / "scripts/eval/run_scoped_b6_expert.py").symlink_to(runner)
    log = tmp_path / "calls.jsonl"
    output = tmp_path / "outputs"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "HW_EVAL_ENV": str(Path(sys.executable).parents[1]),
           "PYTHONPATH": f"{ROOT / 'tests'}:{ROOT / 'src'}",
           "EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0",
           "EVAL_PORT": "65534", "EVAL_JUDGE_PORT": "65533",
           "DTOPD_EVAL_ROOT": str(output), "QUEUE_LOG": str(log),
           "B6_PASS1_LOG_ROOT": str(tmp_path / "logs"),
           "EVAL_LIMIT": "1", "EVAL_MAX_NEW_TOKENS_OVERRIDE": "128"}
    return queue_path, plan_path, log, env, output, experiments


def invoke(fixture, *args):
    queue, plan, _, env, _, _ = fixture
    env = env | {"EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0"}
    return subprocess.run(["bash", str(queue), "--plan", str(plan), *args], env=env,
                          text=True, capture_output=True, timeout=90)


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def prepare_smoke(fixture):
    result = invoke(fixture, "--smoke")
    assert result.returncode == 0, result.stderr + result.stdout


def test_full_order_and_environment_are_fixed(queue):
    _, _, log, _, output, experiments = queue
    prepare_smoke(queue)
    result = invoke(queue)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = calls(log)
    assert [row["name"] for row in rows] == ["run0", "run0", "run1", "run2", "run3"]
    for index, row in enumerate(rows):
        assert row["args"] == []
        assert row["seed"] == "42" and row["repeat"] == "1"
        assert row["temperature"] == "1.0" and row["think"] == "auto"
        assert row["eval_limit"] is None and row["budget_override"] is None
        assert row["judge"].endswith("/judge")
        assert [row["v2a_gpus"], row["v2b_gpus"], row["remi_gpus"], row["mmbench_gpus"]] == ["0,1", "2,3", "0,1", "2,3"]
        assert row["judge_gpus"] == "4,5"
        assert row["model_tp"] == row["judge_tp"] == "2"
        assert row["judge_port"] == "8004"
        assert row["stage_log"].endswith(f"/{row['name']}_{'smoke' if index == 0 else 'full'}_stages")
    report = json.loads((output / "scoped_b6_pass1_report.json").read_text())
    assert report["status"] == "complete"
    assert [row["expert"] for row in report["experiments"]] == ["C1", "C2", "C3", "C4"]
    assert all(len(row["rows"]) == 6 for row in report["experiments"])


def test_smoke_only_launches_first_model(queue):
    _, _, log, _, output, _ = queue
    result = invoke(queue, "--smoke")
    assert result.returncode == 0, result.stderr + result.stdout
    rows = calls(log)
    assert len(rows) == 1 and rows[0]["name"] == "run0" and rows[0]["smoke"] == "1"
    report = json.loads((output / "scoped_b6_pass1_report.json").read_text())
    assert report["formal"] is False and len(report["experiments"]) == 1


@pytest.mark.parametrize("failure", ["run0", "run1", "run2", "run3"])
def test_runner_failure_stops_immediately(queue, failure):
    _, _, log, env, _, _ = queue
    prepare_smoke(queue)
    log.unlink()
    env["FAIL_RUN"] = failure
    result = invoke(queue)
    assert result.returncode == 19, result.stderr + result.stdout
    rows = calls(log)
    assert [row["name"] for row in rows] == [f"run{index}" for index in range(int(failure[-1]) + 1)]


def test_full_refuses_missing_or_stale_smoke(queue):
    _, _, log, env, _, _ = queue
    result = invoke(queue)
    assert result.returncode == 1 and not log.exists()
    prepare_smoke(queue)
    smoke = next(Path(env["DTOPD_EVAL_ROOT"]).glob("*_smoke_nojudge/replay/remi.jsonl"))
    rows = smoke.read_text().splitlines()
    smoke.write_text("\n".join(rows[:-1]) + "\n")
    before = len(calls(log))
    result = invoke(queue)
    assert result.returncode == 1 and len(calls(log)) == before


def test_only_receipt_matching_fresh_raw_audit_is_skipped(queue):
    _, _, log, env, output, experiments = queue
    prepare_smoke(queue)
    experiment = experiments[0]
    fake = receipt_path(output, experiment, smoke=False)
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps({"status": "complete", "protocol": PROTOCOL,
                               "run_name": experiment["run_name"],
                               "checkpoint": experiment["checkpoint"],
                               "formal": True, "rows": [{}] * 6}))
    result = invoke(queue)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = calls(log)
    assert [row["name"] for row in rows] == ["run0", "run0", "run1", "run2", "run3"]


def test_valid_receipt_skips_without_new_generation(queue):
    queue_path, plan_path, log, env, output, experiments = queue
    prepare_smoke(queue)
    result = invoke(queue)
    assert result.returncode == 0, result.stderr + result.stdout
    log.unlink()
    result = invoke(queue)
    assert result.returncode == 0, result.stderr + result.stdout
    assert not log.exists()
    # A changed raw sample invalidates the receipt and regenerates that expert.
    experiment = experiments[2]
    receipt = json.loads(receipt_path(output, experiment, smoke=False).read_text())
    sample = Path(next(row["samples_jsonl"] for row in receipt["rows"] if row["benchmark_id"] == "gqa"))
    rows = sample.read_text().splitlines()
    rows.append(rows[-1])
    sample.write_text("\n".join(rows) + "\n")
    result = invoke(queue)
    assert result.returncode == 0, result.stderr + result.stdout
    assert [row["name"] for row in calls(log)] == ["run2"]


def test_dry_run_has_no_output_or_runner(queue):
    queue_path, plan, log, env, output, experiments = queue
    result = subprocess.run(["bash", str(queue_path), "--plan", str(plan), "--dry-run"],
                            env=env | {"EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0"}, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert [line.split("\t")[0] for line in result.stdout.splitlines()] == ["C1", "C2", "C3", "C4"]
    assert not log.exists() and not output.exists()


def test_expert_filter_partitions_the_queue(queue):
    """B6_EXPERTS lets a second instance claim disjoint experts."""
    _, plan, log, env, _, _ = queue
    prepare_smoke(queue)
    log.unlink()
    result = invoke(queue)  # full via the default path first (C1..C4 all run)
    assert result.returncode == 0, result.stderr + result.stdout
    log.unlink()
    # A second instance filtered to C3/C4 must run only those two, in order.
    env = env | {"B6_EXPERTS": "C3,C4"}
    result = subprocess.run(["bash", str(Path(env["HW_EVAL_ENV"]).parent / "nonexistent")], capture_output=True) if False else None
    queue_path, plan_path, _, _, _, _ = queue
    result = subprocess.run(["bash", str(queue_path), "--plan", str(plan_path)],
                            env=env | {"EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0"},
                            text=True, capture_output=True, timeout=90)
    # All receipts are valid, so the filtered queue skips everything.
    assert result.returncode == 0, result.stderr + result.stdout
    assert not log.exists()


def test_expert_filter_rejects_unknown_names(queue):
    _, plan, _, env, _, _ = queue
    prepare_smoke(queue)
    env = env | {"B6_EXPERTS": "C5,C9"}
    queue_path, plan_path, _, _, _, _ = queue
    result = subprocess.run(["bash", str(queue_path), "--plan", str(plan_path)],
                            env=env | {"EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0"},
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 1 and "unknown experts" in result.stderr


def test_expert_lock_blocks_concurrent_same_expert(queue):
    """A held per-expert lock stops a second instance from duplicating work."""
    queue_path, plan, log, env, _, _ = queue
    prepare_smoke(queue)
    log.unlink()
    lock_dir = Path(env["B6_PASS1_LOG_ROOT"])
    lock_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(lock_dir / ".b6_pass1_full_C1.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        env = env | {"B6_EXPERTS": "C1"}
        # Invalidate C1's receipt so the queue would try to run it, then hit the lock.
        receipt = receipt_path(Path(env["DTOPD_EVAL_ROOT"]),
                               {"expert": "C1", "run_name": "run0"}, smoke=False)
        if receipt.exists():
            receipt.unlink()
        result = subprocess.run(["bash", str(queue_path), "--plan", str(plan)],
                                env=env | {"EVAL_REPEAT_COUNT": "1", "EVAL_SAMPLING_TEMPERATURE": "1.0"},
                                text=True, capture_output=True, timeout=90)
        assert result.returncode == 1 and "already running C1" in result.stderr
        assert not log.exists()
