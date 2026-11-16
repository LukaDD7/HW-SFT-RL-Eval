"""Process-level scoped B6 lane tests with local stand-ins, never GPU servers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/eval/run_scoped_b6_expert.py"


@pytest.fixture
def scoped(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/eval").mkdir(parents=True)
    (repo / "src/dual_track_opd/eval").mkdir(parents=True)
    (repo / "src/dual_track_opd/__init__.py").write_text("")
    (repo / "src/dual_track_opd/eval/__init__.py").write_text("")
    (repo / "src/dual_track_opd/eval/__main__.py").write_text("")
    (repo / "scripts/eval/run_scoped_b6_expert.py").write_text(RUNNER.read_text())
    (repo / "configs/eval").mkdir(parents=True)
    (repo / "configs/eval/project_vision_opd_v2.yaml").write_text("fixture: true\n")
    helper = tmp_path / "child.py"
    helper.write_text(r'''import json, os, sys, time
from pathlib import Path
name = os.environ.get("V2_BENCHMARKS", "").split(",")[0] if os.environ.get("V2_BENCHMARKS") else (
    "mmbench" if os.environ.get("SFT_RL_JUDGE_BENCHMARKS") else "remi")
run = os.environ["EVAL_RUN_NAME"]
smoke = os.environ.get("EVAL_SMOKE") == "1"
suffix = "_smoke" if smoke else ""
if name in ("gqa", "viewspatial"):
    lane = "v2a" if name == "gqa" else "v2b"
    output = Path(os.environ["DTOPD_EVAL_ROOT"]) / f"{run}{suffix}"
else:
    lane = name
    output = Path(os.environ["DTOPD_EVAL_ROOT"]) / f"{run}{suffix}_{'judged' if lane == 'mmbench' else 'nojudge'}"
record = {"event": "start", "lane": lane, "run": run, "output_existed": output.exists(),
          "gpus": os.environ.get("EVAL_GPUS") if os.environ.get("V2_BENCHMARKS") else os.environ.get("SFT_RL_EVAL_GPUS"),
          "tp": os.environ.get("EVAL_TP") if os.environ.get("V2_BENCHMARKS") else os.environ.get("SFT_RL_EVAL_TP"),
          "port": os.environ.get("EVAL_PORT", os.environ.get("SFT_RL_EVAL_PORT")),
          "judge_gpus": os.environ.get("SFT_RL_JUDGE_GPUS"),
          "judge_tp": os.environ.get("SFT_RL_JUDGE_TP"),
          "judge_port": os.environ.get("SFT_RL_JUDGE_PORT"),
          "judge_max_len": os.environ.get("SFT_RL_JUDGE_MAX_LEN"),
          "benchmarks": os.environ.get("V2_BENCHMARKS", os.environ.get("SFT_RL_BENCHMARKS")),
          "judged": os.environ.get("SFT_RL_JUDGE_BENCHMARKS"),
          "limit": os.environ.get("EVAL_LIMIT"), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "config": os.environ.get("SFT_RL_EVAL_CONFIG")}
def log(value):
    with open(os.environ["B6_TEST_LOG"], "a") as handle:
        handle.write(json.dumps(value) + "\n")
log(record)
time.sleep(0.15)
if lane == os.environ.get("B6_TEST_FAIL"):
    sys.exit(19)
output.mkdir(parents=True, exist_ok=True)
(output / "run_manifest.json").write_text(json.dumps({"run_name": output.name}))
if lane == "remi":
    (output / "replay").mkdir(exist_ok=True)
    (output / "replay/remi.jsonl").write_text('{"sample_id": "fixture"}\n')
log({"event": "end", "lane": lane})
''')
    for module in ("merge_v2_shards", "score_remi"):
        (repo / f"src/dual_track_opd/eval/{module}.py").write_text(r'''import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
with open(os.environ["B6_TEST_LOG"], "a") as handle:
    handle.write(json.dumps({"event": "module", "module": os.path.basename(__file__).removesuffix(".py"),
                            "arguments": arguments}) + "\n")
if "--output-run" in arguments:
    output = Path(arguments[arguments.index("--output-run") + 1])
    for index in [i for i, value in enumerate(arguments) if value == "--shard-run"]:
        assert (Path(arguments[index + 1]) / "run_manifest.json").is_file()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps({"run_name": output.name}))
else:
    replay = Path(arguments[arguments.index("--replay") + 1])
    assert replay.is_file() and replay.stat().st_size
    output = Path(arguments[arguments.index("--out") + 1])
    output.write_text(json.dumps({"mode": "exact", "smoke": "--smoke" in arguments}))
''')
    scripts = {}
    for lane, filename in (("v2", "scripts/eval/run_target_benchmarks_v2.sh"),
                           ("v1", "scripts/sft_rl/run_sftrl_benchmarks.sh")):
        path = repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec "$B6_PYTHON" "$B6_TEST_HELPER"\n')
        scripts[lane] = path
    output = tmp_path / "outputs"
    log = tmp_path / "events.jsonl"
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1", "B6_PYTHON": sys.executable,
        "B6_TEST_HELPER": str(helper), "B6_TEST_LOG": str(log),
        "DTOPD_EVAL_ROOT": str(output), "B6_STAGE_LOG_ROOT": str(tmp_path / "stages"),
        "EVAL_CKPT": str(tmp_path / "checkpoint"), "EVAL_RUN_NAME": "scoped_c1",
        "EVAL_SERVED_MODEL": "SCOPED-C1", "SFT_RL_JUDGE_HF": str(tmp_path / "judge"),
        "B6_V2A_GPUS": "0,1", "B6_V2B_GPUS": "2,3", "B6_REMI_GPUS": "0,1",
        "B6_MMBENCH_GPUS": "2,3", "B6_JUDGE_GPUS": "4,5", "B6_MODEL_TP": "2",
        "B6_JUDGE_TP": "2", "B6_V2A_PORT": "8000", "B6_V2B_PORT": "8001",
        "B6_REMI_PORT": "8000", "B6_MMBENCH_PORT": "8002", "B6_JUDGE_PORT": "8004",
        "B6_MAX_MODEL_LEN": "65536", "B6_JUDGE_MAX_MODEL_LEN": "32768",
        "EVAL_LIMIT": "1", "CUDA_VISIBLE_DEVICES": "7", "EVAL_SMOKE": "1",
        "SFT_RL_SYSTEM_INSTRUCTION": "untrusted override",
    }
    return repo, output, log, env


def invoke(fixture, **extra):
    repo, _, _, env = fixture
    return subprocess.run([sys.executable, str(repo / "scripts/eval/run_scoped_b6_expert.py")],
                          cwd=repo, env=env | dict(extra), text=True, capture_output=True, timeout=30)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_two_tp2_lanes_then_v1_judge_and_resume(scoped):
    repo, output, log, env = scoped
    result = invoke(scoped, B6_SMOKE="1")
    assert result.returncode == 0, result.stderr + result.stdout
    rows = events(log)
    starts = [row for row in rows if row["event"] == "start"]
    assert {row["lane"] for row in starts[:2]} == {"v2a", "v2b"}
    assert {row["lane"] for row in starts[2:]} == {"remi", "mmbench"}
    by_lane = {row["lane"]: row for row in starts}
    assert set(by_lane) == {"v2a", "v2b", "remi", "mmbench"}
    assert all(row["tp"] == "2" and row["judge_tp"] == "2" for row in starts)
    assert {lane: row["gpus"] for lane, row in by_lane.items()} == {
        "v2a": "0,1", "v2b": "2,3", "remi": "0,1", "mmbench": "2,3",
    }
    assert {lane: row["port"] for lane, row in by_lane.items()} == {
        "v2a": "8000", "v2b": "8001", "remi": "8000", "mmbench": "8002",
    }
    assert all(row["judge_gpus"] == "4,5" and row["judge_port"] == "8004" for row in starts)
    assert all(row["judge_max_len"] == "32768" for row in starts)
    assert all(row["limit"] is None and row["cuda"] is None for row in starts)
    assert by_lane["v2a"]["benchmarks"] == "gqa,dynamath"
    assert by_lane["v2b"]["benchmarks"] == "viewspatial,mmmu_pro"
    assert by_lane["remi"]["benchmarks"] == "remi" and by_lane["remi"]["judged"] == ""
    assert by_lane["mmbench"]["benchmarks"] == "" and by_lane["mmbench"]["judged"] == "mmbench"
    merge = next(row for row in rows if row.get("module") == "merge_v2_shards")
    assert merge["arguments"][merge["arguments"].index("--output-run") + 1] == str(output / "scoped_c1_smoke")
    score = next(row for row in rows if row.get("module") == "score_remi")
    assert score["arguments"][-1] == "--smoke"
    events_before = len(rows)
    result = invoke(scoped, B6_SMOKE="1")
    assert result.returncode == 0, result.stderr + result.stdout
    resumed = [row for row in events(log)[events_before:] if row["event"] == "start"]
    assert len(resumed) == 4 and all(row["output_existed"] for row in resumed)


def test_v2_failure_waits_for_peer_and_never_starts_v1(scoped):
    _, _, log, _ = scoped
    result = invoke(scoped, B6_TEST_FAIL="v2b")
    assert result.returncode == 1
    assert "v2 failed: v2b" in result.stderr
    rows = events(log)
    assert {row["lane"] for row in rows if row["event"] == "start"} == {"v2a", "v2b"}
    assert any(row == {"event": "end", "lane": "v2a"} for row in rows)
    assert not any(row.get("module") for row in rows)


def test_v1_failure_preserves_outputs_and_skips_remi_scoring(scoped):
    _, output, log, _ = scoped
    result = invoke(scoped, B6_TEST_FAIL="mmbench")
    assert result.returncode == 1
    assert "v1 failed: mmbench" in result.stderr
    rows = events(log)
    assert {row["lane"] for row in rows if row["event"] == "start"} == {"v2a", "v2b", "remi", "mmbench"}
    assert any(row.get("module") == "merge_v2_shards" for row in rows)
    assert not any(row.get("module") == "score_remi" for row in rows)
    assert (output / "scoped_c1_nojudge/replay/remi.jsonl").is_file()


@pytest.mark.parametrize("alias", ["", "Vision-OPD-4B", "SCOPED-C5"])
def test_scoped_alias_is_fail_closed(scoped, alias):
    _, _, log, _ = scoped
    result = invoke(scoped, EVAL_SERVED_MODEL=alias)
    assert result.returncode == 1
    assert not log.exists()
