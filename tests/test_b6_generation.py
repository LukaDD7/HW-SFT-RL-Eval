import json
from pathlib import Path
import subprocess
import sys

import pytest

from dual_track_opd.eval.benchmark_suite import (
    build_commands,
    generation_protocol,
    load_suite,
    validate_resume_generation,
)
from dual_track_opd.eval.score_remi import score_remi_exact, validate_remi_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_temperature_override_distinguishes_sampled_and_greedy(monkeypatch):
    monkeypatch.setenv("EVAL_REPEAT_COUNT", "1")
    monkeypatch.setenv("EVAL_SEED", "42")
    for temperature, label in (("1.0", "single_sample_pass1"), ("0", "greedy_pass1")):
        monkeypatch.setenv("EVAL_SAMPLING_TEMPERATURE", temperature)
        suite = load_suite(ROOT / "configs/eval/project_vision_opd.yaml")
        commands = build_commands(
            suite.benchmarks["remi"], suite=suite, run_dir=Path("/tmp/run"), python="python",
            inference_backend="openai", checkpoint="/checkpoint", api_base="http://unused/v1",
            limit=8, judge_policy="defer",
        )
        assert len(commands) == 1
        command = commands[0]
        assert float(command[command.index("--temperature") + 1]) == float(temperature)
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--max-tokens") + 1] == "16384"
        assert command[command.index("--output-jsonl") + 1].endswith("remi.jsonl")
        assert generation_protocol(1, float(temperature)) == label
    assert generation_protocol(4, 1.0) == "avg@4"


@pytest.mark.parametrize("key,previous", [
    ("seed", 43), ("repeat_count", 4), ("sampling_temperature", 0.0),
    ("protocol", "greedy_pass1"),
])
def test_resume_rejects_generation_protocol_mismatch(key, previous, monkeypatch):
    monkeypatch.setenv("EVAL_REPEAT_COUNT", "1")
    monkeypatch.setenv("EVAL_SAMPLING_TEMPERATURE", "1.0")
    suite = load_suite(ROOT / "configs/eval/project_vision_opd.yaml")
    manifest = {"seed": 42, "repeat_count": 1, "sampling_temperature": 1.0,
                "protocol": "single_sample_pass1"}
    validate_resume_generation(manifest, suite)
    manifest[key] = previous
    with pytest.raises(ValueError, match="resume generation protocol mismatch"):
        validate_resume_generation(manifest, suite)


@pytest.mark.parametrize("key", ["seed", "repeat_count", "sampling_temperature"])
def test_resume_rejects_missing_generation_protocol(key, monkeypatch):
    monkeypatch.setenv("EVAL_REPEAT_COUNT", "1")
    monkeypatch.setenv("EVAL_SAMPLING_TEMPERATURE", "1.0")
    suite = load_suite(ROOT / "configs/eval/project_vision_opd.yaml")
    manifest = {"seed": 42, "repeat_count": 1, "sampling_temperature": 1.0}
    manifest.pop(key)
    with pytest.raises(ValueError, match="missing prior"):
        validate_resume_generation(manifest, suite)


def receipt(smoke):
    return {"mode": "exact", "metric": "exact_accuracy_full_denominator",
            "aggregation": "mean", "repeat_count": 1, "values": [0.5], "mean": 0.5,
            "row_counts": [8 if smoke else 2600]}


@pytest.mark.parametrize("smoke", [False, True])
def test_remi_single_exact_scorer_uses_canonical_cli_and_receipt(tmp_path, monkeypatch, smoke):
    replay = tmp_path / "remi.jsonl"
    replay.write_text("{}\n")
    result = receipt(smoke)
    command_seen = []
    def run(command, **kwargs):
        command_seen.append(command)
        Path(command[command.index("--out") + 1]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    audit = score_remi_exact([replay], output=tmp_path / "remi_result.json", smoke=smoke)
    command = command_seen[0]
    assert Path(command[1]).resolve() == ROOT / "scripts/sft_rl/remi_reeval.py"
    assert Path(command[command.index("--label-jsonl") + 1]).name == "qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl"
    assert command[command.index("--jsonl") + 1] == str(replay.resolve())
    assert "--aggregate-mean" in command
    assert (tmp_path / "remi_scoring.json").is_file()
    assert audit["formal"] is not smoke


def test_remi_avg4_output_name_unchanged(tmp_path, monkeypatch):
    replays = []
    for index in range(4):
        path = tmp_path / f"remi.repeat_{index}.jsonl"
        path.write_text("{}\n")
        replays.append(path)
    result = receipt(False) | {"repeat_count": 4, "values": [0.5] * 4, "row_counts": [2600] * 4}
    def run(command, **kwargs):
        assert command[command.index("--out") + 1].endswith("remi_avg4.json")
        Path(command[command.index("--out") + 1]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    score_remi_exact(replays, output=tmp_path / "remi_result.json", smoke=False)
    assert (tmp_path / "remi_avg4.json").is_file()


@pytest.mark.parametrize("mutation", ["partial", "bad-repeat", "nan-mean", "wrong-metric", "scorer-zero-no-output"])
def test_remi_receipt_fails_closed(tmp_path, monkeypatch, mutation):
    result = receipt(False)
    if mutation == "partial":
        result["row_counts"] = [8]
    elif mutation == "bad-repeat":
        result["repeat_count"] = 4
    elif mutation == "nan-mean":
        result["mean"] = float("nan")
    elif mutation == "wrong-metric":
        result["metric"] = "normalized_exact_diagnostic"
    if mutation == "scorer-zero-no-output":
        with pytest.raises(RuntimeError, match="without a result"):
            def run(command, **kwargs):
                return subprocess.CompletedProcess(command, 0)
            monkeypatch.setattr(subprocess, "run", run)
            replay = tmp_path / "remi.jsonl"
            replay.write_text("{}\n")
            score_remi_exact([replay], output=tmp_path / "remi_result.json", smoke=False)
    else:
        with pytest.raises(ValueError):
            validate_remi_receipt(result, 1, smoke=False)
