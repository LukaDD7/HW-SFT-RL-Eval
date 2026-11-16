"""Strict B6 fixtures: metric, sample completeness, judge and truncation gates."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from dual_track_opd.eval.b6_pass1_report import (
    B6_ALL, B6_V2, FULL_SAMPLE_COUNTS, GENERATION, PROTOCOL, SCOPED_SHARD_ENDPOINTS,
    audit_experiment, b6_specs, build_report, load_plan, receipt_is_valid, receipt_path,
)
from dual_track_opd.eval.score_remi import REMI_LABELS


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {key: b6_specs()[key].max_new_tokens for key in B6_ALL}
METRICS = {"gqa": "exact_match", "dynamath": "dynamath_average", "viewspatial": "overall_accuracy",
           "mmmu_pro": "mmmu_acc", "mmbench": "gpt_eval_score"}
JUDGE = {"judge_model": "local-judge", "judge_checkpoint": "/external/judge"}


@pytest.fixture(autouse=True)
def canonical_environment(monkeypatch):
    # A stale interactive endpoint must not affect offline raw-output auditing.
    monkeypatch.setenv("EVAL_PORT", "65534")
    monkeypatch.setenv("EVAL_JUDGE_PORT", "65533")
    monkeypatch.setenv("VISION_OPD_API_BASE", "http://127.0.0.1:65534/v1")
    monkeypatch.setenv("VISION_OPD_SERVED_MODEL", "SCOPED-C4")


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def make_plan(experiments):
    return {"protocol": PROTOCOL, "think_mode": "auto", **GENERATION,
            "generation_budgets": dict(BUDGETS), "gpu_group": [0, 4],
            "shard_endpoints": dict(SCOPED_SHARD_ENDPOINTS),
            "experiments": experiments, **JUDGE}


def make_manifest(path: Path, checkpoint: Path, ids, *, judge="defer", smoke=False,
                  served_expert="C1", judge_checkpoint: str = JUDGE["judge_checkpoint"],
                  shard="v2a"):
    api_base = SCOPED_SHARD_ENDPOINTS[shard]
    runs = []
    for benchmark_id in ids:
        spec = b6_specs()[benchmark_id]
        served_model = f"SCOPED-{served_expert}"
        command = ["python", "-m", "dual_track_opd.eval.lmms_thinking", "--seed", "42",
                   "--model_args", f"served_model_name={served_model},api_base={api_base}",
                   "--gen_kwargs", f"temperature=1.0,max_new_tokens={BUDGETS[benchmark_id]}"]
        if benchmark_id == "remi":
            command = ["python", "--seed", "42", "--temperature", "1.0",
                       "--max-tokens", "16384", "--input-jsonl", str(REMI_LABELS),
                       "--model", served_model, "--api-base", api_base]
        if smoke:
            command.extend(["--limit", "8"])
        runs.append({"benchmark_id": benchmark_id, "status": "completed", "repeat_count": 1,
                     "primary_metric": spec.primary_metric,
                     "max_new_tokens": BUDGETS[benchmark_id], "returncode_repeat_0": 0,
                     "command": [command]})
    value = {
        "checkpoint_path": str(checkpoint.resolve()), "run_name": path.name,
        "thinking_protocol": {"mode": "auto"}, "inference_backend": "openai",
        "api_base": api_base, "generation_budgets": dict(BUDGETS),
        "dataset_manifest_hash": "fixture-dataset-registry", **GENERATION,
        "protocol": "single_sample_pass1", "returncode": 0, "judge_policy": judge,
        "backend": {"commit": "88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"}, "runs": runs,
    }
    if judge == "score":
        value["judge_provenance"] = {"judge_model": JUDGE["judge_model"],
                                    "judge_checkpoint": str(Path(judge_checkpoint).resolve()),
                                    "served_model": JUDGE["judge_model"],
                                    "judge_api_url": SCOPED_SHARD_ENDPOINTS["judge"]}
    return dump(path / "run_manifest.json", value)


def make_lmms(run: Path, benchmark_id: str, *, smoke=False, predict_only=False):
    directory = run / "lmms" / benchmark_id / "model"
    spec = b6_specs()[benchmark_id]
    count = 8 if smoke else FULL_SAMPLE_COUNTS[benchmark_id]
    result = dump(directory / "20261111_000000_results.json", {
        "config": {"limit": 8 if smoke else None, "predict_only": predict_only,
                   "gen_kwargs": {"temperature": 1.0, "max_new_tokens": BUDGETS[benchmark_id]}},
        "results": {benchmark_id: {METRICS[benchmark_id] + ",none": 0.5}},
        "n-samples": {spec.task: {"original": count, "effective": count}},
    })
    samples = directory / f"20261111_000000_samples_{spec.task}.jsonl"
    with samples.open("w") as handle:
        for index in range(count):
            tokens = BUDGETS[benchmark_id] if index == 0 else 20
            handle.write(json.dumps({"doc_id": index, "token_counts": [{"output_tokens": tokens}]}) + "\n")
    return result


def make_experiment(root: Path, name="run", *, smoke=False, expert="C1",
                    checkpoint: Path | None = None,
                    judge_checkpoint: str = JUDGE["judge_checkpoint"]):
    checkpoint = checkpoint or root / f"checkpoint-{name}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    run_name = name + suffix
    v2, remi, mmbench = root / run_name, root / f"{run_name}_nojudge", root / f"{run_name}_judged"
    for shard, shard_key, ids in ((root / f"{name}_v2a{suffix}", "v2a", B6_V2[:2]),
                                  (root / f"{name}_v2b{suffix}", "v2b", B6_V2[2:])):
        make_manifest(shard, checkpoint, ids, smoke=smoke, served_expert=expert, shard=shard_key)
    manifest = make_manifest(v2, checkpoint, B6_V2, smoke=smoke, served_expert=expert)
    value = json.loads(manifest.read_text())
    value["v2_parallel_shards"] = [str(root / f"{name}_v2a{suffix}"), str(root / f"{name}_v2b{suffix}")]
    value["runs"] = [json.loads((root / f"{name}_{shard}{suffix}/run_manifest.json").read_text())["runs"]
                     for shard in ("v2a", "v2b")]
    value["runs"] = [row for shard_rows in value["runs"] for row in shard_rows]
    manifest.write_text(json.dumps(value))
    make_manifest(remi, checkpoint, ("remi",), smoke=smoke, served_expert=expert, shard="remi")
    make_manifest(mmbench, checkpoint, ("mmbench",), judge="score", smoke=smoke,
                  served_expert=expert, judge_checkpoint=judge_checkpoint, shard="mmbench")
    for key in B6_V2:
        make_lmms(v2, key, smoke=smoke)
    make_lmms(mmbench, "mmbench", smoke=smoke)
    count = 8 if smoke else 2600
    rows = [{"sample_id": str(index), "prediction": "A", "finish_reason": "stop", "error": "",
             "generation_config": {"max_new_tokens": 16384, "temperature": 1.0, "seed": 42}}
            for index in range(count)]
    (remi / "replay").mkdir(parents=True, exist_ok=True)
    (remi / "replay/remi.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = {"mode": "exact", "metric": "exact_accuracy_full_denominator", "aggregation": "mean",
              "repeat_count": 1, "values": [0.5], "mean": 0.5, "row_counts": [count],
              "jsonls": [str((remi / "replay/remi.jsonl").resolve())]}
    dump(remi / "remi_result.json", result)
    scoring_command = ["python", "remi_reeval.py", "--mode", "exact", "--jsonl",
                       str((remi / "replay/remi.jsonl").resolve()), "--label-jsonl",
                       str(REMI_LABELS.resolve()), "--aggregate-mean"]
    dump(remi / "remi_scoring.json", result | {"formal": not smoke, "scoring_command": scoring_command})
    return {"expert": expert, "run_name": name, "checkpoint": str(checkpoint.resolve())}


def test_complete_experiment_requires_and_reports_six_tasks(tmp_path):
    experiment = make_experiment(tmp_path)
    receipt = audit_experiment(experiment, tmp_path, smoke=False, specs=b6_specs(), judge=JUDGE)
    assert [row["benchmark_id"] for row in receipt["rows"]] == list(B6_ALL)
    assert receipt["rows"][0]["truncated"] == 1
    assert receipt["rows"][0]["comparison_status"] == "format-limited"
    assert receipt["rows"][4]["samples"] == 2600
    assert receipt["rows"][4]["replay_source"] == str(REMI_LABELS)
    assert all("scoring_command" in row for row in receipt["rows"])
    path = receipt_path(tmp_path, experiment, smoke=False)
    assert receipt_is_valid(path, experiment, tmp_path, smoke=False, judge=JUDGE) is False
    dump(path, receipt)
    assert receipt_is_valid(path, experiment, tmp_path, smoke=False, judge=JUDGE)
    sample = Path(receipt["rows"][0]["samples_jsonl"])
    rows = sample.read_text().splitlines()
    rows.append(rows[-1])
    sample.write_text("\n".join(rows) + "\n")
    assert not receipt_is_valid(path, experiment, tmp_path, smoke=False, judge=JUDGE)


def test_full_report_is_exactly_four_by_six(tmp_path):
    experiments = []
    for index in range(4):
        experiments.append(make_experiment(tmp_path, name=f"run{index}", expert=f"C{index + 1}"))
    report = build_report(make_plan(experiments), tmp_path, smoke=False)
    assert report["status"] == "complete", report["errors"]
    assert len(report["experiments"]) == 4
    assert all(len(expert["rows"]) == 6 for expert in report["experiments"])


@pytest.mark.parametrize("mutation", ["missing-result", "bad-metric", "partial-remi", "partial-lmms",
                                     "missing-judge", "wrong-judge", "predict-only-judge", "duplicate-doc",
                                     "wrong-generation", "remi-error", "wrong-shard", "missing-truncation",
                                     "wrong-v2b-endpoint", "wrong-mmbench-endpoint", "wrong-judge-endpoint"])
def test_audit_rejects_invalid_or_partial_outputs(tmp_path, mutation):
    experiment = make_experiment(tmp_path)
    run, remi, judged = tmp_path / "run", tmp_path / "run_nojudge", tmp_path / "run_judged"
    if mutation == "missing-result":
        next((run / "lmms/gqa").rglob("*_results.json")).unlink()
    elif mutation == "bad-metric":
        path = next((run / "lmms/gqa").rglob("*_results.json"))
        value = json.loads(path.read_text())
        value["results"]["gqa"] = {"other_metric": 0.5}
        path.write_text(json.dumps(value))
    elif mutation == "partial-remi":
        path = remi / "replay/remi.jsonl"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:2599]))
    elif mutation == "partial-lmms":
        path = next((run / "lmms/gqa").rglob("*samples*.jsonl"))
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    elif mutation == "missing-judge":
        path = judged / "run_manifest.json"
        value = json.loads(path.read_text())
        del value["judge_provenance"]
        path.write_text(json.dumps(value))
    elif mutation == "wrong-judge":
        path = judged / "run_manifest.json"
        value = json.loads(path.read_text())
        value["judge_provenance"]["judge_checkpoint"] = "/different/judge"
        path.write_text(json.dumps(value))
    elif mutation == "predict-only-judge":
        path = next((judged / "lmms/mmbench").rglob("*_results.json"))
        value = json.loads(path.read_text())
        value["config"]["predict_only"] = True
        path.write_text(json.dumps(value))
    elif mutation == "duplicate-doc":
        path = next((run / "lmms/gqa").rglob("*samples*.jsonl"))
        rows = path.read_text().splitlines()
        rows[1] = rows[0]
        path.write_text("\n".join(rows) + "\n")
    elif mutation == "wrong-generation":
        path = next((run / "lmms/gqa").rglob("*_results.json"))
        value = json.loads(path.read_text())
        value["config"]["gen_kwargs"]["temperature"] = 0
        path.write_text(json.dumps(value))
    elif mutation == "remi-error":
        path = remi / "replay/remi.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["error"] = "server failed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif mutation == "wrong-shard":
        value = json.loads((run / "run_manifest.json").read_text())
        value["v2_parallel_shards"][0] = str(tmp_path / "run_v2b")
        (run / "run_manifest.json").write_text(json.dumps(value))
    elif mutation == "wrong-v2b-endpoint":
        path = tmp_path / "run_v2b/run_manifest.json"
        value = json.loads(path.read_text())
        value["api_base"] = SCOPED_SHARD_ENDPOINTS["v2a"]
        value["runs"][0]["command"][0] = [part.replace("8001", "8000")
                                         for part in value["runs"][0]["command"][0]]
        path.write_text(json.dumps(value))
    elif mutation == "wrong-mmbench-endpoint":
        path = judged / "run_manifest.json"
        value = json.loads(path.read_text())
        value["api_base"] = SCOPED_SHARD_ENDPOINTS["v2a"]
        path.write_text(json.dumps(value))
    elif mutation == "wrong-judge-endpoint":
        path = judged / "run_manifest.json"
        value = json.loads(path.read_text())
        value["judge_provenance"]["judge_api_url"] = SCOPED_SHARD_ENDPOINTS["mmbench"]
        path.write_text(json.dumps(value))
    else:
        path = next((run / "lmms/gqa").rglob("*samples*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["token_counts"] = []
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((OSError, ValueError, KeyError)):
        audit_experiment(experiment, tmp_path, smoke=False, specs=b6_specs(), judge=JUDGE)


def test_retry_samples_do_not_confuse_selected_result(tmp_path):
    experiment = make_experiment(tmp_path)
    path = next((tmp_path / "run/lmms/gqa").rglob("*samples*.jsonl"))
    path.with_name("20250101_000000_samples_gqa_v2.jsonl").write_text('{"doc_id": 0}\n')
    receipt = audit_experiment(experiment, tmp_path, smoke=False, specs=b6_specs(), judge=JUDGE)
    assert receipt["rows"][0]["samples"] == FULL_SAMPLE_COUNTS["gqa"]


def test_plan_identity_and_budget_guards(tmp_path):
    plan_path = tmp_path / "plan.json"
    base = make_plan([{"expert": "C1", "run_name": "one", "checkpoint": "/model1"},
                      {"expert": "C2", "run_name": "two", "checkpoint": "/model2"}])
    dump(plan_path, base)
    load_plan(plan_path)
    base["experiments"][1]["run_name"] = "one"
    dump(plan_path, base)
    with pytest.raises(ValueError, match="unique"):
        load_plan(plan_path)
    base["experiments"][1]["run_name"] = "two"
    base["experiments"][1]["checkpoint"] = "/model1"
    dump(plan_path, base)
    with pytest.raises(ValueError, match="unique"):
        load_plan(plan_path)
    base["experiments"][1]["checkpoint"] = "/model2"
    base["generation_budgets"]["gqa"] = 4096
    dump(plan_path, base)
    with pytest.raises(ValueError, match="budget"):
        load_plan(plan_path)


def test_plan_rejects_missing_or_wrong_shard_endpoint(tmp_path):
    plan = make_plan([{"expert": "C1", "run_name": "one", "checkpoint": "/model1"}])
    for endpoint in (None, "http://127.0.0.1:8000/v1"):
        plan["shard_endpoints"] = dict(SCOPED_SHARD_ENDPOINTS)
        if endpoint is None:
            del plan["shard_endpoints"]
        else:
            plan["shard_endpoints"]["v2b"] = endpoint
        dump(tmp_path / "plan.json", plan)
        with pytest.raises(ValueError, match="endpoint registry"):
            load_plan(tmp_path / "plan.json")


def test_smoke_is_not_promoted_to_full(tmp_path):
    experiment = make_experiment(tmp_path, smoke=True)
    with pytest.raises((ValueError, OSError)):
        audit_experiment(experiment, tmp_path, smoke=False, specs=b6_specs(), judge=JUDGE)
    receipt = audit_experiment(experiment, tmp_path, smoke=True, specs=b6_specs(), judge=JUDGE)
    assert all(row["samples"] == 8 for row in receipt["rows"])
