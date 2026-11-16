"""Validate and report one or more complete single-generation B6 runs.

The three canonical output directories (merged v2, no-judge ReMI, and judged
MMBench) are checked together. Exit status alone is never accepted as a score.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .benchmark_suite import generation_protocol, load_suite
from .project_summary import _metric_values, _result_jsons
from .score_remi import REMI_LABELS, validate_remi_receipt

B6_V2 = ("gqa", "dynamath", "viewspatial", "mmmu_pro")
B6_ALL = (*B6_V2, "remi", "mmbench")
SHARD_KEYS = ("v2a", "v2b", "remi", "mmbench", "judge")
SCOPED_SHARD_ENDPOINTS = {
    "v2a": "http://127.0.0.1:8000/v1",
    "v2b": "http://127.0.0.1:8001/v1",
    "remi": "http://127.0.0.1:8000/v1",
    "mmbench": "http://127.0.0.1:8002/v1",
    "judge": "http://127.0.0.1:8004/v1",
}
PROTOCOL = "b6-mixed-single-sample-pass1-seed42-v1"
GENERATION = {"seed": 42, "repeat_count": 1, "sampling_temperature": 1.0}
# Verified against full canonical lmms result manifests, not inferred from a
# smoke run. The task registry can change only with a new pinned protocol.
FULL_SAMPLE_COUNTS = {
    "gqa": 12578, "dynamath": 5010, "viewspatial": 5712,
    "mmmu_pro": 1730, "mmbench": 4329, "remi": 2600,
}
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _load_truncation_audit():
    path = Path(__file__).parents[3] / "scripts/eval/audit_truncation.py"
    spec = importlib.util.spec_from_file_location("hw_eval_truncation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load truncation audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_file


audit_file = _load_truncation_audit()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def b6_suites() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "v1": load_suite(root / "configs/eval/project_vision_opd.yaml"),
        "v2": load_suite(root / "configs/eval/project_vision_opd_v2.yaml"),
    }


def b6_specs() -> dict[str, Any]:
    suites = b6_suites()
    return {**suites["v1"].benchmarks, **suites["v2"].benchmarks}


def validate_plan(plan: Mapping[str, Any], *, expected_experiments: int | None = None) -> dict[str, Any]:
    if (plan.get("protocol") != PROTOCOL or plan.get("think_mode") != "auto"
            or any(plan.get(key) != value for key, value in GENERATION.items())):
        raise ValueError(f"plan must use {PROTOCOL}, auto, seed=42, repeat=1, temperature=1.0")
    budgets = {key: spec.max_new_tokens for key, spec in b6_specs().items() if key in B6_ALL}
    if plan.get("generation_budgets") != budgets:
        raise ValueError("plan must explicitly preserve every canonical B6 generation budget")
    gpu_group = plan.get("gpu_group")
    if (not isinstance(gpu_group, list) or len(gpu_group) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in gpu_group) or gpu_group[0] == gpu_group[1]):
        raise ValueError("gpu_group must contain two distinct non-negative GPU indices")
    if plan.get("shard_endpoints") != SCOPED_SHARD_ENDPOINTS:
        raise ValueError("plan must pin the SCOPED B6 model and judge endpoint registry")
    for key in ("judge_model", "judge_checkpoint"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError(f"local MMBench {key} provenance is required")
    if not Path(plan["judge_checkpoint"]).is_absolute():
        raise ValueError("judge_checkpoint must be an absolute path")
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("plan requires a nonempty experiments list")
    if expected_experiments is not None and len(experiments) != expected_experiments:
        raise ValueError(f"this report requires exactly {expected_experiments} experiments")
    experts, run_names, checkpoints = set(), set(), set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("each experiment must be a JSON object")
        for key in ("expert", "run_name", "checkpoint"):
            value = experiment.get(key)
            if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) and key != "checkpoint":
                raise ValueError(f"each experiment requires a safe nonempty {key}")
            if not value:
                raise ValueError(f"each experiment requires {key}")
        if not Path(experiment["checkpoint"]).is_absolute():
            raise ValueError("experiment checkpoint must be an absolute path")
        checkpoint = str(Path(experiment["checkpoint"]).resolve())
        if (experiment["expert"] in experts or experiment["run_name"] in run_names
                or checkpoint in checkpoints):
            raise ValueError("expert/run names and canonical checkpoints must be unique")
        experts.add(experiment["expert"])
        run_names.add(experiment["run_name"])
        checkpoints.add(checkpoint)
    return dict(plan)


def load_plan(path: Path, *, expected_experiments: int | None = None) -> dict[str, Any]:
    return validate_plan(load_json(path), expected_experiments=expected_experiments)


def receipt_path(root: Path, experiment: Mapping[str, Any], *, smoke: bool,
                 report_name: str = "scoped_b6_pass1") -> Path:
    mode = "smoke" if smoke else "full"
    return root / f"{report_name}_{experiment['expert']}_{experiment['run_name']}_{mode}_receipt.json"


def sample_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object sample in {path}")
                rows.append(row)
    return rows


def _argument(command: list[str], flag: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"generation command must contain one {flag}")
    return command[indexes[0] + 1]


def _check_generation_command(command: Any, *, spec: Any, smoke: bool, served_model: str,
                              api_base: str) -> None:
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError(f"{spec.benchmark_id}: invalid generation command")
    if _argument(command, "--seed") != "42":
        raise ValueError(f"{spec.benchmark_id}: seed mismatch")
    if "--predict_only" in command:
        raise ValueError(f"{spec.benchmark_id}: predict-only output is not a score")
    if spec.runner == "lmms_eval":
        model_args = _argument(command, "--model_args")
        # The openai backend passes the served alias as its `model` field
        # (model=SCOPED-C1); it has no separate served_model_name key.
        if (f"served_model_name={served_model}" not in model_args
                and f"model={served_model}," not in model_args + ","):
            raise ValueError(f"{spec.benchmark_id}: served-model provenance is missing")
        if f"base_url={api_base}" not in model_args and f"api_base={api_base}" not in model_args:
            raise ValueError(f"{spec.benchmark_id}: model endpoint does not match the run manifest")
    elif (_argument(command, "--model") != served_model
            or _argument(command, "--api-base") != api_base):
        raise ValueError(f"{spec.benchmark_id}: replay model/endpoint provenance mismatch")
    limit = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--limit"]
    # lmms-eval records float limits (8.0) while the replay runner uses 8;
    # compare numerically so both spellings audit as the 8-row smoke.
    expected_limit = ["8"] if smoke else []
    if [str(float(value)) for value in limit] != [str(float(v)) for v in expected_limit]:
        raise ValueError(f"{spec.benchmark_id}: generation limit is not smoke=8/full")
    if spec.runner == "lmms_eval":
        raw = _argument(command, "--gen_kwargs").split(",")
        values = dict(part.split("=", 1) for part in raw)
        if (set(values) != {"temperature", "max_new_tokens"}
                or float(values["temperature"]) != 1.0
                or int(values["max_new_tokens"]) != spec.max_new_tokens):
            raise ValueError(f"{spec.benchmark_id}: lmms generation settings mismatch")
    else:
        if (float(_argument(command, "--temperature")) != 1.0
                or int(_argument(command, "--max-tokens")) != spec.max_new_tokens):
            raise ValueError(f"{spec.benchmark_id}: replay generation settings mismatch")
        source = Path(_argument(command, "--input-jsonl"))
        # The manifest records the replay source as a repo-relative path
        # (valid in the HWEval working tree), but the auditor may run from any
        # cwd. Resolve it against the eval repo root, then require it to be
        # the pinned label file.
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[3] / source
        if source.name != spec.source_file or not source.is_file():
            raise ValueError(f"{spec.benchmark_id}: missing canonical replay source")
        # Do not silently accept an unrelated label source.
        if source.resolve() != REMI_LABELS.resolve():
            raise ValueError(f"{spec.benchmark_id}: non-canonical replay source")


def check_manifest(manifest: dict[str, Any], *, checkpoint: Path, ids: tuple[str, ...],
                   specs: dict[str, Any], smoke: bool, run_name: str, served_expert: str,
                   api_base: str | Mapping[str, str], manifest_api_base: str | None = None) -> dict[str, Any]:
    expected_api_base = manifest_api_base if manifest_api_base is not None else api_base
    if not isinstance(expected_api_base, str):
        raise ValueError("a merged shard manifest requires an explicit manifest API base")
    expected = {
        "checkpoint_path": str(checkpoint.resolve()), "run_name": run_name,
        "repeat_count": 1, "seed": 42, "sampling_temperature": 1.0,
        "protocol": generation_protocol(1, 1.0), "returncode": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} mismatch: {manifest.get(key)!r} != {value!r}")
    if manifest.get("thinking_protocol", {}).get("mode") != "auto":
        raise ValueError("prompt protocol is not canonical auto")
    if manifest.get("inference_backend") != "openai" or manifest.get("api_base") != expected_api_base:
        raise ValueError("canonical local OpenAI inference provenance is missing")
    runs = manifest.get("runs", [])
    # A resumed run keeps its history: an earlier failed record for a benchmark
    # stays in the manifest next to the later completed retry. Require exactly
    # one completed record per expected benchmark instead of a total record
    # count; the raw-output audit below still validates every completed row.
    completed_counts: dict[str, int] = {}
    for row in runs:
        if isinstance(row, dict) and row.get("status") == "completed":
            benchmark_id = row.get("benchmark_id")
            completed_counts[benchmark_id] = completed_counts.get(benchmark_id, 0) + 1
    if (set(completed_counts) != set(ids)
            or any(completed_counts[benchmark_id] != 1 for benchmark_id in ids)):
        raise ValueError(
            "manifest does not hold exactly one completed run per benchmark: "
            f"expected={sorted(ids)}, completed={completed_counts}"
        )
    by_id = {row.get("benchmark_id"): row for row in runs
             if isinstance(row, dict) and row.get("status") == "completed"}
    for benchmark_id in ids:
        row, spec = by_id[benchmark_id], specs[benchmark_id]
        command_api_base = api_base["v2a" if benchmark_id in B6_V2[:2] else "v2b"] \
            if isinstance(api_base, Mapping) else api_base
        if (row.get("status") != "completed" or row.get("repeat_count") != 1
                or row.get("primary_metric") != spec.primary_metric):
            raise ValueError(f"{benchmark_id} is not a completed single generation")
        if row.get("returncode_repeat_0") != 0 or row.get("max_new_tokens") != spec.max_new_tokens:
            raise ValueError(f"{benchmark_id} exit status or generation budget is invalid")
        if manifest.get("generation_budgets", {}).get(benchmark_id) != spec.max_new_tokens:
            raise ValueError(f"{benchmark_id}: manifest budget registry mismatch")
        commands = row.get("command", [])
        if len(commands) != 1:
            raise ValueError(f"{benchmark_id} does not contain exactly one generation command")
        _check_generation_command(
            commands[0], spec=spec, smoke=smoke, served_model=f"SCOPED-{served_expert}",
            api_base=command_api_base)
    return by_id


def audit_lmms(run: Path, spec: Any, row: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    benchmark_id = spec.benchmark_id
    result_paths = _result_jsons(run / "lmms" / benchmark_id, 1)
    if len(result_paths) != 1:
        raise ValueError(f"{benchmark_id}: expected one lmms result JSON")
    result_path = result_paths[0]
    stem = result_path.name.removesuffix("_results.json")
    if stem == result_path.name:
        raise ValueError(f"{benchmark_id}: unrecognized timestamped result filename")
    # Select the sample file from the same lmms invocation, never an arbitrary
    # older retry in the same output directory.
    sample_path = result_path.with_name(f"{stem}_samples_{spec.task}.jsonl")
    if not sample_path.is_file():
        raise ValueError(f"{benchmark_id}: timestamp-matched samples JSONL is missing")
    result = load_json(result_path)
    config = result.get("config", {})
    if config.get("limit") != (8 if smoke else None) or config.get("predict_only"):
        raise ValueError(f"{benchmark_id}: result limit or judge stage is incorrect")
    generation = config.get("gen_kwargs", {})
    if generation != {"temperature": 1.0, "max_new_tokens": spec.max_new_tokens}:
        raise ValueError(f"{benchmark_id}: result generation settings do not match")
    values = _metric_values(result.get("groups", {}), spec.primary_metric)
    if not values:
        values = _metric_values(result.get("results", {}), spec.primary_metric)
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError(f"{benchmark_id}: one finite primary metric {spec.primary_metric} is required")
    rows = sample_rows(sample_path)
    counts = result.get("n-samples", {})
    if set(counts) != {spec.task}:
        raise ValueError(f"{benchmark_id}: ambiguous lmms sample counts")
    count = counts[spec.task]
    effective = count.get("effective") if isinstance(count, dict) else None
    original = count.get("original") if isinstance(count, dict) else None
    expected = 8 if smoke else FULL_SAMPLE_COUNTS[benchmark_id]
    # lmms-eval reports original=full-dataset-size and effective=rows actually
    # generated under --limit; full runs (no --limit) have original==effective.
    if effective != len(rows) or len(rows) != expected or (not smoke and original != expected):
        raise ValueError(f"{benchmark_id}: expected complete effective/samples={expected}")
    doc_ids = [item.get("doc_id", item.get("id")) for item in rows]
    if any(value is None or isinstance(value, (dict, list)) for value in doc_ids) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{benchmark_id}: missing or duplicate sample identities")
    for item in rows:
        token_counts = item.get("token_counts")
        if (not isinstance(token_counts, list) or len(token_counts) != 1
                or not isinstance(token_counts[0], dict)
                or not isinstance(token_counts[0].get("output_tokens"), int)
                or isinstance(token_counts[0]["output_tokens"], bool)
                or token_counts[0]["output_tokens"] < 0):
            raise ValueError(f"{benchmark_id}: missing output-token counts for truncation audit")
    truncation = audit_file(sample_path, spec.max_new_tokens)
    if truncation["samples"] != len(rows) or truncation["at_cap"] > len(rows):
        raise ValueError(f"{benchmark_id}: truncation audit is inconsistent")
    return {
        "benchmark_id": benchmark_id, "value": values[0],
        "primary_metric": spec.primary_metric, "samples": len(rows),
        "truncated": truncation["at_cap"], "truncation_percent": truncation["at_cap_percent"],
        "truncation_audit": truncation,
        "comparison_status": "format-limited" if truncation["at_cap"] else "scored",
        "result_json": str(result_path.resolve()), "samples_jsonl": str(sample_path.resolve()),
        "generation_command": row["command"][0], "scoring_command": row["command"][0],
    }


def audit_remi(run: Path, spec: Any, row: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    replay_path = run / "replay/remi.jsonl"
    if replay_path.is_symlink() or not replay_path.is_file():
        raise ValueError("ReMI replay must be a regular file")
    rows = sample_rows(replay_path)
    expected = 8 if smoke else 2600
    if len(rows) != expected:
        raise ValueError(f"ReMI replay must contain {expected} rows")
    expected_generation = {"max_new_tokens": spec.max_new_tokens, "temperature": 1.0, "seed": 42}
    sample_ids = [item.get("sample_id") for item in rows]
    if any(not isinstance(value, str) or not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("ReMI replay sample identities are missing or duplicated")
    for item in rows:
        if (item.get("generation_config") != expected_generation or item.get("error") != ""
                or item.get("finish_reason") not in ("stop", "length")):
            raise ValueError("ReMI replay generation settings or completion status are invalid")
    result_path = run / "remi_result.json"
    result = load_json(result_path)
    validate_remi_receipt(result, 1, smoke=smoke)
    scoring_path = run / "remi_scoring.json"
    scoring = load_json(scoring_path)
    if (not isinstance(scoring.get("formal"), bool) or scoring["formal"] != (not smoke)
            or scoring.get("row_counts") != [expected]
            or {key: value for key, value in scoring.items()
                if key not in ("formal", "scoring_command")} != result):
        raise ValueError("ReMI smoke/full scoring receipt does not match its exact result")
    command = row["command"][0]
    scoring_command = scoring.get("scoring_command")
    if (not isinstance(scoring_command, list)
            or _argument(scoring_command, "--mode") != "exact"
            or "--aggregate-mean" not in scoring_command
            or Path(_argument(scoring_command, "--jsonl")).resolve() != replay_path.resolve()
            or Path(_argument(scoring_command, "--label-jsonl")).resolve() != REMI_LABELS.resolve()):
        raise ValueError("ReMI exact scoring command does not match the audited replay/labels")
    if result.get("jsonls") != [str(replay_path.resolve())]:
        raise ValueError("ReMI exact result is not bound to the single replay")
    truncated = sum(item["finish_reason"] == "length" for item in rows)
    return {
        "benchmark_id": "remi", "value": result["mean"],
        "primary_metric": "exact_accuracy_full_denominator", "samples": len(rows),
        "truncated": truncated, "truncation_percent": 100.0 * truncated / len(rows),
        "comparison_status": "format-limited" if truncated else "scored",
        "result_json": str(result_path.resolve()), "samples_jsonl": str(replay_path.resolve()),
        "generation_command": command, "scoring_command": scoring_command,
        "scoring_receipt": str(scoring_path.resolve()),
        "replay_source": _argument(command, "--input-jsonl"),
    }


def _audit_judge(manifest: dict[str, Any], judge: Mapping[str, Any], api_base: str) -> dict[str, Any]:
    if manifest.get("judge_policy") != "score":
        raise ValueError("MMBench result was not judged")
    actual = manifest.get("judge_provenance")
    if not isinstance(actual, dict):
        raise ValueError("local MMBench judge provenance is missing")
    expected_checkpoint = str(Path(judge["judge_checkpoint"]).resolve())
    for key, expected in (("judge_model", judge["judge_model"]),
                          ("judge_checkpoint", expected_checkpoint),
                          ("served_model", judge["judge_model"])):
        if actual.get(key) != expected:
            raise ValueError(f"MMBench judge {key} does not match the local judge plan")
    if actual.get("judge_api_url") != api_base:
        raise ValueError("MMBench judge endpoint is not the pinned loopback endpoint")
    return actual


def audit_experiment(experiment: dict[str, Any], root: Path, *, smoke: bool,
                     specs: dict[str, Any], judge: Mapping[str, Any],
                     shard_endpoints: Mapping[str, str] | None = None) -> dict[str, Any]:
    endpoints = SCOPED_SHARD_ENDPOINTS if shard_endpoints is None else shard_endpoints
    if endpoints != SCOPED_SHARD_ENDPOINTS:
        raise ValueError("B6 audit requires the pinned per-shard endpoint registry")
    name = experiment["run_name"] + ("_smoke" if smoke else "")
    checkpoint = Path(experiment["checkpoint"])
    paths = {"v2": root / name, "remi": root / f"{name}_nojudge",
             "mmbench": root / f"{name}_judged"}
    manifests = {key: load_json(path / "run_manifest.json") for key, path in paths.items()}
    shard_paths = manifests["v2"].get("v2_parallel_shards", [])
    suffix = "_smoke" if smoke else ""
    expected_shards = [f"{experiment['run_name']}_v2a{suffix}",
                       f"{experiment['run_name']}_v2b{suffix}"]
    if len(shard_paths) != 2 or [Path(path).name for path in shard_paths] != expected_shards:
        raise ValueError("the audited B6 queue requires the two canonical v2 shards")
    for shard, shard_key, ids in zip(shard_paths, ("v2a", "v2b"), (B6_V2[:2], B6_V2[2:]), strict=True):
        shard_path = Path(shard)
        check_manifest(load_json(shard_path / "run_manifest.json"), checkpoint=checkpoint,
                       ids=ids, specs=specs, smoke=smoke, run_name=shard_path.name,
                       served_expert=experiment["expert"], api_base=endpoints[shard_key])
        if str(shard_path.parent.resolve()) != str(root.resolve()):
            raise ValueError("v2 shards must belong to this experiment output root")
    v2_rows = check_manifest(manifests["v2"], checkpoint=checkpoint, ids=B6_V2,
                             specs=specs, smoke=smoke, run_name=name,
                             served_expert=experiment["expert"], api_base=endpoints,
                             manifest_api_base=endpoints["v2a"])
    remi_rows = check_manifest(manifests["remi"], checkpoint=checkpoint, ids=("remi",),
                               specs=specs, smoke=smoke, run_name=f"{name}_nojudge",
                               served_expert=experiment["expert"], api_base=endpoints["remi"])
    mmbench_rows = check_manifest(manifests["mmbench"], checkpoint=checkpoint, ids=("mmbench",),
                                  specs=specs, smoke=smoke, run_name=f"{name}_judged",
                                  served_expert=experiment["expert"], api_base=endpoints["mmbench"])
    judge_provenance = _audit_judge(manifests["mmbench"], judge, endpoints["judge"])
    rows = [audit_lmms(paths["v2"], specs[key], v2_rows[key], smoke=smoke) for key in B6_V2]
    rows.append(audit_remi(paths["remi"], specs["remi"], remi_rows["remi"], smoke=smoke))
    rows.append(audit_lmms(paths["mmbench"], specs["mmbench"], mmbench_rows["mmbench"], smoke=smoke))
    receipt = {
        "expert": experiment["expert"], "run_name": name,
        "checkpoint": str(checkpoint.resolve()),
        "protocol": PROTOCOL, "formal": not smoke, "status": "complete", "rows": rows,
        "manifest_paths": [str((path / "run_manifest.json").resolve()) for path in paths.values()],
        "v2_shard_manifest_paths": [str((Path(path) / "run_manifest.json").resolve()) for path in shard_paths],
        "dataset_manifest_hashes": {key: manifest["dataset_manifest_hash"]
                                    for key, manifest in manifests.items()},
        "backend": manifests["v2"]["backend"], "think_mode": "auto",
        "generation": GENERATION.copy(), "generation_budgets": {
            key: specs[key].max_new_tokens for key in B6_ALL},
        "judge": judge_provenance,
        "judge_model": judge["judge_model"], "judge_checkpoint": str(Path(judge["judge_checkpoint"]).resolve()),
        "shard_endpoints": dict(endpoints),
    }
    if "provenance" in experiment:
        receipt["provenance"] = experiment["provenance"]
    return receipt


def build_report(plan: Mapping[str, Any], output_root: Path, *, smoke: bool,
                 report_name: str = "scoped_b6_pass1",
                 expected_experiments: int | None = None) -> dict[str, Any]:
    plan = validate_plan(plan, expected_experiments=expected_experiments)
    experiments = plan["experiments"][:1] if smoke else plan["experiments"]
    report: dict[str, Any] = {
        "protocol": PROTOCOL, "formal": not smoke, "status": "complete",
        "judge_model": plan["judge_model"],
        "judge_checkpoint": str(Path(plan["judge_checkpoint"]).resolve()),
        "shard_endpoints": dict(plan["shard_endpoints"]),
        "judge_backend": "local OpenAI-compatible vLLM via pinned lmms-eval MMBench judge",
        "experiments": [], "errors": [],
    }
    judge = {key: plan[key] for key in ("judge_model", "judge_checkpoint")}
    for experiment in experiments:
        try:
            report["experiments"].append(audit_experiment(
                experiment, output_root, smoke=smoke, specs=b6_specs(), judge=judge,
                shard_endpoints=plan["shard_endpoints"]))
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            report["errors"].append(f"{experiment['expert']}: {type(exc).__name__}: {exc}")
    if report["errors"] or len(report["experiments"]) != len(experiments):
        report["status"] = "incomplete"
    return report


def receipt_is_valid(path: Path, experiment: dict[str, Any], output_root: Path,
                     *, smoke: bool, judge: Mapping[str, Any]) -> bool:
    """A stale receipt never substitutes for a fresh audit of its raw outputs."""
    try:
        receipt = load_json(path)
        judge = {key: judge[key] for key in ("judge_model", "judge_checkpoint")}
        plan = {"protocol": PROTOCOL, "think_mode": "auto", **GENERATION,
                "generation_budgets": {key: spec.max_new_tokens for key, spec in b6_specs().items()
                                       if key in B6_ALL},
                "experiments": [experiment], **judge, "gpu_group": [0, 4],
                "shard_endpoints": dict(SCOPED_SHARD_ENDPOINTS)}
        report = build_report(plan, output_root, smoke=smoke,
                              report_name=path.stem, expected_experiments=1)
        return report["status"] == "complete" and receipt == report["experiments"][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# B6 single-sample pass@1", "",
        f"Status: **{report['status']}**; protocol: `{report['protocol']}`.", "",
        "| Expert | Benchmark | Score | Samples | At cap | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for expert in report["experiments"]:
        for row in expert.get("rows", []):
            lines.append(f"| {expert['expert']} | {row['benchmark_id']} | {row['value']:.6f} | "
                         f"{row['samples']} | {row['truncated']} | {row['comparison_status']} |")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-name", default="scoped_b6_pass1")
    parser.add_argument("--experiment", help="audit only this expert, without weakening plan validation")
    parser.add_argument("--receipt-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    expected = 1 if args.experiment else 4
    if args.experiment:
        matching = [row for row in plan["experiments"] if row["expert"] == args.experiment]
        if len(matching) != 1 or args.smoke and matching[0] != plan["experiments"][0]:
            raise SystemExit("--experiment must be unique; smoke may only audit the first experiment")
        plan["experiments"] = matching
    report = build_report(plan, args.output_root, smoke=args.smoke,
                          report_name=args.report_name, expected_experiments=expected)
    if report["status"] != "complete":
        # A failed audit still writes the report (its errors explain what is
        # missing) but must never write receipts: build_report returns only
        # the successful subset of experiments, so zip(strict=True) with the
        # full audited list would crash instead of reporting.
        output = args.output_root / f"{args.report_name}_report.json"
        if not args.receipt_only:
            _write_json(output, report)
            write_report(report, output.with_suffix(".md"))
        print(json.dumps({"status": report["status"], "report": None if args.receipt_only else str(output),
                          "experiments": len(report["experiments"]), "errors": report["errors"]}))
        raise SystemExit(1)
    receipts = []
    audited = plan["experiments"][:1] if args.smoke else plan["experiments"]
    for experiment, receipt in zip(audited, report["experiments"], strict=True):
        path = receipt_path(args.output_root, experiment, smoke=args.smoke, report_name=args.report_name)
        _write_json(path, receipt)
        receipts.append(str(path))
    report["receipts"] = receipts
    output = args.output_root / f"{args.report_name}_report.json"
    if not args.receipt_only:
        _write_json(output, report)
        write_report(report, output.with_suffix(".md"))
    print(json.dumps({"status": report["status"], "report": None if args.receipt_only else str(output),
                      "experiments": len(report["experiments"]), "errors": report["errors"]}))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
