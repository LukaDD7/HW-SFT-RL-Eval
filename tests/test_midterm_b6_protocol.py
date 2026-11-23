import json
import os
from pathlib import Path
import subprocess
import sys

from dual_track_opd.eval.benchmark_suite import load_suite


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_B6 = {
    "gqa",
    "mmmu_pro",
    "remi",
    "scienceqa",
    "mindcube",
    "mmbench",
    "mathverse",
}


def test_canonical_b6_selects_exactly_the_midterm_seven() -> None:
    source = (ROOT / "scripts/eval/run_pinned_eval.sh").read_text(encoding="utf-8")
    b6_block = source.rsplit("  b6-mixed)", 1)[1].split("  project15)", 1)[0]
    assert b6_block.count("V2_BENCHMARKS=gqa,mmmu_pro") == 2
    assert "SFT_RL_BENCHMARKS=remi,scienceqa,mindcube" in b6_block
    assert "SFT_RL_JUDGE_BENCHMARKS=mmbench,mathverse" in b6_block
    assert "EVAL_RUN_NAME:-b6_midterm7_avg4" in source

    parallel = (ROOT / "scripts/eval/run_target_benchmarks_v2_parallel.sh").read_text(
        encoding="utf-8"
    )
    assert 'export V2_BENCHMARKS="gqa"' in parallel
    assert 'export V2_BENCHMARKS="mmmu_pro"' in parallel
    assert '--benchmarks "gqa,mmmu_pro"' in parallel

    selected = {
        "gqa",
        "mmmu_pro",
        *("remi,scienceqa,mindcube".split(",")),
        *("mmbench,mathverse".split(",")),
    }
    assert selected == EXPECTED_B6


def test_midterm_b6_task_versions_budgets_and_scoring() -> None:
    v1 = load_suite(ROOT / "configs/eval/project_vision_opd.yaml")
    v2 = load_suite(ROOT / "configs/eval/project_vision_opd_v2.yaml")
    expected = {
        "gqa": (v2, "gqa_v2", 8192, "normalized_exact_tag_or_direct"),
        "mmmu_pro": (v2, "mmmu_pro_standard_v2", 16384, "deterministic_mcq_tag_first"),
        "remi": (v1, None, 16384, "official_exact_or_relaxed_accuracy_sidecar"),
        "scienceqa": (v1, "scienceqa_img", 1024, "deterministic_mcq"),
        "mindcube": (v1, "mindcube_full", 1024, "deterministic_mcq_with_subcategories"),
        "mmbench": (v1, "mmbench_en_dev", 8192, "llm_choice_extractor"),
        "mathverse": (v1, "mathverse_testmini", 4096, "llm_as_judge"),
    }
    assert set(expected) == EXPECTED_B6
    for benchmark_id, (suite, task, budget, scoring) in expected.items():
        spec = suite.benchmarks[benchmark_id]
        assert (spec.task, spec.max_new_tokens, spec.scoring) == (task, budget, scoring)


def test_midterm_b6_does_not_change_independent_v2_or_project15() -> None:
    source = (ROOT / "scripts/eval/run_target_benchmarks_v2.sh").read_text(encoding="utf-8")
    assert 'V2_BENCHMARKS:-gqa,dynamath,viewspatial,mmmu_pro' in source

    suite = load_suite(ROOT / "configs/eval/project_vision_opd.yaml")
    assert len(suite.profiles["all"]) == 15
    assert suite.profiles["project15_complement"] == [
        "vqav2",
        "mv_math",
        "mmsi_bench",
        "blink",
        "mathvista",
        "mmvet",
    ]
    pinned = (ROOT / "scripts/eval/run_pinned_eval.sh").read_text(encoding="utf-8")
    complement = pinned.split("  project15-complement)", 1)[1]
    assert "SFT_RL_BENCHMARKS=vqav2,mv_math,mmsi_bench,blink" in complement
    assert "SFT_RL_JUDGE_BENCHMARKS=mathvista,mmvet" in complement


TRACE_HELPER = r'''
import json
import os
from pathlib import Path
import sys

role, args = sys.argv[1], sys.argv[2:]
record = {
    "role": role,
    "args": args,
    "env": {
        key: os.environ.get(key)
        for key in (
            "EVAL_RUN_NAME",
            "V2_BENCHMARKS",
            "SFT_RL_BENCHMARKS",
            "SFT_RL_JUDGE_BENCHMARKS",
            "DTOPD_EVAL_ROOT",
        )
    },
}
encoded = (json.dumps(record) + "\n").encode("utf-8")
fd = os.open(os.environ["MIDTERM_TEST_TRACE"], os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
try:
    os.write(fd, encoded)
finally:
    os.close(fd)

if (
    role == "bash"
    and args[0].endswith("run_target_benchmarks.sh")
    and "remi" in os.environ.get("SFT_RL_BENCHMARKS", "").split(",")
):
    replay = (
        Path(os.environ["DTOPD_EVAL_ROOT"])
        / (os.environ["EVAL_RUN_NAME"] + "_nojudge")
        / "replay"
    )
    replay.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        (replay / f"remi.repeat_{index}.jsonl").write_text("{}\n", encoding="utf-8")
if role == "python" and args[0].endswith("remi_reeval.py"):
    Path(args[args.index("--out") + 1]).write_text("{}\n", encoding="utf-8")
'''


def trace_runtime(tmp_path: Path) -> tuple[dict[str, str], Path]:
    environment = tmp_path / "env"
    executable = environment / "bin"
    executable.mkdir(parents=True)
    helper = tmp_path / "trace_helper.py"
    helper.write_text(TRACE_HELPER, encoding="utf-8")
    for name, role in (("python", "python"), ("bash", "bash")):
        path = executable / name
        path.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper}" {role} "$@"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "HW_EVAL_ENV": str(environment),
            "MIDTERM_TEST_TRACE": str(trace),
            "DTOPD_EVAL_ROOT": str(tmp_path / "outputs"),
            "PATH": f"{executable}:/usr/bin:/bin",
        }
    )
    return env, trace


def read_trace(trace: Path) -> list[dict]:
    return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]


def run_checked(args: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(args, env=env, text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_b6_executes_seven_task_routing(tmp_path: Path) -> None:
    env, trace = trace_runtime(tmp_path)
    env["EVAL_V2_PARALLEL"] = "0"
    run_checked(
        ["/bin/bash", str(ROOT / "scripts/eval/run_pinned_eval.sh"), "--protocol", "b6-mixed"],
        env=env,
    )
    records = read_trace(trace)
    bash_calls = [record for record in records if record["role"] == "bash"]
    assert [Path(record["args"][0]).name for record in bash_calls] == [
        "run_target_benchmarks_v2.sh",
        "run_target_benchmarks.sh",
    ]
    assert bash_calls[0]["env"]["V2_BENCHMARKS"] == "gqa,mmmu_pro"
    assert bash_calls[1]["env"]["SFT_RL_BENCHMARKS"] == "remi,scienceqa,mindcube"
    assert bash_calls[1]["env"]["SFT_RL_JUDGE_BENCHMARKS"] == "mmbench,mathverse"
    assert all(record["env"]["EVAL_RUN_NAME"] == "b6_midterm7_avg4" for record in bash_calls)

    remi = next(
        record
        for record in records
        if record["role"] == "python" and record["args"][0].endswith("remi_reeval.py")
    )
    assert remi["args"].count("--jsonl") == 4
    assert "--aggregate-mean" in remi["args"]
    assert remi["args"][remi["args"].index("--mode") + 1] == "exact"


def test_parallel_b6_passes_the_two_task_subset_to_merge(tmp_path: Path) -> None:
    env, trace = trace_runtime(tmp_path)
    env["EVAL_RUN_NAME"] = "midterm_parallel_trace"
    run_checked(
        ["/bin/bash", str(ROOT / "scripts/eval/run_target_benchmarks_v2_parallel.sh")],
        env=env,
    )
    records = read_trace(trace)
    shards = [record for record in records if record["role"] == "bash"]
    assert {
        record["env"]["EVAL_RUN_NAME"]: record["env"]["V2_BENCHMARKS"]
        for record in shards
    } == {
        "midterm_parallel_trace_v2a": "gqa",
        "midterm_parallel_trace_v2b": "mmmu_pro",
    }
    merge = next(
        record
        for record in records
        if record["role"] == "python" and "dual_track_opd.eval.merge_v2_shards" in record["args"]
    )
    assert merge["args"][merge["args"].index("--benchmarks") + 1] == "gqa,mmmu_pro"


def test_complement_executes_only_six_non_b6_tasks(tmp_path: Path) -> None:
    env, trace = trace_runtime(tmp_path)
    run_checked(
        [
            "/bin/bash",
            str(ROOT / "scripts/eval/run_pinned_eval.sh"),
            "--protocol",
            "project15-complement",
        ],
        env=env,
    )
    run = next(record for record in read_trace(trace) if record["role"] == "bash")
    assert run["env"]["SFT_RL_BENCHMARKS"] == "vqav2,mv_math,mmsi_bench,blink"
    assert run["env"]["SFT_RL_JUDGE_BENCHMARKS"] == "mathvista,mmvet"
