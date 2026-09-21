from pathlib import Path

import pytest

from dual_track_opd.eval.benchmark_suite import (
    build_commands,
    build_command,
    load_suite,
    select_benchmarks,
    validate_checkpoint_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval" / "project_vision_opd.yaml"


def test_contract_suite_has_five_benchmarks_per_category() -> None:
    suite = load_suite(CONFIG)
    counts: dict[str, int] = {}
    for spec in suite.benchmarks.values():
        counts[spec.category] = counts.get(spec.category, 0) + 1
    assert counts == {
        "vqa": 5,
        "math_science_reasoning": 5,
        "comprehensive": 5,
    }
    assert suite.raw["suite"]["contract_name_aliases"]["MatchVista"] == "MathVista"


def test_acceptance_core_has_two_items_per_category() -> None:
    suite = load_suite(CONFIG)
    selected = select_benchmarks(suite, "acceptance_core")
    categories = [spec.category for spec in selected]
    assert categories.count("vqa") == 2
    assert categories.count("math_science_reasoning") == 2
    assert categories.count("comprehensive") == 2


def test_target_profile_covers_requested_six_benchmarks() -> None:
    suite = load_suite(CONFIG)
    selected = {spec.benchmark_id for spec in select_benchmarks(suite, "target_benchmarks")}
    assert selected == {"gqa", "mmbench", "remi", "dynamath", "viewspatial", "mmmu_pro"}


def test_judge_is_deferred_or_predict_only() -> None:
    suite = load_suite(CONFIG)
    spec = suite.benchmarks["mathvista"]
    common = dict(
        suite=suite,
        run_dir=Path("/tmp/run"),
        python="python",
        inference_backend="openai",
        checkpoint="/checkpoint",
        api_base="http://127.0.0.1:8000/v1",
        limit=8,
    )
    assert build_command(spec, judge_policy="defer", **common) is None
    command = build_command(spec, judge_policy="predict", **common)
    assert command is not None
    assert "--predict_only" in command
    assert "mathvista_testmini" in command


def test_replay_requires_openai_endpoint_backend() -> None:
    suite = load_suite(CONFIG)
    spec = suite.benchmarks["mv_math"]
    common = dict(
        suite=suite,
        run_dir=Path("/tmp/run"),
        python="python",
        checkpoint="/checkpoint",
        api_base="http://127.0.0.1:8000/v1",
        limit=8,
        judge_policy="defer",
    )
    assert build_command(spec, inference_backend="vllm", **common) is None
    command = build_command(spec, inference_backend="openai", **common)
    assert command is not None
    assert "dual_track_opd.eval.run_vlm_eval" in command
    assert any(
        value.endswith("qwen3vl8b_MV_MATH_len65536_maxtok1024_raw.jsonl")
        for value in command
    )
    assert "--resume" in command


def test_lmms_command_records_generation_and_raw_outputs() -> None:
    suite = load_suite(CONFIG)
    spec = suite.benchmarks["viewspatial"]
    command = build_command(
        spec,
        suite=suite,
        run_dir=Path("/tmp/run"),
        python="python",
        inference_backend="vllm",
        checkpoint="/checkpoint",
        api_base="http://127.0.0.1:8000/v1",
        limit=4,
        judge_policy="defer",
    )
    assert command is not None
    assert "model=/checkpoint" in command[command.index("--model_args") + 1]
    assert command[command.index("--gen_kwargs") + 1] == "temperature=1.0,max_new_tokens=1024"
    assert command[command.index("--limit") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    output_path = Path(command[command.index("--output_path") + 1])
    cache_path = Path(command[command.index("--use_cache") + 1])
    assert output_path.parts[-2:] == ("viewspatial", "repeat_0")
    assert cache_path.parts[-1] == "repeat_0"
    assert "--log_samples" in command


def test_canonical_generation_budgets() -> None:
    suite = load_suite(CONFIG)
    budgets = {
        benchmark_id: spec.max_new_tokens
        for benchmark_id, spec in suite.benchmarks.items()
    }
    assert budgets["viewspatial"] == 1024
    assert budgets["mindcube"] == 1024
    assert budgets["scienceqa"] == 1024
    assert budgets["remi"] == 8192
    assert budgets["blink"] == 2048
    assert budgets["mmbench"] == 2048
    assert budgets["mmmu_pro"] == 4096


def test_avg4_builds_four_independent_repeat_commands() -> None:
    suite = load_suite(CONFIG)
    spec = suite.benchmarks["viewspatial"]
    commands = build_commands(
        spec,
        suite=suite,
        run_dir=Path("/tmp/run"),
        python="python",
        inference_backend="openai",
        checkpoint="/checkpoint",
        api_base="http://127.0.0.1:8000/v1",
        limit=4,
        judge_policy="defer",
    )
    assert len(commands) == 4
    seeds = [int(command[command.index("--seed") + 1]) for command in commands]
    assert seeds == [42, 43, 44, 45]
    outputs = [Path(command[command.index("--output_path") + 1]) for command in commands]
    assert [path.parts[-1] for path in outputs] == [f"repeat_{i}" for i in range(4)]
    assert [path.parts[-2] for path in outputs] == ["viewspatial"] * 4


def test_lmms_command_supports_format_pilot_overrides(monkeypatch) -> None:
    suite = load_suite(CONFIG)
    spec = suite.benchmarks["viewspatial"]
    monkeypatch.setenv("SFT_RL_MAX_NEW_TOKENS_OVERRIDE", "1024")
    monkeypatch.setenv("SFT_RL_SYSTEM_INSTRUCTION", "Answer with only the final choice letter.")
    command = build_command(
        spec,
        suite=suite,
        run_dir=Path("/tmp/run"),
        python="python",
        inference_backend="openai",
        checkpoint="/checkpoint",
        api_base="http://127.0.0.1:8000/v1",
        limit=4,
        judge_policy="defer",
    )
    assert command is not None
    assert command[command.index("--model") + 1] == "async_openai"
    assert command[command.index("--gen_kwargs") + 1] == "temperature=1.0,max_new_tokens=1024"
    model_args = command[command.index("--model_args") + 1]
    assert "system_prompt=Answer with only the final choice letter." in model_args
    assert "is_qwen3_vl=true" in model_args


def test_checkpoint_identity_guard_accepts_matching_ptdpo_path() -> None:
    checkpoint = validate_checkpoint_identity(
        "/models/qwen3vl_ptdpo_r4_step390",
        run_name="project15_ptdpo_r4_step390_offline_core_nojudge",
        served_model_name="Qwen3-VL-8B-PTDPO-R4",
    )
    assert checkpoint.endswith("qwen3vl_ptdpo_r4_step390")


def test_checkpoint_identity_guard_rejects_mislabeled_ptdpo_run() -> None:
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        validate_checkpoint_identity(
            "/models/Vision-OPD-Qwen3.5-4B/global_step_65",
            run_name="project15_ptdpo_r4_step390_offline_core_nojudge",
            served_model_name="Qwen3-VL-8B-PTDPO-R4",
        )


def test_checkpoint_identity_guard_rejects_resume_checkpoint_change() -> None:
    with pytest.raises(ValueError, match="resume checkpoint mismatch"):
        validate_checkpoint_identity(
            "/models/qwen3vl_ptdpo_r4_step390",
            run_name="project15_ptdpo_r4_step390_offline_core_nojudge",
            served_model_name="Qwen3-VL-8B-PTDPO-R4",
            previous_checkpoint="/models/Vision-OPD-Qwen3.5-4B/global_step_65",
        )
