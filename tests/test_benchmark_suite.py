from pathlib import Path

from dual_track_opd.eval.benchmark_suite import (
    build_command,
    load_suite,
    select_benchmarks,
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
    assert command[command.index("--gen_kwargs") + 1] == "temperature=0,max_new_tokens=256"
    assert command[command.index("--limit") + 1] == "4"
    assert "--log_samples" in command


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
    assert command[command.index("--gen_kwargs") + 1] == "temperature=0,max_new_tokens=1024"
    model_args = command[command.index("--model_args") + 1]
    assert "system_prompt=Answer with only the final choice letter." in model_args
    assert "is_qwen3_vl=true" in model_args
