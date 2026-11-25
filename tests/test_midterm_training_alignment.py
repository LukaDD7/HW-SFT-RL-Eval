import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIDTERM = ROOT / "docs/midterm_report_202611"
HANDOFF = MIDTERM / "rlsd_rlrt_reproduction_20261124"
SOURCE_RECIPE = HANDOFF / "configs/experiment_recipe.json"
ALIGNED_RECIPE = ROOT / "configs/experiment/midterm_grpo_aligned_20261125.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rl_volume_and_step_units_are_matched() -> None:
    document = (MIDTERM / "training_alignment.md").read_text(encoding="utf-8")
    rows = {
        "T1 / GRPO": (30000, 400, 256, 256, 8),
        "T2 / Base hybrid": (30000, 400, 256, 256, 8),
        "T3 / post-MOPD hybrid": (15000, 400, 256, 256, 8),
        "T5 / post-SFT GRPO": (15000, 400, 256, 256, 8),
    }
    for arm, (records, steps, batch, mini_batch, rollouts) in rows.items():
        prompts = steps * batch
        trajectories = prompts * rollouts
        assert (prompts, trajectories) == (102400, 819200)
        assert steps * (batch // mini_batch) == 400
        expected = (
            f"| {arm} | {records:,} | {steps} | {batch} | {mini_batch} | "
            f"{rollouts} | {prompts:,} | {trajectories:,} | at most 400 |"
        )
        assert expected in document
    assert "global\nrollout steps and optimizer updates" in document
    assert "800/15000" not in document


def test_pr_recipe_retains_original_hybrid_cohort() -> None:
    recipe = load_json(SOURCE_RECIPE)
    common = recipe["common"]
    assert common["trainer.max_steps"] == 400
    assert common["data.rollout_batch_size"] == 256
    assert common["data.mini_rollout_batch_size"] == 256
    assert common["worker.actor.global_batch_size"] == 256
    assert common["worker.rollout.n"] == 8
    assert common["worker.actor.ppo_epochs"] == 1
    assert common["worker.actor.optim.lr"] == 1e-6
    assert common["worker.rollout.seed"] == 1
    assert common["opsd.grpo_opsd_mode"] == "hybrid"
    assert recipe["experiments"]["base_mix30k"]["rows_before_filter"] == 30000
    assert recipe["experiments"]["mopd_rstg15k"]["rows_before_filter"] == 15000
    assert 400 * 256 == 102400
    assert 400 * 256 * 8 == 819200


def test_aligned_grpo_common_fields_match_original_except_mode() -> None:
    original = load_json(SOURCE_RECIPE)
    aligned = load_json(ALIGNED_RECIPE)
    assert aligned["datasets"] == original["datasets"]
    assert aligned["data_policy"] == original["data_policy"]
    assert set(aligned["common"]) == set(original["common"])
    differences = {
        key: (original["common"][key], aligned["common"][key])
        for key in original["common"]
        if original["common"][key] != aligned["common"][key]
    }
    assert differences == {"opsd.grpo_opsd_mode": ("hybrid", "grpo")}
    assert aligned["alignment_contract"]["t2_t3_algorithm"] == {
        "opsd.grpo_opsd_mode": "hybrid"
    }
    assert aligned["common"]["opsd.enable_grpo_opsd_simple"] is True
    assert aligned["common"]["opsd.binary_reward_groups"] is True


def test_pairwise_data_and_initialization_contracts() -> None:
    original = load_json(SOURCE_RECIPE)["experiments"]
    aligned = load_json(ALIGNED_RECIPE)["experiments"]
    t1 = aligned["t1_base_grpo"]
    t2 = original["base_mix30k"]
    t3 = original["mopd_rstg15k"]
    t5 = aligned["t5_post_sft_grpo"]
    assert t1["comparison_arm"] == "T2"
    assert t5["comparison_arm"] == "T3"
    assert t1["datasets"] == t2["datasets"] == ["mopd15k", "rstg15k"]
    assert t5["datasets"] == t3["datasets"] == ["rstg15k"]
    assert t1["rows_before_filter"] == t2["rows_before_filter"] == 30000
    assert t5["rows_before_filter"] == t3["rows_before_filter"] == 15000
    assert t1["model"] == t2["model"] == t5["base_model"]
    assert t1["model_revision"] == t5["base_model_revision"]
    assert t2["model_revision"] == "RECORD_ACTUAL_DOWNLOADED_REVISION"
    assert t1["stage_sequence"] == ["Base", "GRPO"]
    assert t5["stage_sequence"] == ["Base", "SFT", "GRPO"]


def test_aligned_recipe_preserves_blockers_and_original_protocol() -> None:
    recipe = load_json(ALIGNED_RECIPE)
    assert recipe["config_status"] == "PLANNED_BLOCKED_NOT_A_COMPLETE_RUNTIME_CONFIG"
    assert recipe["execution_status"] == "NOT_LAUNCHED"
    assert set(recipe["run_blockers"]) == {
        "original_runtime_and_dependencies",
        "original_oss_v3_reward",
        "original_independent_validation",
        "base_checkpoint_equivalence",
        "t5_sft_parent",
        "resolved_config_and_semantic_smoke",
    }
    for blocker in recipe["run_blockers"].values():
        assert blocker.startswith(("MISSING:", "REQUIRES_SERVER_VERIFICATION:"))
    assert recipe["common"]["reward_identity"] == "rstg_oss_verifier_v3"
    assert recipe["common"]["reward_source_status"] == "ORIGINAL_SOURCE_AND_DEPENDENCIES_REQUIRED"
    assert recipe["data_policy"]["val"] == "ORIGINAL_INDEPENDENT_VALIDATION_ASSET_REQUIRED"
    assert recipe["prompt_protocol"]["published_think_prompt_may_be_used_directly"] is False
    manifest = load_json(HANDOFF / "PACKAGE_SHA256.json")
    template = "reference_code/examples/visual_rl/format_prompt/unified.jinja"
    assert recipe["prompt_protocol"]["sha256"] == manifest[template]["sha256"]
    t5 = recipe["experiments"]["t5_post_sft_grpo"]
    assert t5["model"] == "RECORD_FINAL_SFT_PARENT_CHECKPOINT"
    assert t5["model_revision"] == "RECORD_IMMUTABLE_SFT_PARENT_REVISION"


def test_t5_sft_volume_uses_drop_last_accounting() -> None:
    document = (MIDTERM / "training_alignment.md").read_text(encoding="utf-8")
    sft = load_json(ALIGNED_RECIPE)["experiments"]["t5_post_sft_grpo"]["sft_volume_contract"]
    rows = sft["train_rows"]
    batch = sft["global_batch_size"]
    epochs = sft["total_epochs"]
    steps_per_epoch = rows // batch
    steps = steps_per_epoch * epochs
    assert sft["drop_last"] is True
    assert (rows, batch, epochs) == (148054, 64, 3)
    assert (steps_per_epoch, steps, rows % batch) == (2313, 6939, 22)
    assert steps == sft["planned_optimizer_updates"]
    assert steps * batch == sft["planned_example_consumptions"] == 444096
    assert rows % batch == sft["dropped_rows_per_epoch"]
    assert "6,939 global/optimizer steps" in document
    assert "444,096 consumed examples" in document


def test_handoff_package_manifest_matches_checked_out_bytes() -> None:
    records = load_json(HANDOFF / "PACKAGE_SHA256.json")
    actual_files = {
        path.relative_to(HANDOFF).as_posix()
        for path in HANDOFF.rglob("*")
        if path.is_file()
        and path.name != "PACKAGE_SHA256.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(records) == actual_files
    for name, record in records.items():
        content = (HANDOFF / name).read_bytes()
        assert len(content) == record["bytes"], name
        assert hashlib.sha256(content).hexdigest() == record["sha256"], name


def test_handoff_documents_only_self_contained_reference_tests() -> None:
    guide = (HANDOFF / "环境与数据准备.md").read_text(encoding="utf-8")
    assert (
        "python -m unittest test_hybrid_groups "
        "test_rlsd_rlrt_weights.WeightTests test_rlsd_rlrt_weights.ConfigTests"
    ) in guide
    assert "26" in guide
    assert "6" in guide
    assert "torch_functional.py" in guide
    assert "python -m unittest discover" not in guide
