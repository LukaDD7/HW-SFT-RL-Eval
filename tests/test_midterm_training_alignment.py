import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIDTERM = ROOT / "docs/midterm_report_202611"
HANDOFF = MIDTERM / "rlsd_rlrt_reproduction_20261124"


def test_rl_cohort_volume_and_step_units() -> None:
    document = (MIDTERM / "training_alignment.md").read_text(encoding="utf-8")
    rows = {
        "T1 / GRPO": (15000, 50, 16, 8, 8, 100),
        "T5 / post-SFT GRPO": (15000, 50, 16, 8, 8, 100),
        "T2 / Base hybrid": (30000, 400, 256, 256, 8, 400),
        "T3 / post-MOPD hybrid": (15000, 400, 256, 256, 8, 400),
    }
    for arm, (records, steps, batch, mini_batch, rollouts, updates) in rows.items():
        prompts = steps * batch
        trajectories = prompts * rollouts
        assert steps * (batch // mini_batch) == updates
        expected = (
            f"| {arm} | {records:,} | {steps} | {batch} | {mini_batch} | "
            f"{rollouts} | {prompts:,} | {trajectories:,} |"
        )
        assert expected in document
        assert updates in (100, 400)
    assert "at most 400" in document
    assert "global rollout steps\nand optimizer updates separately" in document


def test_pr_recipe_matches_hybrid_cohort() -> None:
    recipe = json.loads((HANDOFF / "configs/experiment_recipe.json").read_text(encoding="utf-8"))
    common = recipe["common"]
    assert common["trainer.max_steps"] == 400
    assert common["data.rollout_batch_size"] == 256
    assert common["data.mini_rollout_batch_size"] == 256
    assert common["worker.actor.global_batch_size"] == 256
    assert common["worker.rollout.n"] == 8
    assert common["worker.actor.ppo_epochs"] == 1
    assert common["worker.actor.optim.lr"] == 1e-6
    assert common["worker.rollout.seed"] == 1
    assert recipe["experiments"]["base_mix30k"]["rows_before_filter"] == 30000
    assert recipe["experiments"]["mopd_rstg15k"]["rows_before_filter"] == 15000
    assert 400 * 256 == 102400
    assert 400 * 256 * 8 == 819200


def test_t5_sft_volume_uses_drop_last_accounting() -> None:
    rows, batch, epochs = 148054, 64, 3
    steps_per_epoch = rows // batch
    steps = steps_per_epoch * epochs
    assert (steps_per_epoch, steps, rows % batch) == (2313, 6939, 22)
    assert steps * batch == 444096
    document = (MIDTERM / "training_alignment.md").read_text(encoding="utf-8")
    assert "6,939 global/optimizer steps" in document
    assert "444,096 consumed examples" in document


def test_handoff_package_manifest_matches_checked_out_bytes() -> None:
    records = json.loads((HANDOFF / "PACKAGE_SHA256.json").read_text(encoding="utf-8"))
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
