import json
from pathlib import Path

from dual_track_opd.eval.project_summary import summarize_run
from dual_track_opd.eval.score_open import extract_normalized_answer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval" / "project_vision_opd.yaml"


def test_summary_extracts_primary_metric_and_excludes_internal_tier(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result_dir = run_dir / "lmms" / "viewspatial"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "results": {"viewspatial_child": {"overall_accuracy,none": 0.1}},
                "groups": {"viewspatial": {"overall_accuracy,none": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    replay_dir = run_dir / "replay"
    replay_dir.mkdir()
    (replay_dir / "remi.jsonl").write_text(
        json.dumps({"prediction": " Two ", "ground_truth": "two", "error": ""}) + "\n"
        + json.dumps(
            {
                "prediction": "Reason first. <answer>wrong</answer>",
                "ground_truth": "two",
                "error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": "/checkpoint",
                "dataset_manifest_hash": "abc",
                "runs": [
                    {"benchmark_id": "viewspatial", "status": "completed"},
                    {"benchmark_id": "remi", "status": "completed"},
                    {"benchmark_id": "mmvet", "status": "deferred", "reason": "judge required"},
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_run(run_dir, CONFIG)
    values = {row["benchmark_id"]: row["value"] for row in summary["rows"]}
    assert values == {"viewspatial": 0.5, "remi": None, "mmvet": None}
    assert summary["category_macro_excluding_internal_diagnostics"] == {"vqa": 0.5}


def test_replay_scoring_extracts_marked_answer_from_reasoning() -> None:
    assert extract_normalized_answer("Reason first.\n<answer> two </answer>") == "two"
    assert extract_normalized_answer("Reason first. The final answer is **42**.") == "42"
    assert extract_normalized_answer("Reason without an answer marker.") is None
