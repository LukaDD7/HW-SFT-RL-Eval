import json
import os
from pathlib import Path

from dual_track_opd.eval.project_summary import summarize_run
from dual_track_opd.eval.score_open import extract_normalized_answer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval" / "project_vision_opd.yaml"


def test_summary_extracts_primary_metric_and_excludes_internal_tier(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for repeat_index, value in enumerate((0.2, 0.4, 0.6, 0.8)):
        result_dir = run_dir / "lmms" / "viewspatial" / f"repeat_{repeat_index}"
        result_dir.mkdir(parents=True)
        (result_dir / "results.json").write_text(
            json.dumps({"groups": {"viewspatial": {"overall_accuracy,none": value}}}),
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
                    "repeat_count": 4,
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


def test_summary_reports_remi_strict_avg4(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "remi_avg4.json").write_text(
        json.dumps(
            {
                "mode": "exact",
                "metric": "exact_accuracy_full_denominator",
                "aggregation": "mean",
                "repeat_count": 4,
                "values": [0.2, 0.3, 0.4, 0.5],
                "mean": 0.35,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": "/checkpoint",
                "repeat_count": 4,
                "runs": [{"benchmark_id": "remi", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_run(run_dir, CONFIG)
    row = next(row for row in summary["rows"] if row["benchmark_id"] == "remi")
    assert row["value"] == 0.35
    assert row["source"].endswith("remi_avg4.json")


def test_summary_uses_latest_result_when_a_repeat_was_retried(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for repeat_index, value in enumerate((0.2, 0.4, 0.6, 0.8)):
        result_dir = run_dir / "lmms" / "viewspatial" / f"repeat_{repeat_index}"
        result_dir.mkdir(parents=True)
        (result_dir / "old.json").write_text(
            json.dumps({"results": {"viewspatial": {"overall_accuracy,none": 0.0}}}),
            encoding="utf-8",
        )
        latest = result_dir / "new.json"
        latest.write_text(
            json.dumps({"results": {"viewspatial": {"overall_accuracy,none": value}}}),
            encoding="utf-8",
        )
        os.utime(latest, (1_800_000_000 + repeat_index, 1_800_000_000 + repeat_index))
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "repeat_count": 4,
                "runs": [{"benchmark_id": "viewspatial", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_run(run_dir, CONFIG)
    row = next(row for row in summary["rows"] if row["benchmark_id"] == "viewspatial")
    assert row["value"] == 0.5
    assert row["source"].count("new.json") == 4


def test_summary_reports_mv_math_strict_avg4(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "mv_math_avg4.json").write_text(
        json.dumps(
            {
                "metric": "strict_weighted_accuracy",
                "aggregation": "mean",
                "repeat_count": 4,
                "values": [0.2, 0.3, 0.4, 0.5],
                "mean": 0.35,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "repeat_count": 4,
                "runs": [{"benchmark_id": "mv_math", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_run(run_dir, CONFIG)
    row = next(row for row in summary["rows"] if row["benchmark_id"] == "mv_math")
    assert row["value"] == 0.35
    assert row["source"].endswith("mv_math_avg4.json")
