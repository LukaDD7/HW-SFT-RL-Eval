import json

from dual_track_opd.eval.score_mv_math import (
    _parse_judge_verdict,
    _strict_summary,
    extract_choice,
    score_choice_rows,
    score_strict_from_judge_sidecar,
    summarize,
)


def test_extract_choice_prefers_final_answer_markers() -> None:
    response = "A is wrong. Reasoning... The final answer is: C\nD"
    assert extract_choice(response) == "D"
    assert extract_choice("The final answer is <answer>B</answer>") == "B"
    assert extract_choice("\\boxed{A}") == "A"
    assert extract_choice("b") == "B"
    assert extract_choice("no option") is None


def test_choice_full_denominator_and_truncation_stats() -> None:
    rows = [
        {"sample_id": "1", "prediction": "The answer is A", "ground_truth": "A"},
        {"sample_id": "2", "prediction": "", "ground_truth": "B", "finish_reason": "length"},
        {"sample_id": "3", "prediction": "C", "ground_truth": "C", "finish_reason": "length"},
    ]
    scored = score_choice_rows(rows)
    summary = summarize(scored)
    assert [row["correct"] for row in scored] == [True, False, True]
    assert summary == {
        "rows": 3,
        "correct": 2,
        "truncated": 2,
        "judge_unparsed_or_unextracted": 0,
        "accuracy_full_denominator": 2 / 3,
    }


def test_judge_verdicts_use_official_shapes() -> None:
    assert _parse_judge_verdict("True", "single-step") is True
    assert _parse_judge_verdict("false", "choice") is False
    assert _parse_judge_verdict(" 2/3 ", "multi-step") == (2, 3)
    assert _parse_judge_verdict("invalid", "multi-step") is None


def test_strict_mode_gates_truncated_judge_true(tmp_path) -> None:
    rows = [
        {"sample_id": "1", "answer_type": "choice", "finish_reason": "stop", "prediction": "A", "ground_truth": "A"},
        {"sample_id": "2", "answer_type": "choice", "finish_reason": "length", "prediction": "reasoning", "ground_truth": "B"},
        {
            "sample_id": "3",
            "answer_type": "multi-step",
            "finish_reason": "length",
            "prediction": "partial",
            "ground_truth": "(1) 1\n(2) 2",
        },
    ]
    sidecar = tmp_path / "judge.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "1", "judge_verdict": True, "judge_raw": "true"}),
                json.dumps({"sample_id": "2", "judge_verdict": True, "judge_raw": "true"}),
                json.dumps({"sample_id": "3", "judge_verdict": [2, 2], "judge_raw": "2/2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scored = score_strict_from_judge_sidecar(rows, sidecar)
    summary = _strict_summary(scored)
    assert [row["correct"] for row in scored] == [True, False, False]
    assert [row["completion_gated"] for row in scored] == [False, True, True]
    assert summary["correct"] == 1
    assert summary["completion_gated_rows"] == 2
    assert summary["strict_weighted_accuracy"] == 1 / 3


def test_json_output_shape_matches_official_protocol(tmp_path) -> None:
    # This exercises metadata loading without requiring a judge endpoint.
    from dual_track_opd.eval.score_mv_math import load_metadata

    metadata = [
        {"problem_id": 1, "answer": "B", "answer_type": "choice"},
        {"problem_id": 2, "answer": "7", "answer_type": "single-step"},
        {"problem_id": 3, "answer": "(1) 1\n(2) 2", "answer_type": "multi-step"},
    ]
    path = tmp_path / "MV-MATH.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded = load_metadata(path)
    assert set(loaded) == {"1", "2", "3"}
    assert loaded["3"]["answer_type"] == "multi-step"
