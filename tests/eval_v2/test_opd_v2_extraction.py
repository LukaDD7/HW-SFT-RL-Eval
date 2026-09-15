"""Fixture-driven tests for the opd_v2 re-evaluation protocol.

Fixtures are verbatim responses from the nojudge baseline runs
(``eval_runs/vision_opd_project_baseline/{base_qwen3vl8b,mmf_only_1ep}``),
so every case encodes an observed generation shape, not a hypothetical.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
OPD_V2 = HERE.parents[1] / "eval_tasks" / "opd_v2"
sys.path.insert(0, str(OPD_V2))

from _extraction import (  # noqa: E402
    extract_answer_tag,
    extract_option_letter,
    last_line_option_letter,
    mcq_answer,
    normalize_word,
    short_answer,
)

from mmmu_pro_v2 import (  # noqa: E402
    mmmu_pro_v2_extract_pred as mmmu_extract_pred,
)

from viewspatial_v2 import (  # noqa: E402
    viewspatial_v2_process_results as vs_process_results,
)


# ---------------------------------------------------------------------------
# <answer>-tag extraction
# ---------------------------------------------------------------------------


def test_extract_answer_tag_takes_last_match() -> None:
    # MMF-SFT emits directed-style blocks; the settled answer is the last tag.
    pred = "first attempt <answer> 12 </answer> rechecking <answer> 7 </answer> done"
    assert extract_answer_tag(pred) == "7"


def test_extract_answer_tag_none_when_absent() -> None:
    assert extract_answer_tag("no tags, just reasoning") is None


def test_extract_option_letter_standalone_only() -> None:
    assert extract_option_letter("The answer is B.") == "B"
    assert extract_option_letter("therefore") is None  # no standalone capital


def test_mcq_answer_prefers_tag_over_last_line() -> None:
    pred = "A is wrong, B fits. Therefore <answer>B</answer>."
    assert mcq_answer(pred, fallback=last_line_option_letter(pred)) == "B"


# ---------------------------------------------------------------------------
# ViewSpatial v2 — the 0.0940 vs 0.4231 artifact case
# ---------------------------------------------------------------------------


def _doc(answer: str) -> dict:
    return {"question": "Where is the counter?", "choices": "A. right\nB. front-up\nC. back-left\nD. front", "answer": answer}


def test_viewspatial_tagged_response_scores_correct() -> None:
    # mmf_only_1ep row: target "B. right", response ends with the tag.
    resp = "The counter is to the right of the television.\n\nTherefore, the final answer is <answer>B</answer>."
    out = vs_process_results(_doc("B. right"), [resp])
    assert out["overall_accuracy"]["score"] == 1.0


def test_viewspatial_truncated_cot_falls_back_to_last_line() -> None:
    # Truncated mid-CoT at the 256 budget — no tag, no resolvable letter.
    resp = "The counter is to the right. D is front, which would be in front of"
    out = vs_process_results(_doc("A. right"), [resp])
    # Last line has no standalone option letter -> unparsed -> wrong (0.0),
    # which is the honest outcome for an unanswered row.
    assert out["overall_accuracy"]["score"] == 0.0


def test_viewspatial_base_direct_answer_still_works() -> None:
    # Base checkpoints answer with the bare letter.
    out = vs_process_results(_doc("A. right"), ["A"])
    assert out["overall_accuracy"]["score"] == 1.0


# ---------------------------------------------------------------------------
# GQA v2 — direct and tagged answers both scoreable
# ---------------------------------------------------------------------------


def test_gqa_short_answer_tag_priority() -> None:
    # mmf arm: "...the final answer is <answer>catcher</answer>." (gold: catcher)
    resp = "The person crouching behind the batter is ready to swing.\n\nTherefore, the final answer is <answer>catcher</answer>."
    assert short_answer(resp) == "catcher"
    assert normalize_word(short_answer(resp)) == "catcher"


def test_gqa_short_answer_direct_passthrough() -> None:
    # Base arm: "Yes" for gold "yes"; ignore_case handles it downstream.
    assert short_answer("Yes") == "Yes"
    assert normalize_word("Yes") == normalize_word("yes")


def test_gqa_truncated_cot_word_is_not_the_tag() -> None:
    # Without a tag the raw tail becomes the prediction — long CoT tails
    # score wrong against single-word gold, which is the real capability gap.
    resp = "The image shows a beach scene with people near the water and several umbrellas"
    assert short_answer(resp) == resp.strip()
    assert "<answer>" not in short_answer(resp)


# ---------------------------------------------------------------------------
# GQA v2 — process_results must emit numeric exact_match for mean aggregation
# (regression: returning the prediction string made `mean` raise TypeError
# after all docs were scored, and the run die rc=0 with no results.json)
# ---------------------------------------------------------------------------


def test_gqa_process_results_returns_numeric_exact_match() -> None:
    import gqa_v2 as g

    doc = {"question": "Is it raining?", "answer": "yes"}
    out = g.gqa_v2_process_results(doc, ["Yes."])
    assert out == {"exact_match": 1.0}
    assert isinstance(out["exact_match"], (int, float))

    out = g.gqa_v2_process_results(doc, ["No."])
    assert out == {"exact_match": 0.0}


def test_gqa_process_results_tag_priority_and_normalization() -> None:
    import gqa_v2 as g

    # ignore_case + ignore_punctuation semantics on both sides.
    doc = {"question": "Who is behind the batter?", "answer": "catcher"}
    resp = "The person crouching behind the batter is ready to swing.\n\nTherefore, the final answer is <answer>Catcher.</answer>."
    assert g.gqa_v2_process_results(doc, [resp])["exact_match"] == 1.0

    # Direct short response, case/punct handled.
    assert g.gqa_v2_process_results({"answer": "right"}, ["Right."])["exact_match"] == 1.0

    # Truncated CoT tail without a tag: raw text vs single-word gold -> 0.
    assert g.gqa_v2_process_results({"answer": "umbrella"}, ["beach scene with people near water"])["exact_match"] == 0.0


def test_gqa_mean_aggregation_over_numeric_scores() -> None:
    # The exact failure mode of the v2 first run: mean() over strings raised
    # TypeError.  With numeric per-doc values it must aggregate cleanly.
    import gqa_v2 as g

    items = [
        g.gqa_v2_process_results({"answer": "yes"}, ["Yes."])["exact_match"],
        g.gqa_v2_process_results({"answer": "no"}, ["Yes."])["exact_match"],
        g.gqa_v2_process_results({"answer": "yes"}, ["Yes."])["exact_match"],
    ]
    assert sum(items) / len(items) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# MMMU-Pro v2 — the first-letter-wins artifact
# ---------------------------------------------------------------------------


def _mmmu_options() -> tuple[dict, list]:
    options = ["Political instability leading to population decline", "The spread of pathogens across the Silk Road", "Development of trade routes", "Another factor"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    index2ans = {letters[i]: opt for i, opt in enumerate(options)}
    return index2ans, letters[: len(options)]


def test_mmmu_pro_tag_beats_earlier_cot_letters() -> None:
    # test_History_1 artifact: official parse scored "C" (first letter in CoT)
    # while the tagged final answer was "B" (the gold).
    index2ans, all_choices = _mmmu_options()
    resp = "A seems unlikely given the timeline. C may contribute locally, but the pattern spans regions.\n\nThus, **B** best explains the shared decline.\n\nTherefore, the final answer is <answer>B</answer>."
    assert mmmu_extract_pred(resp, all_choices, index2ans) == "B"


def test_mmmu_pro_tag_with_invalid_letter_is_unparsed() -> None:
    # Tag present but no valid option inside -> explicit unparsed, not a
    # stray CoT letter and not random.choice noise.
    index2ans, all_choices = _mmmu_options()
    resp = "Reasoning... <answer>Z</answer>"
    assert mmmu_extract_pred(resp, all_choices, index2ans) == ""


def test_mmmu_pro_no_tag_no_candidates_is_unparsed_not_random() -> None:
    index2ans, all_choices = _mmmu_options()
    resp = "The chart shows a decline across all regions over the period."
    assert mmmu_extract_pred(resp, all_choices, index2ans) == ""


def test_mmmu_pro_base_direct_letter() -> None:
    index2ans, all_choices = _mmmu_options()
    assert mmmu_extract_pred("B", all_choices, index2ans) == "B"


def test_mmmu_pro_no_tag_option_text_fallback() -> None:
    # >5 words without letter markers: option-text containment, as upstream.
    index2ans, all_choices = _mmmu_options()
    resp = "The best explanation is the spread of pathogens across the Silk Road."
    assert mmmu_extract_pred(resp, all_choices, index2ans) == "B"


# ---------------------------------------------------------------------------
# DynaMath v2 — last-tag semantics through the official scoring chain
# ---------------------------------------------------------------------------


def test_dynamath_v2_last_tag_wins_over_stale_first_tag() -> None:
    import dynamath_reasoning_v2 as d

    # Observed shape: an <answer> mid-solution then a final restated tag.
    # math_verify is exact on numerics, so 8.062 vs gold 8.0 scores 0 — the
    # extraction fix is orthogonal to numeric tolerance; assert the parse.
    doc = {"question": "Find AO.", "ground_truth": "8.062", "question_id": 0, "id": "1"}
    resp = "Compute AO.\n$$ AO = \\sqrt{65.000356} \\approx 8.062 $$\n\n<answer>8.062</answer>\nTherefore, the final answer is <answer>8.062</answer>."
    out = d.dynamath_v2_process_results(doc, [resp])
    assert out["dynamath_average"]["acc"] == 1.0

    # Same response with a stale first tag carrying a different value: the
    # settled (last) tag must be what gets scored.
    doc = {"question": "Find AO.", "ground_truth": "8.062", "question_id": 0, "id": "2"}
    resp_stale = "Compute AO.\n\n<answer>7.1</answer> Wait, rechecking.\n\nTherefore, the final answer is <answer>8.062</answer>."
    out = d.dynamath_v2_process_results(doc, [resp_stale])
    assert out["dynamath_average"]["acc"] == 1.0


def test_dynamath_v2_stale_tag_correction() -> None:
    import dynamath_reasoning_v2 as d

    # First tag is a wrong intermediate; the settled tag carries the fix.
    doc = {"question": "What is 2+3?", "ground_truth": "5", "question_id": 0, "id": "1"}
    resp = "Let me think. <answer>4</answer> Wait, rechecking. <answer>5</answer> Done."
    out = d.dynamath_v2_process_results(doc, [resp])
    assert out["dynamath_average"]["acc"] == 1.0


def test_dynamath_v2_single_tag_unchanged() -> None:
    import dynamath_reasoning_v2 as d

    doc = {"question": "What is 2+3?", "ground_truth": "5", "question_id": 0, "id": "1"}
    resp = "Adding gives five.\n<answer>5</answer>"
    out = d.dynamath_v2_process_results(doc, [resp])
    assert out["dynamath_average"]["acc"] == 1.0


def test_dynamath_v2_mcq_tag() -> None:
    import dynamath_reasoning_v2 as d

    # DynaMath MCQ ground truths are option letters; relax_exact_match path.
    doc = {"question": "How many zeros does f cross?", "ground_truth": "B", "question_id": 0, "id": "1"}
    resp = "The function has exactly one zero.\n<answer>B</answer>\nTherefore, the final answer is <answer>B</answer>."
    out = d.dynamath_v2_process_results(doc, [resp])
    assert out["dynamath_average"]["acc"] == 1.0
