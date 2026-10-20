from eval_tasks.opd_v1_offline.utils import extract_final_answer_tag


def test_extracts_final_answer_tag():
    response = "reason one\nreason two\n<answer>B</answer>"
    assert extract_final_answer_tag(response) == "B"


def test_extracts_last_answer_tag():
    response = "first <answer>A</answer> later <answer>down</answer>"
    assert extract_final_answer_tag(response) == "down"


def test_preserves_native_response_without_tag():
    response = "picnic table"
    assert extract_final_answer_tag(response) == "picnic table"
