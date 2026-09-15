import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

# Sibling-module import: lmms-eval loads task utils via spec_from_file_location
# (no package context), so put this directory on sys.path deterministically.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _extraction import extract_answer_tag, extract_option_letter  # noqa: E402

from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
    get_multi_choice_info as shared_get_multi_choice_info,
)
from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
    parse_mmmu_multi_choice_response,
)

# Same question/options rendering as the upstream mmmu_pro_standard task,
# but the answer instruction asks for <answer> tags so long-CoT checkpoints
# terminate cleanly and can be scored without first-letter-in-CoT artifacts.
POST_PROMPT = "Answer with the option letter inside <answer></answer> tags."


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(f"{letter}. {option}" for letter, option in zip(option_letters, options))


def construct_prompt(doc, post_prompt=POST_PROMPT):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(doc["options"]))
    return f"{question}\n{parsed_options}\n\n{post_prompt}"


def mmmu_pro_v2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = construct_prompt(doc)
    return question


def mmmu_pro_v2_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    image_tokens = sorted({token.strip("<>").replace(" ", "_") for token in image_tokens})
    return [doc[token].convert("RGB") for token in image_tokens]


def _deterministic_candidate_found(response: str, all_choices, index2ans) -> bool:
    """Replicate the official parser's candidate scans without its random branch.

    parse_mmmu_multi_choice_response calls random.choice(all_choices) when no
    candidate matched; the return value is then indistinguishable from a real
    parse.  Running the same scans here lets us skip the parser exactly when
    it would be random.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    for choice in all_choices:
        if f"({choice})" in response:
            return True
    for choice in all_choices:
        if f"{choice} " in response:
            return True
    for choice in all_choices:
        if f"{choice}." in response:
            return True
    if len(response.split()) > 5:
        for ans in index2ans.values():
            if str(ans).lower() in response.lower():
                return True
    return False


def mmmu_pro_v2_extract_pred(pred, all_choices, index2ans):
    """Deterministic extraction: <answer> tag first, official parser fallback.

    The official parser's random.choice branch (used when nothing parses) is
    replaced by an empty prediction so scores stay reproducible.
    """
    tag = extract_answer_tag(pred)
    if tag is not None:
        letter = extract_option_letter(tag)
        if letter in all_choices:
            return letter
        # Tag present but no valid option letter inside — treat as unparsed
        # rather than letting a stray CoT letter win.
        return ""
    if not _deterministic_candidate_found(pred, all_choices, index2ans):
        # No candidate matched: the upstream random.choice fallback would
        # inject noise, so return an explicit unparsed marker instead.
        return ""
    return parse_mmmu_multi_choice_response(pred, all_choices, index2ans)


def mmmu_pro_v2_process_results(doc, results):
    pred = results[0]
    index2ans, all_choices = shared_get_multi_choice_info(ast.literal_eval(doc["options"]))
    parsed_pred = mmmu_pro_v2_extract_pred(pred, all_choices, index2ans)
    mmmu_acc = {"id": doc["id"], "subject": doc["subject"], "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {"mmmu_acc": mmmu_acc}


def mmmu_pro_v2_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subject"]].append(result)

    for subset, sub_eval_samples in subset_to_eval_samples.items():
        acc = sum(1 for s in sub_eval_samples if s["answer"] == s["parsed_pred"]) / len(sub_eval_samples)
        evaluation_result[subset] = {"acc": acc, "num_example": len(sub_eval_samples)}

    total = sum(v["num_example"] for v in evaluation_result.values())
    if total == 0:
        return 0.0
    all_ins_acc = sum(v["acc"] * v["num_example"] for v in evaluation_result.values()) / total
    return all_ins_acc
