import re
import sys
from pathlib import Path

# Sibling-module import: lmms-eval loads task utils via spec_from_file_location
# (no package context), so a bare "from _extraction import ..." only works when
# this directory happens to be on sys.path.  Make that true deterministically.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _extraction import (  # noqa: E402
    extract_answer_tag,
    extract_option_letter,
    last_line_option_letter,
    mcq_answer,
)

# Prompt mirrors the original ViewSpatial-Bench task but asks for the
# letter inside <answer> tags so long-CoT checkpoints can finish within
# budget and still be scored deterministically.
POST_PROMPT = (
    "Reply only to the corresponding option.\n"
    "Answer with the option letter inside <answer></answer> tags."
)


def viewspatial_v2_doc_to_text(doc):
    question = doc["question"]
    choices = doc["choices"]
    question_text = f"Question: {question}\n"
    choices_text = f"Choices: {choices}\n"
    return question_text + choices_text + POST_PROMPT


def viewspatial_v2_doc_to_visual(doc):
    return [visual.convert("RGB") for visual in doc["images"]]


def viewspatial_v2_process_results(doc, results):
    grounded_output = doc["answer"]
    grounded_option = extract_option_letter(grounded_output)
    pred = mcq_answer(results[0], fallback=last_line_option_letter(results[0]))
    score = 1.0 if pred == grounded_option else 0.0
    return {"overall_accuracy": {"score": score, "pred": pred}}


def viewspatial_v2_aggregate_results(results):
    total_score = 0.0
    for res in results:
        total_score += res["score"]
    return total_score / len(results) if results else 0.0
