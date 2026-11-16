import re

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_final_answer_tag(text: str) -> str:
    """Return the final Open-MOPD answer-tag payload, preserving native text."""
    matches = list(ANSWER_TAG_RE.finditer(str(text or "")))
    if not matches:
        return str(text or "").strip()
    return matches[-1].group(1).strip()


def answer_tag_first():
    """Factory expected by lmms-eval filter_list YAML."""

    class AnswerTagFirstFilter:
        def apply(self, resps, docs):
            def select(resp):
                if not resp:
                    return ""
                return extract_final_answer_tag(resp[0])

            return map(select, resps)

    return AnswerTagFirstFilter()


def scienceqa_process_results(doc, results):
    from lmms_eval.tasks.scienceqa.utils import sqa_process_results as upstream
    return upstream(doc, [extract_final_answer_tag(results[0])])


def blink_process_results(doc, results):
    from lmms_eval.tasks.blink.utils import blink_process_results as upstream
    return upstream(doc, [extract_final_answer_tag(results[0])])


def vqav2_process_results_val(doc, results):
    from lmms_eval.tasks.vqav2.utils import vqav2_process_results_val as upstream
    return upstream(doc, [extract_final_answer_tag(results[0])])


def mindcube_process_results(doc, results):
    from lmms_eval.tasks.mindcube.utils import mindcube_process_results as upstream
    return upstream(doc, [extract_final_answer_tag(results[0])])


def msr_process_results(doc, results):
    from lmms_eval.tasks.mmsi_bench.utils import msr_process_results as upstream
    return upstream(doc, [extract_final_answer_tag(results[0])])
