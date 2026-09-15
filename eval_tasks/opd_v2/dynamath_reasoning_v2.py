import sys
from pathlib import Path

# Sibling-module import (repo-owned extraction helpers) plus reuse of the
# upstream dynamath scoring chain.  ``eval_tasks/opd_v2`` is injected via
# TaskManager --include_path, so yaml !function resolution loads this file by
# path; make the sibling import work in that context.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _extraction import extract_answer_tag  # noqa: E402,F401  (re-exported for tests)

from lmms_eval.tasks.dynamath.reasoning.utils import (  # noqa: E402
    SYSTEM_PROMPT,
    dynamath_aggregate_results_average,
    dynamath_aggregate_results_worst,
    dynamath_doc_to_messages_cot,
    dynamath_doc_to_text_cot,
    dynamath_doc_to_visual,
    dynamath_process_results as _upstream_dynamath_process_results,
)

# lmms-eval v0.7.1's TaskManager indexes bundled tasks first, so a yaml with
# the upstream task name would silently lose the conflict resolution.  The
# v2 name sidesteps that; these aliases keep the official doc-to-* functions
# and scoring chain exactly as-is.
dynamath_v2_doc_to_text = dynamath_doc_to_text_cot
dynamath_v2_doc_to_visual = dynamath_doc_to_visual
dynamath_v2_doc_to_messages = dynamath_doc_to_messages_cot
dynamath_v2_aggregate_results_average = dynamath_aggregate_results_average
dynamath_v2_aggregate_results_worst = dynamath_aggregate_results_worst


def dynamath_v2_process_results(doc, results):
    # Upstream dynamath_process_results feeds raw predictions into
    # reasoning_utils.compute_score, whose extract_anwser_tag uses re.search
    # (first <answer> match).  MMF-SFT models emit multiple directed-style
    # blocks; the final one holds the settled answer.  Strip earlier tags so
    # the first-match semantics land on the settled answer, keeping every
    # other scoring step identical.
    preds = [str(p) for p in results]
    retagged = [_strip_stale_answer_tags(p) for p in preds]
    return _upstream_dynamath_process_results(doc, retagged)


def _strip_stale_answer_tags(pred: str) -> str:
    """Delete every <answer>...</answer> block except the last one.

    compute_score's extract_anwser_tag takes the *first* <answer> match and
    format_reward fullmatch wants a think-then-answer shape.  Stripping the
    earlier tags and leaving the final one intact makes both behave on
    multi-block directed responses.
    """
    import re as _re

    matches = list(_re.finditer(r"<answer>.*?</answer>", pred, _re.DOTALL))
    if len(matches) <= 1:
        return pred
    out = []
    pos = 0
    for m in matches[:-1]:
        out.append(pred[pos : m.start()])
        pos = m.end()
    out.append(pred[pos:])
    return "".join(out)
