"""Shared answer extraction for the OPD v2 evaluation tasks.

The v2 protocol scores every generation through a two-stage chain:

1. If the response contains an explicit ``<answer>...</answer>`` tag (the format
   our MMF-SFT checkpoints emit after long chain-of-thought), extract the tag
   content and treat it as the prediction.
2. Otherwise fall back to the benchmark's legacy extraction applied to the raw
   response (e.g. last-line option letter, official MCQ parser).

All fallbacks are deterministic — we never fabricate a prediction via
``random.choice`` when nothing can be extracted.

These helpers are imported by the sibling task ``utils.py`` files through
lmms-eval's relative-file ``!function`` resolution, and by repository unit
tests (``tests/eval_v2/``) directly.
"""

from __future__ import annotations

import re

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
STANDALONE_LETTER_RE = re.compile(r"\b([A-Z])\b")


def extract_answer_tag(response: str) -> str | None:
    """Return the content of the last ``<answer>`` tag, if any."""

    matches = list(ANSWER_TAG_RE.finditer(str(response or "")))
    return matches[-1].group(1).strip() if matches else None


def extract_option_letter(text: str) -> str | None:
    """Return the first standalone option letter (``A``-``Z``) in ``text``."""

    match = STANDALONE_LETTER_RE.search(str(text or ""))
    return match.group(1) if match else None


def mcq_answer(response: str, *, fallback: str | None = None) -> str | None:
    """Extract an MCQ letter with ``<answer>``-tag priority.

    Order: ``<answer>`` tag content -> ``fallback`` (task-specific legacy
    extraction over the raw response, e.g. last-line letter).
    """

    tag = extract_answer_tag(response)
    if tag is not None:
        return extract_option_letter(tag)
    return fallback


def last_line_option_letter(response: str) -> str | None:
    """Legacy ViewSpatial extraction: standalone letter on the last line."""

    lines = [line for line in str(response or "").split("\n") if line.strip()]
    if not lines:
        return None
    return extract_option_letter(lines[-1])


def normalize_word(text: str) -> str:
    """GQA-style normalization: lowercase, drop punctuation, collapse spaces."""

    lowered = str(text or "").strip().lower()
    stripped = re.sub(r"[^\w\s]", "", lowered)
    return " ".join(stripped.split())


def short_answer(response: str) -> str:
    """GQA-style extraction: ``<answer>`` tag content, else the full response."""

    tag = extract_answer_tag(response)
    return tag if tag is not None else str(response or "").strip()
