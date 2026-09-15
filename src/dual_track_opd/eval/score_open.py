"""Open-ended scoring helpers."""

from __future__ import annotations

import re


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)
XML_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is\s*|[:=]\s*)+([^\n\r]+)",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Normalize answer text for lightweight comparisons."""

    return " ".join(text.strip().lower().split())


def extract_normalized_answer(response: str) -> str | None:
    """Extract and normalize an explicitly marked final answer.

    Replay datasets intentionally preserve historical prompts.  A trained model may
    still produce reasoning even when the prompt requests an answer-only response.
    Comparing the full generation against a short gold string would therefore
    under-count correct answers.  We accept explicit markers used by our models
    (``\\boxed{}``, XML ``<answer>``, and a final-answer line).  Ambiguous unmarked
    reasoning is treated as unanswered rather than silently taking its first or
    last number; callers may compare a normalized full response separately when it
    exactly equals gold.
    """
    text = str(response or "").strip()
    if not text:
        return None
    candidate: str | None = None
    for pattern in (BOXED_RE, XML_ANSWER_RE, FINAL_ANSWER_RE):
        matches = list(pattern.finditer(text))
        if matches:
            candidate = matches[-1].group(1)
            break
    if candidate is None:
        return None
    normalized = normalize_answer(candidate.strip().rstrip(".*_`$ ").strip("*_`$ "))
    return normalized or None
