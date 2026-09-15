"""Multiple-choice scoring helpers."""

from __future__ import annotations


def exact_match(prediction: str, answer: str) -> bool:
    """Case-insensitive exact match for simple smoke scoring."""

    return prediction.strip().lower() == answer.strip().lower()
