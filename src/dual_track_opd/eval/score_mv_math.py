"""Scoring utilities for MV-MATH replay outputs.

MV-MATH's official evaluator is not an exact-match scorer.  The official
repository sends each model response and gold answer to an LLM judge:

* choice and single-step questions are judged as true/false;
* multi-step questions are judged as ``correct_steps/total_steps``;
* the headline score counts choice+single-step true rows and multi-step rows
  where all steps are correct, divided by 2,009.

This module also provides a conservative deterministic diagnostic for choice
rows.  It is useful when no judge endpoint is available, but it is not the
official full MV-MATH metric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.IGNORECASE)
ANSWER_IS_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n\r]+)", re.IGNORECASE
)
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
FRACTION_RE = re.compile(r"([0-9]+)\s*/\s*([0-9]+)")
TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(item)
    return rows


def _coerce_judge_verdict(value: Any) -> Any:
    """Restore a JSON-serialized multi-step verdict to a tuple."""

    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    ):
        return value[0], value[1]
    return value


def load_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load MV-MATH metadata indexed by problem id."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("MV-MATH metadata must be a JSON array")
    metadata: dict[str, dict[str, Any]] = {}
    for entry in data:
        problem_id = str(entry.get("problem_id", ""))
        if not problem_id:
            raise ValueError("MV-MATH metadata contains a row without problem_id")
        if problem_id in metadata:
            raise ValueError(f"duplicate MV-MATH problem_id: {problem_id}")
        metadata[problem_id] = entry
    return metadata


def _last_match(matches: list[re.Match[str]]) -> str | None:
    return matches[-1].group(1).strip() if matches else None


def extract_choice(response: str) -> str | None:
    """Extract an A-D option, preferring explicit final-answer markers."""

    text = str(response or "").strip()
    if not text:
        return None

    tag = _last_match(list(ANSWER_TAG_RE.finditer(text)))
    boxed = _last_match(list(BOXED_RE.finditer(text)))
    marked = _last_match(list(ANSWER_IS_RE.finditer(text)))
    candidates = [value for value in (tag, boxed, marked) if value]
    for value in candidates:
        match = re.fullmatch(r"\(?([A-D])\)?", value.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # An answer-only response is valid.  For long chain-of-thought, use the
    # final standalone A-D token rather than the first mention of an option.
    letters = list(LETTER_RE.finditer(text))
    if letters:
        return letters[-1].group(1).upper()
    return None


def _normalize_text(value: str) -> str:
    value = str(value or "").strip().strip("$").strip()
    value = value.replace("（", "(").replace("）", ")")
    return " ".join(value.lower().replace(",", ", ").split()).strip().strip(".").strip()


def _numeric(value: str) -> float | None:
    cleaned = str(value or "").strip().removeprefix("$").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _deterministic_equal(prediction: str, gold: str) -> bool:
    pred_num, gold_num = _numeric(prediction), _numeric(gold)
    if pred_num is not None and gold_num is not None:
        tolerance = 0.01
        if gold_num == 0:
            return abs(pred_num) <= 1e-9
        return abs(pred_num - gold_num) <= tolerance * abs(gold_num)
    return _normalize_text(prediction) == _normalize_text(gold)


def score_choice_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict deterministic choice scoring with the full row denominator."""

    scored: list[dict[str, Any]] = []
    for row in rows:
        prediction = str(row.get("prediction", "") or "")
        extracted = extract_choice(prediction)
        gold = str(row.get("ground_truth", "") or "").strip().upper()
        scored.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "answer_type": "choice",
                "extracted": extracted,
                "ground_truth": gold,
                "correct": extracted == gold and extracted in {"A", "B", "C", "D"},
                "finish_reason": row.get("finish_reason"),
            }
        )
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    correct = sum(bool(item.get("correct")) for item in scored)
    truncated = sum(item.get("finish_reason") == "length" for item in scored)
    unextracted = sum(item.get("correct") is None for item in scored)
    return {
        "rows": total,
        "correct": correct,
        "truncated": truncated,
        "judge_unparsed_or_unextracted": unextracted,
        "accuracy_full_denominator": correct / total if total else None,
    }


def _judge_request(
    *,
    question: str,
    prediction: str,
    gold: str,
    answer_type: str,
    judge_url: str,
    judge_model: str,
    judge_key: str,
    timeout: float,
) -> str:
    if answer_type == "multi-step":
        system = (
            "You are a math expert. Check whether each sub-answer in the model's "
            "response matches the standard answer. Reply with only correct_steps/"
            "total_steps, for example 2/3 or 0/3."
        )
        user = (
            f"Question: {question}\n"
            f"Standard answer: {gold}\n"
            f"Model response: {prediction}\n"
            "Reply with only correct_steps/total_steps."
        )
        max_tokens = 32
    else:
        system = (
            "You are a math expert. Check whether the model's final answer matches "
            "the standard answer. Reply with only true or false."
        )
        user = (
            f"Question: {question}\n"
            f"Standard answer: {gold}\n"
            f"Model response: {prediction}\n"
            "Reply with only true or false."
        )
        max_tokens = 16

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{judge_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {judge_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    return str(message.get("content", "") or "")


def _parse_judge_verdict(content: str, answer_type: str) -> bool | tuple[int, int] | None:
    text = str(content or "").strip()
    if answer_type == "multi-step":
        match = FRACTION_RE.search(text)
        if not match:
            return None
        correct, total = int(match.group(1)), int(match.group(2))
        if 0 <= correct <= total and total > 0:
            return correct, total
        return None
    match = TRUE_FALSE_RE.search(text)
    return match.group(1).lower() == "true" if match else None


def score_official_with_judge(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    judge_url: str,
    judge_model: str,
    judge_key: str,
    workers: int,
    timeout: float,
    resume_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Score all MV-MATH types with an OpenAI-compatible LLM judge."""

    completed: dict[str, dict[str, Any]] = {}
    if resume_path is not None and resume_path.is_file():
        completed = {
            str(item.get("sample_id", "")): item
            for item in _read_jsonl(resume_path)
            if item.get("judge_verdict") is not None
        }
        for item in completed.values():
            item["judge_verdict"] = _coerce_judge_verdict(item.get("judge_verdict"))

    pending: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        entry = metadata.get(sample_id)
        if entry is None:
            raise KeyError(f"sample_id {sample_id!r} is absent from MV-MATH metadata")
        answer_type = str(entry.get("answer_type", ""))
        base = {
            "sample_id": sample_id,
            "answer_type": answer_type,
            "question": str(row.get("question", "") or entry.get("question", "")),
            "ground_truth": str(entry.get("answer", "") or row.get("ground_truth", "")),
            "prediction": str(row.get("prediction", "") or ""),
            "finish_reason": row.get("finish_reason"),
        }
        if sample_id in completed:
            item = dict(completed[sample_id])
            item.update({key: base[key] for key in ("sample_id", "answer_type", "question", "ground_truth", "prediction", "finish_reason")})
            prepared.append(item)
        else:
            pending.append(base)

    def judge_one(item: dict[str, Any]) -> dict[str, Any]:
        output = dict(item)
        try:
            content = _judge_request(
                question=item["question"],
                prediction=item["prediction"],
                gold=item["ground_truth"],
                answer_type=item["answer_type"],
                judge_url=judge_url,
                judge_model=judge_model,
                judge_key=judge_key,
                timeout=timeout,
            )
            verdict = _parse_judge_verdict(content, item["answer_type"])
            output["judge_raw"] = content
            output["judge_verdict"] = verdict
        except (OSError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
            output["judge_error"] = f"{type(exc).__name__}: {exc}"
            output["judge_verdict"] = None
        return output

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(judge_one, item) for item in pending]
        for future in as_completed(futures):
            prepared.append(future.result())

    scored: list[dict[str, Any]] = []
    for item in prepared:
        verdict = item.get("judge_verdict")
        if isinstance(verdict, tuple):
            correct_steps, total_steps = verdict
            item["correct_steps"] = correct_steps
            item["total_steps"] = total_steps
            item["correct"] = correct_steps == total_steps
            item["strictly_correct_steps"] = correct_steps
            item["total_judged_steps"] = total_steps
        else:
            item["correct"] = verdict is True
        scored.append(item)

    scored.sort(key=lambda item: int(item["sample_id"]))
    if resume_path is not None:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resume_path.with_suffix(resume_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in scored),
            encoding="utf-8",
        )
        temporary.replace(resume_path)
    return scored


def score_strict_from_judge_sidecar(
    rows: list[dict[str, Any]], sidecar_path: Path
) -> list[dict[str, Any]]:
    """Apply a completed-answer gate to an existing official judge sidecar.

    The official MV-MATH judge prompt can infer an intended answer from a
    truncated response.  For Project15's strict reporting rule, a row is only
    eligible for the headline score when the model finished generation
    (``finish_reason == "stop"``).  This function never mutates the source
    sidecar; it preserves judge verdicts and emits a gated copy for summary
    computation.
    """

    judged = {str(item.get("sample_id", "")): item for item in _read_jsonl(sidecar_path)}
    for item in judged.values():
        item["judge_verdict"] = _coerce_judge_verdict(item.get("judge_verdict"))
    scored: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in judged:
            raise KeyError(f"sample_id {sample_id!r} is absent from judge sidecar")
        item = dict(judged[sample_id])
        item.update(
            {
                "sample_id": sample_id,
                "answer_type": str(row.get("answer_type", item.get("answer_type", ""))),
                "finish_reason": row.get("finish_reason"),
                "prediction": str(row.get("prediction", "") or ""),
                "ground_truth": str(row.get("ground_truth", "") or ""),
                "completion_gated": row.get("finish_reason") != "stop",
            }
        )

        verdict = item.get("judge_verdict")
        if isinstance(verdict, tuple):
            correct_steps, total_steps = verdict
            item["judge_question_complete"] = correct_steps == total_steps
            item["correct"] = not item["completion_gated"] and item["judge_question_complete"]
        else:
            item["judge_correct"] = verdict is True
            item["correct"] = not item["completion_gated"] and item["judge_correct"]
        scored.append(item)

    scored.sort(key=lambda item: int(item["sample_id"]))
    return scored


def _official_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    total_correct = total_rows = 0
    total_correct_steps = total_steps = 0
    for item in scored:
        answer_type = str(item.get("answer_type", "unknown"))
        bucket = by_type.setdefault(answer_type, {"rows": 0, "correct": 0})
        bucket["rows"] += 1
        correct = bool(item.get("correct"))
        bucket["correct"] += int(correct)
        total_rows += 1
        total_correct += int(correct)
        if isinstance(item.get("judge_verdict"), tuple):
            total_correct_steps += int(item["judge_verdict"][0])
            total_steps += int(item["judge_verdict"][1])
    return {
        "rows": total_rows,
        "correct": total_correct,
        "official_weighted_accuracy": total_correct / total_rows if total_rows else None,
        "by_answer_type": by_type,
        "multi_step": {
            "step_accuracy_rate": total_correct_steps / total_steps if total_steps else None,
            "question_completeness_rate": (
                by_type.get("multi-step", {}).get("correct", 0)
                / by_type.get("multi-step", {}).get("rows", 0)
                if by_type.get("multi-step", {}).get("rows")
                else None
            ),
        },
    }


def _strict_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(scored)
    correct = sum(bool(item.get("correct")) for item in scored)
    gated = sum(bool(item.get("completion_gated")) for item in scored)
    judge_unparsed = sum(item.get("judge_verdict") is None for item in scored)
    multi_rows = [item for item in scored if item.get("answer_type") == "multi-step"]
    completed_multi = [item for item in multi_rows if not item.get("completion_gated")]
    completed_steps = sum(int(item["judge_verdict"][1]) for item in completed_multi)
    completed_correct_steps = sum(int(item["judge_verdict"][0]) for item in completed_multi)
    by_type: dict[str, dict[str, int]] = {}
    for item in scored:
        bucket = by_type.setdefault(str(item.get("answer_type", "unknown")), {"rows": 0, "correct": 0})
        bucket["rows"] += 1
        bucket["correct"] += int(bool(item.get("correct")))
    return {
        "rows": total_rows,
        "correct": correct,
        "completion_gated_rows": gated,
        "judge_unparsed_or_unextracted": judge_unparsed,
        "strict_weighted_accuracy": correct / total_rows if total_rows else None,
        "by_answer_type": by_type,
        "multi_step": {
            "completed_rows": len(completed_multi),
            "completed_steps": completed_steps,
            "step_accuracy_rate_completed_rows": (
                completed_correct_steps / completed_steps if completed_steps else None
            ),
            "question_completeness_rate": (
                sum(bool(item.get("correct")) for item in multi_rows) / len(multi_rows)
                if multi_rows
                else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-jsonl", required=True)
    parser.add_argument(
        "--metadata-json",
        default=None,
    )
    parser.add_argument("--mode", choices=("choice", "official", "strict"), default="choice")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8801/v1")
    parser.add_argument("--judge-model", default="Qwen3-VL-32B-Instruct")
    parser.add_argument("--judge-key", default="dummy")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-json")
    parser.add_argument("--judge-sidecar")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    replay_path = Path(args.replay_jsonl).expanduser()
    rows = _read_jsonl(replay_path)
    if args.mode == "choice":
        choice_rows = [row for row in rows if str(row.get("ground_truth", "")).strip().upper() in {"A", "B", "C", "D"}]
        scored = score_choice_rows(choice_rows)
        summary = {"protocol": "strict_deterministic_choice_diagnostic", **summarize(scored)}
    else:
        if args.metadata_json is None:
            raise ValueError("--metadata-json is required for official and strict modes")
        metadata = load_metadata(args.metadata_json)
        sidecar = Path(args.judge_sidecar).expanduser() if args.judge_sidecar else None
        if args.mode == "strict":
            if sidecar is None:
                raise ValueError("--mode strict requires --judge-sidecar")
            scored = score_strict_from_judge_sidecar(rows, sidecar)
            summary = {
                "protocol": "strict_completed_answer_official_judge",
                "judge_model": args.judge_model,
                **_strict_summary(scored),
            }
        else:
            if args.resume and sidecar is None:
                raise ValueError("--resume requires --judge-sidecar")
            scored = score_official_with_judge(
                rows,
                metadata,
                judge_url=args.judge_url,
                judge_model=args.judge_model,
                judge_key=args.judge_key,
                workers=args.workers,
                timeout=args.timeout,
                resume_path=sidecar if args.resume else None,
            )
            summary = {
                "protocol": "official_llm_equivalence_compatible",
                "judge_model": args.judge_model,
                **_official_summary(scored),
            }

    result = {
        "replay_jsonl": str(replay_path),
        **summary,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
