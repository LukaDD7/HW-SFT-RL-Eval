#!/usr/bin/env python3
"""ReMI re-evaluation: task-aware exact-match first, llm-as-judge drain.

Two-tier scoring for replay JSONL outputs produced by
``dual_track_opd.eval.run_vlm_eval`` for the ReMI benchmark:

  1. EXACT tier (``--mode exact``) — task-aware exact match against the official
     label, the cousin of the ReMI colab rule.  Per-task answer extraction +
     normalization (numeric tolerance for float tasks, comma-list set equality
     for Maps/RefCOCO, TikZ collapse for CodeEdit, letter/city/tuple handling
     elsewhere), and — critically — a FULL denominator of every row in the file
     (2600), including rows whose prediction cannot be extracted.  This is the
     honest number that ``project_summary._summarize_replay`` hides: that helper
     counts only extractable/matched rows, so it inflates accuracy and masks the
     answer-only-format divergences that the SFT/RL checkpoints actually emit.

  2. JUDGE tier (``--mode judge``) — for rows the EXACT tier could not match
     (format-divergent predictions, e.g. a model emitting chain-of-thought
     despite the answer-only prompt), asks a hosted Qwen3-VL-32B-Instruct judge
     whether the prediction is semantically correct.  It is a DIAGNOSTIC for the
     format-alignment gap, not a replacement for the exact rule, and only drains
     rows that are extractable-but-wrong.

Neither tier modifies the input replay files nor ``project_summary.py``.

Usage (exact tier is read-only; no GPU, no network):
  python3 scripts/sft_rl/remi_reeval.py --mode exact --jsonl <remi.jsonl> \
      [--jsonl <remi2.jsonl> ...] \
      --label-jsonl eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl

  python3 scripts/sft_rl/remi_reeval.py --mode judge --jsonl <remi.jsonl> \
      --label-jsonl <raw_source.jsonl> \
      --judge-url http://127.0.0.1:8001/v1 --judge-model Qwen3-VL-32B-Instruct \
      [--workers 8] [--limit 50] [--out <sidecar.jsonl>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Task taxonomy.  `NUMERIC`, `LIST`, `LETTER`, `SCHEDULE`, `CODE`, `CHARTS`
# each get a dedicated extraction+match path; everything else falls through to
# a numeric-or-exact comparison.
# ---------------------------------------------------------------------------
NUMERIC = {"GeomCost", "GeomShape", "Collisions", "Clocks", "EmojiAlgebra", "FuncRead"}
LIST_LETTERS = {"Maps"}   # A/B/C/D/E multi-select, or "Equal"
LIST_NUMS = {"RefCoco"}   # comma-list of object-box indices
LETTER = {"IQ"}
BINARY = {"Isomorphism"}  # label 0/1, model may emit Yes/No
CITY = {"Schedule"}       # free-form city name
CODE = {"CodeEdit"}
CHARTS = {"Charts"}

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.IGNORECASE)
_XML_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is\s*|[:=]\s*)+([^\n\r]+)", re.IGNORECASE
)
_NUM_RE = r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def _strip(text: str) -> str:
    return text.strip().strip("`").strip("$").strip(".").strip()


def _norm(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def _num(text: Any) -> float | None:
    try:
        return float(str(text).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _extract_prediction(prediction: str, *, task: str) -> str | None:
    """Task-aware answer extraction from a prediction."""
    if not prediction or not prediction.strip():
        return None
    text = prediction.strip()

    m = _BOXED_RE.search(text)
    if m:
        return _strip(m.group(1))
    m = _XML_RE.search(text)
    if m:
        return _strip(m.group(1))
    m = _FINAL_RE.search(text)
    if m:
        return _strip(m.group(1))

    # Answer-only models (with no marker) fall back to the last answer-shaped
    # token.  Ambiguous free reasoning is still resolved to a trailing figure
    # rather than silently dropped, so the extraction rate is honest.
    if task in NUMERIC or task in CHARTS:
        found = re.findall(_NUM_RE, text)
        if found:
            return _strip(found[-1])
        return None
    if task in LIST_LETTERS:
        # Multi-select letter list (or single letter / "equal").  Whitespace
        # around the comma differs: "A,C" vs "A, C".
        m = re.search(r"([A-Ea-e](?:\s*,\s*[A-Ea-e])+)", text)
        if m:
            return _strip(re.sub(r"\s+", "", m.group(1))).upper()
        equal = re.search(r"\b(equal)\b", text, re.IGNORECASE)
        if equal:
            return "EQUAL"
        single = re.search(r"\b([A-Ea-e])\b", text)
        return single.group(1).upper() if single else None
    if task in LIST_NUMS:
        nums = re.findall(r"\b\d+\b", text)
        return ",".join(nums) if nums else None
    if task == "IQ":
        m = re.search(r"\b([A-Ea-e])\b", text)
        return m.group(1).upper() if m else None
    if task in BINARY:
        if re.search(r"\byes\b", text, re.IGNORECASE):
            return "1"
        if re.search(r"\bno\b", text, re.IGNORECASE):
            return "0"
        found = re.findall(_NUM_RE, text)
        return _strip(found[-1]) if found else None
    if task == "Schedule":
        # City names (Los Angeles, Los_Angeles) OR gate codes (G23) — both appear.
        code = re.search(r"\b([A-Z]\d{1,3})\b", text, re.IGNORECASE)
        if re.search(r"\b[A-Z]+_[A-Z]+\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b", text):
            toks = [t for t in _norm(text).replace(",", "").split() if t]
            if toks:
                # drop any "answer:"-style lead words like "the"/"answer"/"is"
                words = [t for t in toks if t.lower() not in ("the", "answer", "is", "city")]
                if words:
                    return _strip(_norm(" ".join(words[-2:])).title())
        if code:
            return code.group(1).upper()
        toks = [t for t in _norm(text).split() if t]
        return toks[-1] if toks else None
    if task == "CodeEdit":
        return _strip(text)
    return None


def _numeric_match(pred: str, label: str, rel_tol: float = 1e-3) -> bool:
    a, b = _num(pred), _num(label)
    if a is None or b is None:
        return False
    # Integer vs trailing-".0" labels ("930.0" vs "930") are the same quantity.
    if a.is_integer() and b.is_integer():
        return int(a) == int(b)
    if b == 0.0:
        return abs(a) < rel_tol
    return abs(a - b) / max(abs(b), 1.0) < rel_tol


def _list_match(pred: str, label: str) -> bool:
    def toset(x: str) -> set[str]:
        return {p.strip().upper() for p in re.split(r"[,\s]+", str(x).strip()) if p.strip()}

    return toset(pred) == toset(label)


def _norm_tikz(s: str) -> str:
    s = re.sub(r"%.*$", "", s, flags=re.MULTILINE)
    s = s.replace(" ,", ",").replace(", ", ",")
    return " ".join(s.strip().lower().split())


def _match(prediction: str, ground_truth: Any, task: str) -> bool:
    label = str(ground_truth).strip()
    if not label:
        return False
    pred = _extract_prediction(prediction, task=task)
    if pred is None:
        return False

    if task in LIST_LETTERS or task in LIST_NUMS:
        return _list_match(pred, label)
    if task in CODE:
        return _norm_tikz(pred) == _norm_tikz(label)
    if task in CHARTS:
        # Pure-numeric label ("0.30") -> numeric compare; structured labels
        # ("(Orange, decreased)", "(5, 0)") -> normalized string equality.
        if _num(label) is not None and _num(pred) is not None:
            return _numeric_match(pred, label)
        return _norm(pred) == _norm(label)
    if task in NUMERIC:
        return _numeric_match(pred, label)
    if task in BINARY:
        a, b = _num(pred), _num(label)
        if a is not None and b is not None:
            return _numeric_match(pred, label, 0.0)
        return _norm(pred) == _norm(label)
    if task in LETTER:
        return _norm(pred).upper() == _norm(label).upper()
    if task in CITY:
        return _norm(pred) == _norm(label)
    return _norm(pred) == _norm(label)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_label_sidecar(path: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path or not Path(path).is_file():
        return mapping
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(item.get("sample_id", ""))
            task = str((item.get("meta") or {}).get("task", ""))
            if sid and task:
                mapping[sid] = task
    return mapping


def exact_score(jsonl: Path, label_map: dict[str, str]) -> dict[str, Any]:
    rows = _load_jsonl(jsonl)
    total = len(rows)
    correct = extractable = err = 0
    per_task: dict[str, dict[str, int]] = {}
    per_row: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("sample_id", ""))
        task = str((row.get("meta") or {}).get("task", ""))
        if not task:
            task = label_map.get(sid, "unknown")
        pred = str(row.get("prediction", "") or "")
        gt = row.get("ground_truth")
        is_err = bool(row.get("error"))
        ext = None if is_err else _extract_prediction(pred, task=task)
        is_match = (not is_err) and gt not in (None, "") and _match(pred, gt, task)
        correct += int(is_match)
        extractable += int(ext is not None)
        err += int(is_err)
        t = per_task.setdefault(task, {"n": 0, "correct": 0, "extractable": 0})
        t["n"] += 1
        t["correct"] += int(is_match)
        t["extractable"] += int(ext is not None)
        per_row.append(
            {
                "sample_id": sid,
                "task": task,
                "prediction": pred,
                "ground_truth": gt,
                "exact": bool(is_match),
                "extractable": ext is not None,
            }
        )
    return {
        "jsonl": str(jsonl),
        "row_count": total,
        "correct": correct,
        "extractable": extractable,
        "errors": err,
        "exact_accuracy_full_denominator": (correct / total) if total else None,
        "extraction_rate": (extractable / total) if total else None,
        "per_task": per_task,
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# llm-as-judge (only for extractable-but-wrong rows)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You grade a multimodal reasoning answer. Compare the model's final answer "
    "to the gold label. Reply with ONLY the single digit 1 (semantically "
    "correct; numeric tolerance and list-order differences allowed) or 0 "
    "(incorrect). No other output."
)


def _judge_one(
    *, answer: str, gold: str, judge_url: str, judge_key: str, judge_model: str, timeout: float
) -> bool | None:
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Model answer: {answer}\nGold label: {gold}"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    request = urllib.request.Request(
        f"{judge_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {judge_key}", "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = str(body.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    digits = re.findall(r"[01]", content)
    if not digits:
        return None  # unparseable verdict -> treat as judge error, not wrong
    return digits[0] == "1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("exact", "judge"), required=True)
    ap.add_argument("--jsonl", action="append", required=True, help="replay remi.jsonl (repeatable)")
    ap.add_argument("--label-jsonl", default=None,
                    help="raw source jsonl; recovers per-sample task when replay dropped meta")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--judge-model", default="Qwen3-VL-32B-Instruct")
    ap.add_argument("--judge-key", default="EMPTY")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int, default=None, help="cap judge drain rows")
    ap.add_argument("--out", default=None, help="sidecar JSONL path for judge decisions")
    args = ap.parse_args()

    try:
        label_map = _load_label_sidecar(args.label_jsonl)
    except OSError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    summaries: list[dict[str, Any]] = []
    for jp in args.jsonl:
        path = Path(jp).expanduser()
        if not path.is_file():
            print(json.dumps({"error": f"not found: {path}"}), file=sys.stderr)
            return 2
        summary = exact_score(path, label_map)

        if args.mode == "judge":
            pending = [r for r in summary["per_row"] if not r["exact"] and r["extractable"]]
            if args.limit is not None:
                pending = pending[: args.limit]
            judged: list[dict[str, Any]] = []

            def work(row: dict[str, Any]) -> dict[str, Any]:
                out = dict(row)
                try:
                    out["judge"] = _judge_one(
                        answer=row.get("prediction", "") or "",
                        gold=str(row.get("ground_truth", "") or ""),
                        judge_url=args.judge_url,
                        judge_key=args.judge_key,
                        judge_model=args.judge_model,
                        timeout=args.timeout,
                    )
                except (OSError, urllib.error.URLError, ValueError, KeyError) as exc:
                    out["judge"] = None
                    out["judge_error"] = f"{type(exc).__name__}: {exc}"
                return out

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                judged = list(pool.map(work, pending))

            judge_correct = sum(1 for r in judged if r.get("judge") is True)
            judge_error = sum(1 for r in judged if r.get("judge") is None)
            summary["judge_pending"] = len(pending)
            summary["judge_correct"] = judge_correct
            summary["judge_error"] = judge_error
            summary["exact_or_judge_accuracy_full_denominator"] = (
                (summary["correct"] + judge_correct) / summary["row_count"]
                if summary["row_count"]
                else None
            )
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in judged) + "\n",
                    encoding="utf-8",
                )
        summary.pop("per_row", None)
        summaries.append(summary)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())