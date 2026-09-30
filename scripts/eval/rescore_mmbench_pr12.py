#!/usr/bin/env python
"""Offline MMBench rescoring with the repository-owned PR #12 evaluator.

This tool intentionally does not regenerate model outputs. It reads the
submission xlsx files saved by a completed MMBench run and feeds their rows to
the patched external lmms-eval evaluator. Rows with a terminal answer marker
are scored directly; unresolved rows are sent to the configured LLM judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore completed MMBench submissions using PR #12 semantics."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="MMBench run directory containing repeat_N/submissions/*.xlsx",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Where to write the compact rescore summary",
    )
    parser.add_argument(
        "--judge-base-url",
        required=True,
        help="OpenAI-compatible judge base URL, e.g. http://127.0.0.1:8004/v1",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen3-VL-32B-Instruct",
        help="Served judge model name (default: Qwen3-VL-32B-Instruct)",
    )
    parser.add_argument(
        "--lmms-eval-root",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/"
            "Dual-Track-OPD/third_party_runtime/lmms-eval"
        ),
        help="Pinned external lmms-eval checkout containing the PR #12 patch",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=4,
        help="Expected number of repeat result files (default: 4)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional arm label stored in the output JSON",
    )
    return parser


def normalize_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def find_submission_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("repeat_*/submissions/*_results.xlsx"))


def load_evaluator(lmms_eval_root: Path, judge_url: str, judge_model: str) -> Any:
    if not (lmms_eval_root / "lmms_eval" / "tasks" / "mmbench" / "mmbench_evals.py").is_file():
        raise FileNotFoundError(
            f"MMBench evaluator not found under {lmms_eval_root}; "
            "pass --lmms-eval-root for the PR #12 patched checkout"
        )

    sys.path.insert(0, str(lmms_eval_root.resolve()))
    try:
        from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
    finally:
        try:
            sys.path.remove(str(lmms_eval_root.resolve()))
        except ValueError:
            pass

    return MMBench_Evaluator(
        sys_prompt="There are several options:",
        API_KEY="EMPTY",
        API_URL=judge_url,
        model_version=judge_model,
    )


def main() -> int:
    args = build_parser().parse_args()
    files = find_submission_files(args.run_dir)
    if not files:
        raise FileNotFoundError(f"No submission xlsx files found under {args.run_dir}")
    if len(files) != args.repeat_count:
        raise ValueError(
            f"Expected {args.repeat_count} submission files, found {len(files)}: "
            + ", ".join(str(path) for path in files)
        )

    import pandas as pd

    judge_url = normalize_chat_url(args.judge_base_url)
    evaluator = load_evaluator(args.lmms_eval_root, judge_url, args.judge_model)

    per_repeat: list[dict[str, Any]] = []
    for path in files:
        rows = pd.read_excel(path).to_dict("records")
        overall, category, l2_category = evaluator.eval_result(
            rows, eval_method="openai"
        )
        per_repeat.append(
            {
                "file": str(path),
                "overall_acc": overall,
                "category_acc": category,
                "l2_category_acc": l2_category,
            }
        )

    values = [item["overall_acc"] for item in per_repeat]
    summary = {
        "scorer": "MMBench PR #12 offline rescore",
        "judge_url": judge_url,
        "judge_model": args.judge_model,
        "lmms_eval_root": str(args.lmms_eval_root),
        "repeat_count": len(files),
        "mean_overall_acc": sum(values) / len(values),
        "per_repeat": per_repeat,
    }
    if args.label is not None:
        summary["label"] = args.label

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
