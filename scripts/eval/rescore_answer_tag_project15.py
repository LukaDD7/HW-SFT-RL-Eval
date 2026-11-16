#!/usr/bin/env python3
"""Offline answer-tag rescoring for explicit-think Project15 runs.

The original offline task YAMLs use the upstream evaluators directly. Those
evaluators do not recognize the Open-MOPD `<answer>...</answer>` tag, so an
explicit-think response can be scored as its full reasoning text. This tool
extracts the final answer tag from logged sample responses and re-applies the
upstream process/aggregate functions without regenerating model requests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
import glob
from pathlib import Path

import pyarrow.parquet as pq


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str) -> str:
    matches = list(ANSWER_TAG_RE.finditer(text or ""))
    if not matches:
        return str(text or "").strip()
    return matches[-1].group(1).strip()


def load_samples(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_audit(path: Path, task_name: str):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Response is a JSON-encoded string in the audit log.
            response = row.get("response", "")
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    pass
            if row.get("task_name") != task_name:
                continue
            yield {"doc_id": row.get("doc_id"), "filtered_resps": response}


def load_docs(*files: str, split: str = "test"):
    if not files:
        return {}
    paths = []
    for pattern in files:
        paths.extend(glob.glob(pattern))
    docs = {}
    for path in sorted(paths):
        table = pq.read_table(path)
        for index, row in enumerate(table.to_pylist()):
            docs[len(docs) + index] = row
    return docs


def metric_row(benchmark: str, value: float, samples: int, source: str):
    return {
        "benchmark_id": benchmark,
        "primary_metric": {
            "vqav2": "exact_match",
            "scienceqa": "exact_match",
            "mindcube": "overall_accuracy",
            "mmsi_bench": "average",
            "blink": "blink_acc",
        }[benchmark],
        "value": value,
        "samples": samples,
        "source": source,
    }


def rescore_vqav2(path: Path):
    from lmms_eval.tasks.vqav2.utils import (
        EvalAIAnswerProcessor, vqav2_process_results_val,
    )
    scores = []
    tagged = 0
    docs = load_docs(
        "assets/datasets/VQAv2/data/validation-*.parquet", split="validation"
    )
    for sample in load_audit(path, "vqav2_val_offline"):
        raw = str(sample.get("filtered_resps", ""))
        pred = extract_final_answer(raw)
        tagged += ANSWER_TAG_RE.search(raw) is not None
        # Sample logs omit the full document; recover the official ten answer
        # records from the input field recorded by the OpenAI adapter.
        doc = docs.get(int(sample.get("doc_id", -1)))
        if doc is None or "answers" not in doc:
            continue
        # The logged target is the field name for this task. Replace it only
        # in this in-memory document so the upstream scorer sees real answers.
        result = vqav2_process_results_val(doc, [pred])["exact_match"]
        scores.append(float(result))
    if not scores:
        raise ValueError(f"no VQAv2 rows with recoverable answers in {path}")
    return metric_row("vqav2", sum(scores) / len(scores), len(scores), str(path)), tagged


def rescore_scienceqa(path: Path):
    from lmms_eval.tasks.scienceqa.utils import sqa_process_results
    scores = []
    tagged = 0
    docs = load_docs("assets/datasets/ScienceQA-IMG/data/test-*.parquet")
    for sample in load_audit(path, "scienceqa_img_offline"):
        raw = str(sample.get("filtered_resps", ""))
        pred = extract_final_answer(raw)
        tagged += ANSWER_TAG_RE.search(raw) is not None
        doc = docs.get(int(sample.get("doc_id", -1)))
        if doc is None or "answer" not in doc or "choices" not in doc:
            continue
        scores.append(float(sqa_process_results(doc, [pred])["exact_match"]))
    if not scores:
        raise ValueError(f"no ScienceQA rows with recoverable documents in {path}")
    return metric_row("scienceqa", sum(scores) / len(scores), len(scores), str(path)), tagged


def rescore_blink(path: Path):
    from lmms_eval.tasks.blink.utils import blink_process_results, blink_aggregate_results
    processed = []
    tagged = 0
    docs = load_docs("assets/datasets/BLINK/*/val-*.parquet")
    for sample in load_audit(path, "blink_offline"):
        raw = str(sample.get("filtered_resps", ""))
        pred = extract_final_answer(raw)
        tagged += ANSWER_TAG_RE.search(raw) is not None
        doc = docs.get(int(sample.get("doc_id", -1)))
        if doc is None or "answer" not in doc:
            continue
        processed.append(blink_process_results(doc, [pred])["blink_acc"])
    return metric_row("blink", blink_aggregate_results(processed), len(processed), str(path)), tagged


def rescore_mindcube(path: Path):
    from lmms_eval.tasks.mindcube.utils import (
        mindcube_aggregate_results, mindcube_process_results,
    )
    processed = defaultdict(list)
    tagged = 0
    docs = load_docs("assets/datasets/MindCube_lmmseval/full/combined-*.parquet")
    for sample in load_audit(path, "mindcube_full_offline"):
        raw = str(sample.get("filtered_resps", ""))
        pred = extract_final_answer(raw)
        tagged += ANSWER_TAG_RE.search(raw) is not None
        doc = docs.get(int(sample.get("doc_id", -1)))
        if doc is None or "gt_answer" not in doc:
            continue
        result = mindcube_process_results(doc, [pred])
        for metric, value in result.items():
            processed[metric].append(value)
    # mindcube_aggregate_results expects each metric's result list separately.
    overall = mindcube_aggregate_results(processed["overall_accuracy"])
    return metric_row("mindcube", overall, len(processed["overall_accuracy"]), str(path)), tagged


def rescore_mmsi(path: Path):
    from lmms_eval.tasks.mmsi_bench.utils import (
        msr_aggregate_results, msr_process_results,
    )
    processed = []
    tagged = 0
    docs = load_docs("assets/datasets/MMSI-Bench/MMSI_Bench.parquet")
    for sample in load_audit(path, "mmsi_bench_offline"):
        raw = str(sample.get("filtered_resps", ""))
        pred = extract_final_answer(raw)
        tagged += ANSWER_TAG_RE.search(raw) is not None
        doc = docs.get(int(sample.get("doc_id", -1)))
        if doc is None or "answer" not in doc:
            continue
        values = msr_process_results(doc, [pred])
        processed.append(values["average"])
    return metric_row("mmsi_bench", msr_aggregate_results(processed), len(processed), str(path)), tagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqav2-samples")
    parser.add_argument("--scienceqa-samples")
    parser.add_argument("--mindcube-samples")
    parser.add_argument("--mmsi-samples")
    parser.add_argument("--blink-samples")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    diagnostics = {}
    jobs = (
        ("vqav2", args.vqav2_samples, rescore_vqav2),
        ("scienceqa", args.scienceqa_samples, rescore_scienceqa),
        ("mindcube", args.mindcube_samples, rescore_mindcube),
        ("mmsi_bench", args.mmsi_samples, rescore_mmsi),
        ("blink", args.blink_samples, rescore_blink),
    )
    for benchmark, value, function in jobs:
        if not value:
            continue
        row, tagged = function(Path(value).expanduser())
        rows.append(row)
        diagnostics[benchmark] = {
            "answer_tag_rows": tagged,
            "total_rows": row["samples"],
        }

    output = {
        "protocol": "offline answer-tag extraction followed by upstream evaluator",
        "rows": rows,
        "diagnostics": diagnostics,
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
