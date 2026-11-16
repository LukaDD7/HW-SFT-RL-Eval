"""Score ReMI replay generations with a strict full-denominator receipt."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REMI_SCORER = Path(__file__).resolve().parents[3] / "scripts/sft_rl/remi_reeval.py"
REMI_LABELS = (
    Path(__file__).resolve().parents[3]
    / "assets/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl"
)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_remi_receipt(receipt: dict, repeats: int, *, smoke: bool) -> None:
    expected_rows = 8 if smoke else 2600
    if receipt.get("mode") != "exact" or receipt.get("metric") != "exact_accuracy_full_denominator":
        raise ValueError("ReMI receipt is not the exact full-denominator protocol")
    if receipt.get("repeat_count") != repeats:
        raise ValueError(f"ReMI repeat mismatch: {receipt.get('repeat_count')} != {repeats}")
    if receipt.get("row_counts") != [expected_rows] * repeats:
        raise ValueError(f"ReMI requires {expected_rows} rows per repeat: {receipt.get('row_counts')}")
    mean = receipt.get("mean")
    values = receipt.get("values")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool) or not math.isfinite(mean):
        raise ValueError("ReMI mean is missing or non-finite")
    if (not isinstance(values, list) or len(values) != repeats
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values)):
        raise ValueError("ReMI repeat values are incomplete or non-finite")
    if not math.isclose(float(mean), sum(values) / len(values), rel_tol=0, abs_tol=1e-12):
        raise ValueError("ReMI mean does not match per-repeat values")


def score_remi_exact(
    replays: Sequence[Path], *, output: Path, smoke: bool, labels: Path = REMI_LABELS
) -> dict:
    """Run the pinned scorer, then validate its persisted result, not just its exit code."""
    if not replays:
        raise ValueError("ReMI requires at least one replay")
    command = [sys.executable, str(REMI_SCORER), "--mode", "exact"]
    for replay in replays:
        if replay.is_symlink() or not replay.is_file() or not replay.stat().st_size:
            raise FileNotFoundError(f"ReMI replay must be a regular nonempty file: {replay}")
        command.extend(("--jsonl", str(replay.resolve())))
    if labels.is_symlink() or not labels.is_file():
        raise FileNotFoundError(f"ReMI labels must be a regular file: {labels}")
    command.extend(("--label-jsonl", str(labels.resolve()), "--aggregate-mean"))
    if len(replays) > 1:
        result_path = output.with_name("remi_avg4.json")
    else:
        result_path = output
    command.extend(("--out", str(result_path)))
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"ReMI scorer exited with status {completed.returncode}")
    if not result_path.is_file():
        raise RuntimeError(f"ReMI scorer returned zero without a result: {result_path}")
    receipt = load_json(result_path)
    validate_remi_receipt(receipt, len(replays), smoke=smoke)
    formal = not smoke
    # Existing result JSONs are immutable scorer output. A sibling receipt records
    # whether this invocation is smoke or full, so an eight-row score cannot leak
    # into a formal result table during a later summary/resume.
    if receipt.get("formal") not in (None, formal):
        raise ValueError("ReMI result belongs to a different smoke/full protocol")
    audit = dict(receipt, formal=formal, scoring_command=command)
    audit_path = output.with_name("remi_scoring.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=REMI_LABELS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    audit = score_remi_exact(args.replay, output=args.out, smoke=args.smoke, labels=args.labels)
    print(json.dumps({"status": "complete", "formal": audit["formal"], "receipt": str(args.out)}))


if __name__ == "__main__":
    main()
