import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_remi_strict_four_repeat_full_denominator_mean(tmp_path: Path) -> None:
    replay_paths = []
    for repeat_index in range(4):
        path = tmp_path / f"remi.repeat_{repeat_index}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample_index in range(2600):
                row = {
                    "sample_id": f"sample-{sample_index}",
                    "meta": {"task": "GeomShape"},
                    "prediction": "1" if sample_index < repeat_index else "2",
                    "ground_truth": "1",
                    "error": "",
                }
                handle.write(json.dumps(row) + "\n")
        replay_paths.append(path)

    output = tmp_path / "remi_avg4.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/sft_rl/remi_reeval.py"),
        "--mode",
        "exact",
        "--aggregate-mean",
        "--out",
        str(output),
    ]
    for path in replay_paths:
        command.extend(["--jsonl", str(path)])
    result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["mode"] == "exact"
    assert aggregate["metric"] == "exact_accuracy_full_denominator"
    assert aggregate["repeat_count"] == 4
    assert aggregate["row_counts"] == [2600] * 4
    assert aggregate["correct_counts"] == [0, 1, 2, 3]
    assert aggregate["values"] == [index / 2600 for index in range(4)]
    assert aggregate["mean"] == sum(index / 2600 for index in range(4)) / 4
