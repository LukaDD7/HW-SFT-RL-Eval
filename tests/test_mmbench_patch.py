from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "lmms_eval_mmbench_tag_first.patch"
SCRIPT = ROOT / "scripts" / "eval" / "apply_mmbench_patch.sh"


def test_patch_is_well_formed_git_diff() -> None:
    text = PATCH.read_text()
    assert text.endswith("\n")
    assert text.startswith("--- a/lmms_eval/tasks/mmbench/mmbench_evals.py\n")

    lines = text.splitlines()
    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    old_count = 0
    new_count = 0
    declared_old = 0
    declared_new = 0
    in_hunk = False

    for line in lines:
        match = hunk_header.match(line)
        if match:
            if in_hunk:
                assert old_count == declared_old
                assert new_count == declared_new
            declared_old = int(match.group(2) or 1)
            declared_new = int(match.group(4) or 1)
            old_count = 0
            new_count = 0
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            if in_hunk:
                assert old_count == declared_old
                assert new_count == declared_new
                in_hunk = False
            continue
        if line.startswith(" ") or not line:
            old_count += 1
            new_count += 1
        elif line.startswith("-"):
            old_count += 1
        elif line.startswith("+"):
            new_count += 1
        else:
            break

    if in_hunk:
        assert old_count == declared_old
        assert new_count == declared_new


def test_patch_contains_tag_first_fix_and_removes_random_fallback() -> None:
    text = PATCH.read_text()
    assert "def extract_final_answer" in text
    assert "Extract only a terminal single-letter answer marker" in text
    assert "Never turn an extraction failure into a random answer." in text
    added_lines = [line[1:] for line in text.splitlines() if line.startswith("+")]
    assert not any("randomly generate one" in line for line in added_lines)


def test_apply_script_uses_git_apply_and_is_idempotent() -> None:
    text = SCRIPT.read_text()
    assert "git -C \"${LMMS_ROOT}\" apply --check" in text
    assert "git -C \"${LMMS_ROOT}\" apply " in text
    assert "patch --forward" not in text
    assert "def extract_final_answer" in text
    assert "terminal answer" in text
