"""Prompt-mode regression tests: conflicts, actual Jinja render, and run isolation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dual_track_opd.eval.thinking_adapter import (
    ASSISTANT_HEADER, DYNAMATH_SYSTEM_PROMPT, NO_THINK_SYSTEM_PROMPT,
    SUFFIXES, THINKING_MATH_INSTRUCTION, THINKING_SYSTEM_PROMPT,
    apply_environment, apply_openai_messages, build_chat_template,
    load_chat_template, max_new_tokens_for_mode, normalize_mode,
    protocol_record, render_probe, validate_resume_protocol,
)

PLAIN = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"""
HARD_THINK = PLAIN.replace("assistant\\n'", "assistant\\n<think>\\n'")
CONDITIONAL = PLAIN.replace("assistant\\n'", "assistant\\n'") + """
{%- if add_generation_prompt %}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"""


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("think", 8192), ("no-think", 4096), ("auto", 4096)],
)
def test_think_generation_cap_is_fixed_for_think_mode(mode, expected):
    assert max_new_tokens_for_mode(4096, mode) == expected


@pytest.mark.parametrize("base", [1024, 2048, 4096])
def test_all_b6_think_caps_are_8192(base):
    assert max_new_tokens_for_mode(base, "think") == 8192


def test_think_protocol_records_generation_cap_policy():
    record = protocol_record("think")
    assert record["version"] == "open-mopd-b6-prompt-v3"
    assert record["max_new_tokens"] == 8192


@pytest.mark.parametrize("source", [PLAIN, HARD_THINK, CONDITIONAL])
@pytest.mark.parametrize("mode", ["think", "no-think"])
def test_render_changes_only_generation_suffix(source, mode):
    messages = [{"role": "system", "content": "Instructions"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "<think>Past reasoning</think>Earlier answer"},
                {"role": "user", "content": "Next question"}]
    original = render_probe(source, messages)
    expected = original.rpartition(ASSISTANT_HEADER)[0] + ASSISTANT_HEADER
    expected += "<think>\n" if mode == "think" else ""
    assert render_probe(build_chat_template(source, mode), messages) == expected


def test_unknown_template_fails():
    with pytest.raises(ValueError, match="Qwen assistant"):
        render_probe(build_chat_template("USER: question ASSISTANT:", "think"), [])


def test_auto_preserves_native_protocol():
    messages = [{"role": "system", "content": DYNAMATH_SYSTEM_PROMPT}]
    assert apply_openai_messages(messages, task_name="dynamath", mode="auto") is messages
    assert build_chat_template(HARD_THINK, "auto") == HARD_THINK
    env = {"SFT_RL_SYSTEM_INSTRUCTION": "historical override"}
    apply_environment(env, "auto")
    assert env["SFT_RL_SYSTEM_INSTRUCTION"] == "historical override"


@pytest.mark.parametrize("mode", ["think", "no-think"])
def test_adapter_clears_legacy_injection(mode):
    env = {"SFT_RL_SYSTEM_INSTRUCTION": "stale",
           "VISION_OPD_THINK_SYSTEM_PROMPT": "stale", "VISION_OPD_THINK_MATH_TASKS": "stale",
           "VISION_OPD_MAX_NEW_TOKENS_CAP": "4096"}
    apply_environment(env, mode)
    assert env == {"SFT_RL_THINK_MODE": mode}


@pytest.mark.parametrize("mode", ["think", "no-think"])
@pytest.mark.parametrize("task", ["gqa", "mmbench", "viewspatial", "mmmu_pro", "dynamath", "remi"])
def test_six_messages_are_conflict_free_and_preserve_media(task, mode):
    question = "Original question, A. 1 B. 2."
    suffix = SUFFIXES.get(task, ("Answer with the option letter inside <answer></answer> tags.", ""))[0]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": question + "\n\n" + suffix},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,TEST"}},
    ]}]
    if task == "dynamath":
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": DYNAMATH_SYSTEM_PROMPT}]})
    saved = copy.deepcopy(messages)
    result = apply_openai_messages(messages, task_name=task, mode=mode)
    expected = THINKING_SYSTEM_PROMPT if mode == "think" else NO_THINK_SYSTEM_PROMPT
    if task == "dynamath":
        expected += THINKING_MATH_INSTRUCTION
    assert result[0] == {"role": "system", "content": expected}
    assert result[-1]["content"][0]["text"].startswith(question)
    assert result[-1]["content"][1] == messages[-1]["content"][1]
    assert "Do not provide reasoning." not in result[-1]["content"][0]["text"]
    assert "response must include two parts" not in result[0]["content"]
    assert messages == saved
    assert apply_openai_messages(result, task_name=task, mode=mode) == result


def test_original_question_is_not_rewritten():
    # A formatting phrase embedded within a question is not a benchmark footer.
    text = 'Quote "Return only the final answer. Do not provide reasoning." and explain its meaning.'
    result = apply_openai_messages([{"role": "user", "content": text}], task_name="remi", mode="think")
    assert result[-1]["content"] == text


def test_resume_requires_same_mode_and_protocol():
    validate_resume_protocol(None, protocol_record("auto"))
    validate_resume_protocol(protocol_record("think"), protocol_record("think"))
    for previous in (None, protocol_record("auto"), protocol_record("no-think")):
        with pytest.raises(ValueError, match="resume thinking protocol mismatch"):
            validate_resume_protocol(previous, protocol_record("think"))


def test_load_checkpoint_template_variants(tmp_path):
    config = tmp_path / "chat_template.json"
    config.write_text(json.dumps({"chat_template": PLAIN}))
    assert load_chat_template(tmp_path) == PLAIN
    (tmp_path / "chat_template.jinja").write_text(HARD_THINK)
    assert load_chat_template(tmp_path) == HARD_THINK


def test_invalid_mode_and_unsupported_tasks_fail():
    with pytest.raises(ValueError, match="invalid think mode"):
        normalize_mode("thnik")
    with pytest.raises(ValueError, match="B6 tasks only"):
        apply_openai_messages([], task_name="mv_math", mode="think")
