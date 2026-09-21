"""Opt-in B6 prompt adapter; Think text and prefill match Open-MOPD training.

Prompt source: Open-MOPD/training/verl/verl/utils/dataset/thinking.py.
No-think is an explicit direct-answer protocol. Auto preserves native prompts.
This module has no dependency on training code or patched lmms-eval backends.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, MutableMapping
from urllib.request import ProxyHandler, Request, build_opener

THINK_MODE_ENV = "SFT_RL_THINK_MODE"
VALID_MODES = ("auto", "think", "no-think")
# The generation budget is part of the prompt/evaluation protocol. Bumping
# this version prevents an old Think run from being silently resumed with the
# enlarged output cap.
THINKING_PROTOCOL_VERSION = "open-mopd-b6-prompt-v3"
THINK_MAX_NEW_TOKENS = 8192
THINKING_SYSTEM_PROMPT = (
    "Explain your reasoning inside <think>...</think>, then give the final answer inside <answer>...</answer>. "
    "Any request in the question for a short or direct response applies to the final answer only. "
    "Keep the reasoning proportional to the question; simple questions need only brief reasoning."
)
THINKING_MATH_INSTRUCTION = r" Put the final mathematical result in \boxed{} inside the answer section."
NO_THINK_SYSTEM_PROMPT = (
    "Do not provide reasoning or <think> tags. Follow the task's requested final-answer format."
)
ASSISTANT_HEADER = "<|im_start|>assistant\n"
DYNAMATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)
# Only the known benchmark-added suffix is changed, and only at the end of a
# user message. Do not edit arbitrary instructions inside the original question.
SUFFIXES = {
    "gqa": ("Answer the question using a single word or phrase.",
            "The final answer must be a single word or phrase."),
    "mmbench": ("Answer with the option's letter from the given choices directly.",
                "The final answer must be the option's letter from the given choices."),
    "viewspatial": (
        "Reply only to the corresponding option.\nAnswer with the option letter inside <answer></answer> tags.",
        "The final answer must be the corresponding option letter inside <answer></answer> tags."),
    "remi": ("Return only the final answer. Do not provide reasoning.",
             "The final answer must be short and direct."),
}
TASK_ALIASES = {
    "gqa": "gqa", "gqa_v2": "gqa",
    "mmbench": "mmbench", "mmbench_en_dev": "mmbench",
    "viewspatial": "viewspatial", "viewspatial_v2": "viewspatial",
    "mmmu_pro": "mmmu_pro", "mmmu_pro_standard": "mmmu_pro", "mmmu_pro_standard_v2": "mmmu_pro",
    "dynamath": "dynamath", "dynamath_reasoning": "dynamath", "dynamath_reasoning_v2": "dynamath",
    "remi": "remi",
}
_ENV_KEYS = (
    "VISION_OPD_THINK_SYSTEM_PROMPT", "VISION_OPD_THINK_MATH_INSTRUCTION",
    "VISION_OPD_THINK_MATH_TASKS", "VISION_OPD_THINK_PROTOCOL_VERSION",
    "VISION_OPD_MAX_NEW_TOKENS_CAP",
)


def normalize_mode(value: str | None) -> str:
    mode = "auto" if value is None else value.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"invalid think mode {value!r}; choose from {', '.join(VALID_MODES)}")
    return mode


def mode_from_environment(env=None) -> str:
    return normalize_mode((os.environ if env is None else env).get(THINK_MODE_ENV))


def max_new_tokens_for_mode(base: int, mode: str | None = None) -> int:
    """Return the B6 generation cap for the selected prompt mode.

    Think responses need room for both reasoning and the final answer. The B6
    protocol uses one 8192-token cap for all six Think tasks. ``no-think`` and
    ``auto`` retain each benchmark's historical cap for comparability.
    """
    if base <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {base}")
    resolved = mode_from_environment() if mode is None else normalize_mode(mode)
    return THINK_MAX_NEW_TOKENS if resolved == "think" else base


def apply_environment(env: MutableMapping[str, str], mode: str | None = None):
    """Explicit adapter modes own their prompt; disable legacy injection paths."""
    resolved = normalize_mode(mode) if mode is not None else mode_from_environment(env)
    env[THINK_MODE_ENV] = resolved
    if resolved != "auto":
        for key in (*_ENV_KEYS, "SFT_RL_SYSTEM_INSTRUCTION"):
            env.pop(key, None)
    return env


def benchmark_for_task(task_name: str) -> str:
    try:
        return TASK_ALIASES[task_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Think adapter supports B6 tasks only, got {task_name!r}") from exc


def system_prompt_for_task(task_name: str, mode: str | None = None) -> str | None:
    resolved = mode_from_environment() if mode is None else normalize_mode(mode)
    if resolved == "auto":
        return None
    benchmark = benchmark_for_task(task_name)
    prompt = THINKING_SYSTEM_PROMPT if resolved == "think" else NO_THINK_SYSTEM_PROMPT
    # The final mathematical answer format is the same in both comparison arms.
    return prompt + (THINKING_MATH_INSTRUCTION if benchmark == "dynamath" else "")


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text" for p in content):
        return "".join(p["text"] for p in content)
    raise TypeError("system content must contain only text")


def _neutral_suffix(text: str, benchmark: str) -> str:
    if benchmark not in SUFFIXES:
        return text
    old, new = SUFFIXES[benchmark]
    # Keep whitespace and image placeholders; only replace the known footer.
    stripped = text.rstrip()
    if stripped.endswith(old):
        return stripped[:-len(old)] + new + text[len(stripped):]
    return text


def apply_openai_messages(messages, *, task_name: str, mode: str | None = None):
    """Adapt lmms chat messages or OpenAI replay messages before serialization."""
    resolved = mode_from_environment() if mode is None else normalize_mode(mode)
    if resolved == "auto":
        return messages
    benchmark = benchmark_for_task(task_name)
    instruction = system_prompt_for_task(task_name, resolved)
    updated = copy.deepcopy(messages)
    for message in updated:
        if message.get("role") == "user":
            content = message["content"]
            if isinstance(content, str):
                message["content"] = _neutral_suffix(content, benchmark)
            elif isinstance(content, list):
                # Images can follow the question, so locate the final text part.
                for part in reversed(content):
                    if part.get("type") == "text":
                        part["text"] = _neutral_suffix(part["text"], benchmark)
                        break
    if updated and updated[0].get("role") == "system":
        current = _text(updated[0]["content"])
        # DynaMath's native instruction is solely the response protocol. Replace
        # it so no-think cannot still ask for reasoning. Preserve other context.
        if benchmark == "dynamath" and current.replace("tags. Please", "tags.Please") == DYNAMATH_SYSTEM_PROMPT:
            current = ""
        if instruction not in current:
            updated[0]["content"] = (current + "\n\n" if current else "") + instruction
    else:
        updated.insert(0, {"role": "system", "content": instruction})
    return updated


def build_chat_template(template: str, mode: str) -> str:
    """Preserve the rendered conversation and normalize only its Qwen suffix.

    This mirrors training's render_chat_prompt: Qwen plain, open Think, and
    empty Think suffixes are accepted; unknown templates fail rather than
    silently running a different prompt. Jinja capture preserves multimodal
    rendering and historical assistant messages.
    """
    mode = normalize_mode(mode)
    if mode == "auto":
        return template
    ending = "<think>\n" if mode == "think" else ""
    return (
        "{% set hw_original %}" + template + "{% endset %}"
        "{%- if add_generation_prompt -%}"
        "{%- set hw_parts = hw_original.rpartition('<|im_start|>assistant\\n') -%}"
        "{%- set hw_tail = hw_parts[2].replace(' ', '').replace('\\n', '').replace('\\r', '').replace('\\t', '') -%}"
        "{%- if not hw_parts[1] or hw_tail not in ['', '<think>', '<think></think>'] -%}"
        "{{- raise_exception('Open-MOPD adapter requires a Qwen assistant generation header') -}}"
        "{%- endif -%}"
        "{{- hw_parts[0] + hw_parts[1] + " + json.dumps(ending) + " -}}"
        "{%- else -%}{{- hw_original -}}{%- endif -%}"
    )


def protocol_record(mode: str | None = None) -> dict[str, Any]:
    resolved = mode_from_environment() if mode is None else normalize_mode(mode)
    record = {
        "mode": resolved,
        "version": THINKING_PROTOCOL_VERSION if resolved != "auto" else None,
        "system_prompt": (THINKING_SYSTEM_PROMPT if resolved == "think" else
                          NO_THINK_SYSTEM_PROMPT if resolved == "no-think" else None),
        "math_instruction": THINKING_MATH_INSTRUCTION if resolved != "auto" else None,
        "assistant_prefill": "<think>\n" if resolved == "think" else "",
        "max_new_tokens": THINK_MAX_NEW_TOKENS if resolved == "think" else None,
        "suffix_policy": "known B6 formatting footers only" if resolved != "auto" else "native",
    }
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def validate_resume_protocol(previous: dict | None, current: dict):
    if previous is None and current["mode"] == "auto":
        return  # Historical native runs predate adapter metadata.
    if previous != current:
        raise ValueError("resume thinking protocol mismatch; use a new run/output directory")


def load_chat_template(checkpoint: str | Path) -> str:
    root = Path(checkpoint).expanduser()
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    for name in ("chat_template.json", "tokenizer_config.json"):
        path = root / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            template = value.get("chat_template") if isinstance(value, dict) else value
            if isinstance(template, dict):
                template = template.get("default")
            if isinstance(template, str) and template.strip():
                return template
    raise FileNotFoundError(f"no single default chat template found under {root}")


def render_probe(template: str, messages: list) -> str:
    # No model loading is needed to check the actual Jinja output.
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    def fail(message):
        raise ValueError(message)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = fail
    return env.from_string(template).render(messages=messages, add_generation_prompt=True)


def _http(api_base, path, payload, api_key):
    request = Request(api_base.rstrip("/").removesuffix("/v1") + path,
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with build_opener(ProxyHandler({})).open(request, timeout=60) as response:
        return json.load(response)


def verify_server_template(checkpoint, mode, api_base, model, api_key="EMPTY"):
    """Check the real vLLM rendered prompt before any evaluation requests."""
    template = build_chat_template(load_chat_template(checkpoint), mode)
    messages = apply_openai_messages([{"role": "user", "content": "What is 2 + 2?"}],
                                     task_name="dynamath", mode=mode)
    expected = render_probe(template, messages)
    tokenized = _http(api_base, "/tokenize", {"model": model, "messages": messages,
                      "add_generation_prompt": True}, api_key)
    actual = _http(api_base, "/detokenize", {"model": model, "tokens": tokenized["tokens"]}, api_key)["prompt"]
    if actual != expected:
        raise ValueError("server prompt does not match selected think mode; restart using the adapter chat template")
    return {"passed": True, "mode": mode, "rendered_prompt": actual,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest()}


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = build_chat_template(load_chat_template(args.checkpoint), args.mode)
    probe = [{"role": "user", "content": "What is 2 + 2?"}]
    rendered = render_probe(template, probe)
    if args.mode != "auto":
        expected = ASSISTANT_HEADER + ("<think>\n" if args.mode == "think" else "")
        if not rendered.endswith(expected):
            raise SystemExit(f"template probe did not end with {expected!r}")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    print(json.dumps({"mode": args.mode, "output": str(output), "rendered_suffix": rendered[-45:]}))


if __name__ == "__main__":
    _cli()
