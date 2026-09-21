"""Run the pinned lmms OpenAI chat backend through the repository B6 adapter.

Intercept only the document-to-messages boundary, before the existing backend
serializes images and sends the API request. Apply the Think budget at the
SDK boundary after the pinned backend's 4096-token clamp. Scoring stays in
lmms-eval; no edits to the external checkout are needed.
"""
from __future__ import annotations

import copy
import json
import os
from functools import wraps
from pathlib import Path

from .thinking_adapter import (
    THINK_MAX_NEW_TOKENS, apply_environment, apply_openai_messages,
    mode_from_environment, protocol_record,
)


def adapt_requests(requests, mode):
    adapted = []
    for request in requests:
        context, converter, kwargs, doc_id, task, split = request.args
        if not callable(converter):
            raise TypeError("Think adapter requires the lmms OpenAI chat request interface")

        def adapted_converter(doc, original=converter, task_name=task):
            messages = apply_openai_messages(original(doc), task_name=task_name, mode=mode)
            if directory := os.environ.get("HW_EVAL_PROMPT_AUDIT_DIR"):
                path = Path(directory) / f"{task_name}.json"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    preview = copy.deepcopy(messages)
                    for message in preview:
                        if isinstance(message["content"], list):
                            message["content"] = [p if p.get("type") == "text" else
                                                   {"type": p.get("type"), "media": "omitted"}
                                                   for p in message["content"]]
                    path.write_text(json.dumps({"protocol": protocol_record(mode), "messages": preview},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
            return messages

        updated = copy.copy(request)
        updated.arguments = (context, adapted_converter, kwargs, doc_id, task, split)
        adapted.append(updated)
    return adapted


def install_adapter():
    from lmms_eval.models import get_model
    backend = get_model("openai")
    if backend.is_simple or backend.__module__ != "lmms_eval.models.chat.openai":
        raise RuntimeError("Think adapter requires the pinned lmms OpenAI chat backend")
    apply_environment(os.environ)
    mode = mode_from_environment()
    if mode == "auto":
        raise ValueError("Native auto mode should invoke lmms_eval directly")
    original = backend.generate_until

    @wraps(original)
    def generate_until(self, requests):
        adapted = adapt_requests(requests, mode)
        if mode != "think":
            return original(self, adapted)

        # The pinned chat backend clamps max_new_tokens to 4096 even when the
        # task requests 8192. Override only this evaluation client's outgoing
        # payload, retaining the backend's image serialization and retries.
        completions = self.client.chat.completions
        create = completions.create

        @wraps(create)
        def create_with_think_budget(*args, **payload):
            key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
            payload[key] = THINK_MAX_NEW_TOKENS
            return create(*args, **payload)

        completions.create = create_with_think_budget
        try:
            # generate_until joins its worker threads before returning.
            return original(self, adapted)
        finally:
            completions.create = create

    backend.generate_until = generate_until
    return backend


def main():
    install_adapter()
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    main()
