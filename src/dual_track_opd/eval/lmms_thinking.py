"""Run the pinned lmms OpenAI chat backend through the repository B6 adapter.

Intercept only the document-to-messages boundary, before the existing backend
serializes images and sends the API request. Preserve each request's configured
budget at the SDK boundary after the pinned backend's 4096-token clamp, in all
prompt modes including native auto. Scoring stays in
lmms-eval; no edits to the external checkout are needed.
"""
from __future__ import annotations

import copy
import json
import os
from functools import wraps
from pathlib import Path

from .thinking_adapter import (
    apply_environment, apply_openai_messages,
    mode_from_environment, protocol_record,
)


def adapt_requests(requests, mode):
    if mode == "auto":
        return list(requests)
    adapted = []
    for request in requests:
        context, converter, kwargs, doc_id, task, split = request.args
        if not callable(converter):
            raise TypeError("Think adapter requires the lmms OpenAI chat request interface")

        def adapted_converter(doc, original=converter, task_name=task):
            messages = apply_openai_messages(original(doc), task_name=task_name, mode=mode)
            # lmms ChatMessages requires typed content lists even for system
            # text; the direct OpenAI replay API also accepts plain strings.
            for message in messages:
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
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
    original = backend.generate_until

    @wraps(original)
    def generate_until(self, requests):
        adapted = adapt_requests(requests, mode)
        if not adapted:
            return []
        # A suite process normally evaluates one task. Group mixed-task calls
        # by budget as well, so concurrent SDK calls cannot borrow another
        # task's cap. Restore results to the caller's original order.
        groups = {}
        for index, request in enumerate(adapted):
            budget = int(request.args[2].get("max_new_tokens", 1024))
            if budget <= 0:
                raise ValueError(f"max_new_tokens must be positive, got {budget}")
            groups.setdefault(budget, []).append((index, request))

        completions = self.client.chat.completions
        create = completions.create
        responses = [None] * len(adapted)
        try:
            for budget, indexed_requests in groups.items():
                @wraps(create)
                def create_with_budget(*args, **payload):
                    key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                    payload[key] = budget
                    return create(*args, **payload)

                completions.create = create_with_budget
                # generate_until joins its worker threads before returning.
                results = original(self, [request for _, request in indexed_requests])
                if len(results) != len(indexed_requests):
                    raise RuntimeError("lmms backend returned an unexpected number of responses")
                for (index, _), result in zip(indexed_requests, results):
                    responses[index] = result
            return responses
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
