"""Run the pinned lmms OpenAI chat backend through the repository B6 adapter.

Intercept only the document-to-messages boundary, before the existing backend
serializes images and sends the API request. Preserve each request's configured
budget at the SDK boundary after the pinned backend's 4096-token clamp, in all
prompt modes including native auto. Scoring stays in
lmms-eval; no edits to the external checkout are needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path

from .thinking_adapter import (
    apply_environment, apply_openai_messages,
    mode_from_environment, protocol_record,
)


def _sqlite_connect(path: Path):
    db = sqlite3.connect(path, timeout=120, check_same_thread=False)
    db.execute("PRAGMA busy_timeout=120000")
    # Several four-GPU shards can open the same durable cache at the same
    # instant.  Switching a fresh database into WAL mode takes a brief write
    # lock; retry instead of letting one benchmark process die immediately.
    for attempt in range(120):
        try:
            db.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError:
            if attempt == 119:
                raise
            time.sleep(0.25)
    db.execute("PRAGMA synchronous=NORMAL")
    for attempt in range(120):
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "cache_key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            db.commit()
            break
        except sqlite3.OperationalError:
            if attempt == 119:
                raise
            time.sleep(0.25)
    return db


_PAYLOAD_KEY_FIELDS = (
    "messages", "model", "max_tokens", "max_completion_tokens", "temperature",
    "top_p", "top_k", "seed", "n", "best_of", "num_return_sequences",
)


def _payload_cache_key(payload: dict, mode: str, seed: int) -> str:
    canonical = {key: payload[key] for key in _PAYLOAD_KEY_FIELDS if key in payload}
    canonical["mode"] = mode
    canonical["seed"] = seed
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _request_payload(request, model: str) -> dict:
    context, converter, kwargs, doc_id, task, split = request.args
    try:
        converted = converter(doc_id)
        messages = converted
    except Exception:
        messages = str(context)
    payload = {"messages": messages, "model": model}
    for key in ("temperature", "top_p", "top_k", "n", "best_of", "num_return_sequences"):
        if key in kwargs:
            payload[key] = kwargs[key]
    if "max_new_tokens" in kwargs:
        payload["max_tokens"] = kwargs["max_new_tokens"]
    return payload


def _request_cache_key(request, mode: str, seed: int, model: str) -> str:
    return _payload_cache_key(
        _request_payload(request, model),
        mode,
        seed,
    )


def _cache_record(response):
    if not hasattr(response, "choices"):
        return {"text": response}
    choice = response.choices[0]
    text = choice.message.content
    usage = getattr(response, "usage", None)
    return {
        "text": text,
        "token_counts": {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                0,
            ) or 0,
        },
    }


def _generation_result(value):
    if isinstance(value, dict):
        from lmms_eval.api.instance import GenerationResult, TokenCounts
        return GenerationResult(
            text=value.get("text", ""),
            token_counts=TokenCounts(**value.get("token_counts", {})),
        )
    return value


class DurableResponses:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = _sqlite_connect(path)
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT response FROM responses WHERE cache_key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, response) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO responses(cache_key,response,created_at) VALUES(?,?,?)",
                (key, json.dumps(response, ensure_ascii=False, default=str), time.time()),
            )
            self.db.commit()


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
    original = getattr(backend, "_lmms_thinking_original_generate_until", None)
    if original is None:
        original = backend.generate_until
        backend._lmms_thinking_original_generate_until = original
    cache_root = os.environ.get("HW_EVAL_DURABLE_RESPONSES")
    durable = DurableResponses(Path(cache_root).expanduser()) if cache_root else None
    seed = int(os.environ.get("EVAL_SEED", os.environ.get("SFT_RL_SEED", "42")))

    def run_original(self, selected_requests):
        groups = {}
        for index, request in enumerate(selected_requests):
            budget = int(request.args[2].get("max_new_tokens", 1024))
            if budget <= 0:
                raise ValueError(f"max_new_tokens must be positive, got {budget}")
            groups.setdefault(budget, []).append((index, request))

        completions = self.client.chat.completions
        create = completions.create
        responses = [None] * len(selected_requests)
        try:
            for budget, indexed_requests in groups.items():
                @wraps(create)
                def create_with_budget_and_cache(*args, **payload):
                    token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                    payload[token_key] = budget
                    response = create(*args, **payload)
                    if durable is not None:
                        durable.put(
                            _payload_cache_key(payload, mode, seed),
                            _cache_record(response),
                        )
                    return response

                completions.create = create_with_budget_and_cache
                results = original(self, [request for _, request in indexed_requests])
                if len(results) != len(indexed_requests):
                    raise RuntimeError("lmms backend returned an unexpected number of responses")
                for (index, _), result in zip(indexed_requests, results):
                    responses[index] = result
        finally:
            completions.create = create
        return responses

    @wraps(original)
    def generate_until(self, requests):
        adapted = adapt_requests(requests, mode)
        if not adapted:
            return []
        if durable is not None:
            missing = []
            cached_responses = [None] * len(adapted)
            for index, request in enumerate(adapted):
                key = _request_cache_key(request, mode, seed, self.model_version)
                response = durable.get(key)
                cached_responses[index] = response
                if response is None:
                    missing.append((index, request, key))
            if not missing:
                return [_generation_result(response) for response in cached_responses]
            missing_requests = [request for _, request, _ in missing]
            new_responses = run_original(self, missing_requests)
            for (index, _, key), response in zip(missing, new_responses):
                cached_responses[index] = response
            return [_generation_result(response) for response in cached_responses]
        return run_original(self, adapted)

    backend.generate_until = generate_until
    return backend


def main():
    install_adapter()
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    main()
