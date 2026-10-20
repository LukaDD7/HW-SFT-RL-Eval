"""Check the SDK payload after a backend applies its historical token clamp."""
from concurrent.futures import ThreadPoolExecutor
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dual_track_opd.eval.lmms_thinking import adapt_requests, install_adapter
from dual_track_opd.eval.lmms_thinking import DurableResponses, _request_cache_key
from dual_track_opd.eval.thinking_adapter import THINKING_SYSTEM_PROMPT


@pytest.mark.parametrize("mode", ["auto", "think", "no-think"])
@pytest.mark.parametrize("expected", [8192, 16384])
@pytest.mark.parametrize("token_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("fails", [False, True])
def test_sdk_budget_survives_backend_clamp_and_is_restored(monkeypatch, mode, expected, token_key, fails):
    sent = []

    def create(**payload):
        sent.append(payload)
        return "answer"

    class Backend:
        is_simple = False

        def generate_until(self, requests):
            # Reproduce the pinned backend's payload creation and concurrent
            # SDK calls without importing the GPU evaluation environment.
            def send(request):
                _, converter, kwargs, *_ = request.arguments
                return self.client.chat.completions.create(
                    messages=converter({}),
                    model="qwen-test", temperature=1.0, seed=44,
                    **{token_key: min(kwargs["max_new_tokens"], 4096)},
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(send, requests))
            if fails:
                raise RuntimeError("backend failed")
            return results

    Backend.__module__ = "lmms_eval.models.chat.openai"
    models = ModuleType("lmms_eval.models")
    models.get_model = lambda name: Backend
    monkeypatch.setitem(sys.modules, "lmms_eval.models", models)
    monkeypatch.setenv("SFT_RL_THINK_MODE", mode)
    install_adapter()
    backend = Backend()
    completions = SimpleNamespace(create=create)
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    media = {"type": "image_url", "image_url": {"url": "fixture-image"}}
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Question"}, media,
    ]}]
    args = ("context", lambda doc: messages, {"max_new_tokens": expected}, 0, "gqa_v2", "test")
    request = SimpleNamespace(args=args, arguments=args)
    if fails:
        with pytest.raises(RuntimeError, match="backend failed"):
            backend.generate_until([request, request])
    else:
        assert backend.generate_until([request, request]) == ["answer", "answer"]
    assert completions.create is create
    assert len(sent) == 2
    for payload in sent:
        assert payload[token_key] == expected
        assert payload["temperature"] == 1.0
        assert payload["seed"] == 44
        assert payload["model"] == "qwen-test"
        assert payload["messages"][-1]["content"][-1] == media
        if mode == "think":
            assert payload["messages"][0]["content"] == [{"type": "text", "text": THINKING_SYSTEM_PROMPT}]
        elif mode == "auto":
            assert payload["messages"] == messages
    assert messages[0]["role"] == "user"


def test_mixed_request_budgets_preserve_order_and_do_not_leak(monkeypatch):
    seen = []

    def create(**payload):
        seen.append((payload["model"], payload["max_tokens"]))
        return payload["model"]

    class Backend:
        is_simple = False

        def generate_until(self, requests):
            def send(request):
                return self.client.chat.completions.create(model=request.args[0], max_tokens=4096)

            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(send, requests))

    Backend.__module__ = "lmms_eval.models.chat.openai"
    models = ModuleType("lmms_eval.models")
    models.get_model = lambda name: Backend
    monkeypatch.setitem(sys.modules, "lmms_eval.models", models)
    monkeypatch.setenv("SFT_RL_THINK_MODE", "auto")
    install_adapter()
    backend = Backend()
    completions = SimpleNamespace(create=create)
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    budgets = [8192, 16384, 8192, 1024]
    requests = [SimpleNamespace(args=(str(i), None, {"max_new_tokens": cap}, i, "task", "test"))
                for i, cap in enumerate(budgets)]
    assert backend.generate_until(requests) == ["0", "1", "2", "3"]
    assert dict(seen) == {str(i): cap for i, cap in enumerate(budgets)}
    assert completions.create is create
    assert backend.generate_until([]) == []


def test_durable_responses_reuse_completed_requests_across_restarts(monkeypatch, tmp_path):
    calls = []
    model = "qwen-test"

    def create(**payload):
        assert payload["max_tokens"] == 128
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer-{payload['doc_id']}"))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )

    class Backend:
        is_simple = False
        model_version = model

        def generate_until(self, requests):
            calls.extend(requests)
            from lmms_eval.api.instance import GenerationResult, TokenCounts
            results = []
            for request in requests:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": f"Question {request.args[3]}"}],
                    model=model,
                    max_tokens=4096,
                    doc_id=request.args[3],
                )
                results.append(
                    GenerationResult(
                        text=response.choices[0].message.content,
                        token_counts=TokenCounts(input_tokens=10, output_tokens=5, reasoning_tokens=1),
                    )
                )
            return results

    Backend.__module__ = "lmms_eval.models.chat.openai"
    models = ModuleType("lmms_eval.models")
    models.get_model = lambda name: Backend
    monkeypatch.setitem(sys.modules, "lmms_eval.models", models)
    monkeypatch.setenv("SFT_RL_THINK_MODE", "auto")
    monkeypatch.setenv("HW_EVAL_DURABLE_RESPONSES", str(tmp_path / "responses.sqlite"))
    monkeypatch.setenv("EVAL_SEED", "43")
    install_adapter()
    backend = Backend()
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    requests = [
        SimpleNamespace(
            args=("", lambda doc, index=i: [{"role": "user", "content": f"Question {index}"}], {"max_new_tokens": 128}, i, "task", "test")
            )
            for i in range(3)
        ]
    assert [response.text for response in backend.generate_until(requests)] == ["answer-0", "answer-1", "answer-2"]
    assert len(calls) == 3

    # A simulated instance reclamation and restart: all completed requests are served from SQLite.
    calls.clear()
    install_adapter()
    backend = Backend()
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert [response.text for response in backend.generate_until(requests)] == ["answer-0", "answer-1", "answer-2"]
    assert calls == []


def test_durable_response_key_is_stable():
    messages = [{"role": "user", "content": "Question"}]
    request = SimpleNamespace(
        args=("", lambda doc: messages, {"max_new_tokens": 128}, 7, "task", "test"),
        idx=2,
    )
    assert _request_cache_key(request, "think", 43, "model") == _request_cache_key(request, "think", 43, "model")
    assert _request_cache_key(request, "think", 43, "model") != _request_cache_key(request, "think", 44, "model")


@pytest.mark.parametrize("mode", ["think", "no-think"])
def test_prompt_adapter_satisfies_real_lmms_message_schema(mode):
    from lmms_eval.protocol import ChatMessages

    messages = [{"role": "user", "content": "Which option?"}]
    request = SimpleNamespace(args=("", lambda doc: messages, {"max_new_tokens": 16384},
                                    0, "mmmu_pro_standard_v2", "test"))
    adapted = adapt_requests([request], mode)
    result = adapted[0].arguments[1]({})
    chat = ChatMessages(messages=result)
    payload = chat.to_openai_messages()
    assert payload[0]["role"] == "system"
    assert payload[-1]["content"] == [{"type": "text", "text": "Which option?"}]
    assert messages == [{"role": "user", "content": "Which option?"}]
