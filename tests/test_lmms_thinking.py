"""Check the SDK payload after a backend applies its historical token clamp."""
from concurrent.futures import ThreadPoolExecutor
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dual_track_opd.eval.lmms_thinking import install_adapter
from dual_track_opd.eval.thinking_adapter import THINKING_SYSTEM_PROMPT


@pytest.mark.parametrize("mode,expected", [("think", 8192), ("no-think", 4096)])
@pytest.mark.parametrize("fails", [False, True])
def test_sdk_budget_survives_backend_clamp_and_is_restored(monkeypatch, mode, expected, fails):
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
                    max_tokens=min(kwargs["max_new_tokens"], 4096),
                    model="qwen-test", temperature=1.0, seed=44,
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
    request = SimpleNamespace(args=(
        "context", lambda doc: messages, {"max_new_tokens": 8192}, 0, "gqa_v2", "test",
    ))
    if fails:
        with pytest.raises(RuntimeError, match="backend failed"):
            backend.generate_until([request, request])
    else:
        assert backend.generate_until([request, request]) == ["answer", "answer"]
    assert completions.create is create
    assert len(sent) == 2
    for payload in sent:
        assert payload["max_tokens"] == expected
        assert payload["temperature"] == 1.0
        assert payload["seed"] == 44
        assert payload["model"] == "qwen-test"
        assert payload["messages"][-1]["content"][-1] == media
        if mode == "think":
            assert payload["messages"][0]["content"] == THINKING_SYSTEM_PROMPT
    assert messages[0]["role"] == "user"
