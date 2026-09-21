import json
import io
import sys
from pathlib import Path

import pytest

from dual_track_opd.eval import run_vlm_eval
from dual_track_opd.eval.thinking_adapter import THINKING_SYSTEM_PROMPT


@pytest.mark.parametrize("mode,cap", [("auto", 1024), ("think", 8192), ("no-think", 1024)])
def test_replay_prompt_preserves_sampled_payload(monkeypatch, mode, cap):
    monkeypatch.setenv("SFT_RL_THINK_MODE", mode)
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append(json.loads(request.data))
            return io.BytesIO(b'{"choices": [{"message": {"content": "A"}}]}')

    monkeypatch.setattr(run_vlm_eval.urllib.request, "build_opener", lambda *args: Opener())
    question = "Which option?\nReturn only the final answer. Do not provide reasoning."
    for seed in range(42, 46):
        response = run_vlm_eval._request(
            api_base="http://unused/v1", api_key="EMPTY", model="test-model",
            question=question, images=[b"image-fixture"], max_tokens=1024,
            temperature=1.0, seed=seed, timeout=1, task_name="ReMI",
        )
        assert response["choices"][0]["message"]["content"] == "A"
    assert [payload["seed"] for payload in sent] == [42, 43, 44, 45]
    for payload in sent:
        assert payload["temperature"] == 1.0
        assert payload["max_tokens"] == cap
        assert payload["model"] == "test-model"
        assert payload["messages"][-1]["content"][0]["type"] == "image_url"
        if mode == "think":
            assert payload["messages"][0]["content"] == THINKING_SYSTEM_PROMPT
            assert "Do not provide reasoning" not in payload["messages"][-1]["content"][-1]["text"]
        elif mode == "auto":
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["content"][-1]["text"] == question


def test_mv_math_replay_resolves_historical_problem_id(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "MV-MATH" / "images" / "images"
    image_root.mkdir(parents=True)
    (image_root / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nimage")
    (dataset_root / "MV-MATH" / "MV-MATH.json").write_text(
        json.dumps([{"problem_id": 17, "input_image": ["a.png"]}]),
        encoding="utf-8",
    )

    def fake_request(**kwargs):
        assert kwargs["question"] == "Choose one."
        assert kwargs["images"] == [image_root / "a.png"]
        return {
            "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

    monkeypatch.setattr(run_vlm_eval, "_request", fake_request)
    results = run_vlm_eval.replay_rows(
        [{"sample_id": 17, "question": "Choose one.", "ground_truth": "B"}],
        dataset="MV-MATH",
        dataset_root=dataset_root,
        api_base="http://unused/v1",
        api_key="EMPTY",
        model="Vision-OPD-4B",
        max_tokens=32,
        temperature=1.0,
        seed=42,
        workers=2,
        timeout=1,
    )

    assert results == [
        {
            "dataset": "MV-MATH",
            "sample_id": "17",
            "source_row_index": 1,
            "question": "Choose one.",
            "ground_truth": "B",
            "prediction": "B",
            "finish_reason": "stop",
            "image_count": 1,
            "model": "Vision-OPD-4B",
            "usage": {"completion_tokens": 1},
            "error": "",
        }
    ]


def test_replay_resume_preserves_successful_rows_and_retries_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "MV-MATH" / "images" / "images"
    image_root.mkdir(parents=True)
    (image_root / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nimage")
    (image_root / "b.png").write_bytes(b"\x89PNG\r\n\x1a\nimage")
    (dataset_root / "MV-MATH" / "MV-MATH.json").write_text(
        json.dumps(
            [
                {"problem_id": 17, "input_image": ["a.png"]},
                {"problem_id": 18, "input_image": ["b.png"]},
            ]
        ),
        encoding="utf-8",
    )
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"sample_id": 17, "question": "First?", "ground_truth": "A"}) + "\n"
        + json.dumps({"sample_id": 18, "question": "Second?", "ground_truth": "B"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps(
            {
                "dataset": "MV-MATH",
                "sample_id": "17",
                "source_row_index": 1,
                "prediction": "OLD",
                "error": "",
                "model": "Vision-OPD-4B",
            }
        )
        + "\n"
        + json.dumps({"dataset": "MV-MATH", "sample_id": "18", "error": "URLError"}) + "\n",
        encoding="utf-8",
    )

    def fake_request(**kwargs):
        return {
            "choices": [{"message": {"content": "NEW"}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr(run_vlm_eval, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vlm_eval",
            "--input-jsonl",
            str(source),
            "--output-jsonl",
            str(output),
            "--dataset",
            "MV-MATH",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "Vision-OPD-4B",
            "--resume",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        run_vlm_eval.main()
    assert caught.value.code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(row["sample_id"], row["prediction"]) for row in rows] == [
        ("17", "OLD"),
        ("18", "NEW"),
    ]
