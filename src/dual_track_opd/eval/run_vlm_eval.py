"""Replay canonical prior benchmark inputs through an OpenAI-compatible VLM endpoint.

This is a fallback for project benchmarks that are not registered in the pinned
lmms-eval release. It deliberately preserves the prior prompt and ground truth
while replacing only the model response.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence


QUESTION_KEYS = ("question", "query", "prompt", "problem", "instruction", "input", "user_prompt")
GROUND_TRUTH_KEYS = (
    "ground_truth",
    "gt",
    "gt_answer",
    "answer",
    "answers",
    "label",
    "target",
    "gold",
    "correct_answer",
)
ID_KEYS = ("row_id", "sample_id", "id", "question_id", "uid", "index")
IMAGE_KEYS = ("image", "image_path", "image_paths", "image_url", "image_urls", "images")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _first(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(item)
    return rows


def _existing_image_paths(value: Any) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def walk(obj: Any, key: str = "") -> None:
        if isinstance(obj, dict):
            for nested_key, nested in obj.items():
                walk(nested, nested_key)
        elif isinstance(obj, list):
            for nested in obj:
                walk(nested, key)
        elif isinstance(obj, str) and (
            key in IMAGE_KEYS or "image" in key.lower() or "path" in key.lower()
        ):
            path = Path(obj).expanduser()
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)

    walk(value)
    return found


class DatasetImageResolver:
    def __init__(self, dataset: str, dataset_root: Path) -> None:
        self.dataset = dataset
        self.dataset_root = dataset_root
        self._mv_math: dict[str, list[Path]] | None = None
        self._remi_tables: list[Any] | None = None
        self._load_lock = threading.Lock()

    def resolve(self, item: dict[str, Any], row_index: int) -> list[bytes | Path]:
        direct = _existing_image_paths(item)
        if direct:
            return direct
        sample_id = _stringify(_first(item, ID_KEYS)) or str(row_index)
        normalized = self.dataset.lower().replace("_", "-")
        if normalized == "mv-math":
            return self._resolve_mv_math(sample_id)
        if normalized == "remi":
            return self._resolve_remi(sample_id)
        raise ValueError(
            f"no image resolver for {self.dataset!r}; use lmms-eval for supported tasks"
        )

    def _resolve_mv_math(self, sample_id: str) -> list[Path]:
        if self._mv_math is None:
            with self._load_lock:
                if self._mv_math is None:
                    source = self.dataset_root / "MV-MATH" / "MV-MATH.json"
                    if not source.is_file():
                        raise FileNotFoundError(f"MV-MATH metadata not found: {source}")
                    entries = json.loads(source.read_text(encoding="utf-8"))
                    image_root = self.dataset_root / "MV-MATH" / "images" / "images"
                    self._mv_math = {
                        str(entry["problem_id"]): [
                            image_root / name for name in entry.get("input_image", [])
                        ]
                        for entry in entries
                    }
        paths = self._mv_math.get(sample_id, [])
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"MV-MATH image(s) missing: {missing[:3]}")
        return paths

    def _resolve_remi(self, sample_id: str) -> list[bytes]:
        try:
            index = int(sample_id)
        except ValueError as exc:
            raise ValueError(f"ReMI sample id must be an integer, got {sample_id!r}") from exc
        if self._remi_tables is None:
            with self._load_lock:
                if self._remi_tables is None:
                    try:
                        import pyarrow.parquet as pq
                    except ImportError as exc:
                        raise RuntimeError("ReMI replay requires pyarrow") from exc
                    files = sorted((self.dataset_root / "ReMI").glob("*.parquet"))
                    if not files:
                        raise FileNotFoundError(
                            f"no ReMI parquet files under {self.dataset_root / 'ReMI'}"
                        )
                    self._remi_tables = [pq.read_table(path) for path in files]
        remaining = index
        for table in self._remi_tables:
            if remaining >= table.num_rows:
                remaining -= table.num_rows
                continue
            images: list[bytes] = []
            for column in sorted(name for name in table.column_names if name.startswith("image_")):
                value = table.column(column)[remaining].as_py()
                if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
                    images.append(value["bytes"])
                elif isinstance(value, bytes):
                    images.append(value)
            return images
        raise IndexError(f"ReMI row {index} is outside the available parquet rows")


def _data_url(image: bytes | Path) -> str:
    if isinstance(image, Path):
        data = image.read_bytes()
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
    else:
        data = image
        if data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif data.startswith(b"RIFF"):
            mime = "image/webp"
        else:
            mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _request(
    *,
    api_base: str,
    api_key: str,
    model: str,
    question: str,
    images: list[bytes | Path],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images
    ]
    content.append({"type": "text", "text": question})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    # Replay endpoints are normally local vLLM servers.  Some HPC login shells
    # export HTTP(S)_PROXY globally; urllib would otherwise route 127.0.0.1
    # through that proxy and fail with an unrelated connection error.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


def replay_rows(
    rows: list[dict[str, Any]],
    *,
    row_indices: Sequence[int] | None = None,
    dataset: str,
    dataset_root: Path,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    workers: int,
    timeout: float,
) -> list[dict[str, Any]]:
    resolver = DatasetImageResolver(dataset, dataset_root)
    indices = list(row_indices) if row_indices is not None else list(range(1, len(rows) + 1))
    if len(indices) != len(rows):
        raise ValueError("row_indices must have the same length as rows")
    indexed_rows = list(zip(indices, rows))
    source_items = dict(indexed_rows)

    def process(source_index: int) -> dict[str, Any]:
        item = source_items[source_index]
        question = _stringify(_first(item, QUESTION_KEYS))
        if not question:
            raise ValueError(f"row {source_index} has no recognized question field")
        sample_id = _stringify(_first(item, ID_KEYS)) or str(source_index)
        images = resolver.resolve(item, source_index)
        if not images:
            raise ValueError(f"row {source_index} ({sample_id}) resolved zero images")
        response = _request(
            api_base=api_base,
            api_key=api_key,
            model=model,
            question=question,
            images=images,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        return {
            "dataset": dataset,
            "sample_id": sample_id,
            "source_row_index": source_index,
            "question": question,
            "ground_truth": _first(item, GROUND_TRUTH_KEYS),
            "prediction": message.get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "image_count": len(images),
            "model": model,
            "usage": response.get("usage", {}),
            "error": "",
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {
            pool.submit(process, source_index): source_index
            for source_index, _item in indexed_rows
        }
        indexed_results: dict[int, dict[str, Any]] = {}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:  # Keep a complete, auditable raw output.
                item = source_items[index]
                indexed_results[index] = {
                    "dataset": dataset,
                    "sample_id": _stringify(_first(item, ID_KEYS)) or str(index),
                    "source_row_index": index,
                    "question": _stringify(_first(item, QUESTION_KEYS)),
                    "ground_truth": _first(item, GROUND_TRUTH_KEYS),
                    "prediction": "",
                    "finish_reason": None,
                    "image_count": 0,
                    "model": model,
                    "usage": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
        results.extend(indexed_results[index] for index in sorted(indexed_results))
    return results


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed_by_sample_id(
    path: Path, *, dataset: str, model: str, expected_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", ""))
            if item.get("error"):
                continue
            if item.get("dataset") != dataset or item.get("model") != model:
                raise ValueError(
                    f"{path}:{line_number}: output belongs to a different dataset/model; "
                    "use a new output path or remove the old run"
                )
            if sample_id in completed:
                raise ValueError(f"{path}:{line_number}: duplicate successful sample {sample_id!r}")
            completed[sample_id] = item
    return {sample_id: completed[sample_id] for sample_id in expected_ids if sample_id in completed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--dataset", required=True, choices=("MV-MATH", "ReMI"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Vision-OPD-4B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful output rows and retry only missing/failed rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_jsonl).expanduser()
    rows = _read_jsonl(input_path, args.limit)
    output_path = Path(args.output_jsonl).expanduser()
    sample_ids = [
        _stringify(_first(item, ID_KEYS)) or str(index)
        for index, item in enumerate(rows, start=1)
    ]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("input JSONL contains duplicate sample ids")
    completed = (
        _completed_by_sample_id(
            output_path,
            dataset=args.dataset,
            model=args.model,
            expected_ids=sample_ids,
        )
        if args.resume
        else {}
    )
    pending_positions = [
        index - 1 for index, sample_id in enumerate(sample_ids, start=1)
        if sample_id not in completed
    ]
    retried = replay_rows(
        [rows[position] for position in pending_positions],
        row_indices=[position + 1 for position in pending_positions],
        dataset=args.dataset,
        dataset_root=Path(args.dataset_root).expanduser(),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        workers=args.workers,
        timeout=args.timeout,
    )
    results_by_id = {str(row["sample_id"]): row for row in retried}
    merged = [
        results_by_id.get(sample_id) or completed.get(sample_id)
        for sample_id in sample_ids
    ]
    write_jsonl(output_path, merged)
    errors = sum(bool(row["error"]) for row in merged)
    print(
        json.dumps(
            {
                "rows": len(merged),
                "errors": errors,
                "resumed_successful": len(completed),
                "retried": len(retried),
                "output": args.output_jsonl,
            }
        )
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
