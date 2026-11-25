"""Collect original run evidence on Qizhi, without changing or starting training.

This archive excludes image bytes and model/checkpoint weights. It records their
locations; recipients must also receive those assets and relocate config paths.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import tempfile

RUN = "hybrid-rstg-ossv3-mopd351-teacher200-pos05-full400-h200-20261112-100312-423389"
MANIFEST_SHA = "550f5e9b18b7bc59360da44f9fd8338b843752787dc12b0f66517ea09a071bed"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(run, out, expected_sha):
    run, out = run.resolve(strict=True), out.resolve()
    repo = run.parent.parent
    manifest = run / "run_manifest.json"
    if sha(manifest) != expected_sha:
        raise ValueError("Run manifest hash differs from the supplied historical identity")
    runtime = run / "experiment_config.json"
    if runtime.exists():
        cfg = json.loads(runtime.read_text())
    else:
        resolved = sorted(run.glob("resolved*.yaml"), key=lambda p: p.stat().st_mtime)
        if resolved:
            import yaml
            cfg = yaml.safe_load(resolved[-1].read_text())
        else:
            cfg = json.loads(manifest.read_text())["config"]
    reward = cfg["worker"]["reward"]["reward_function"].rsplit(":", 1)[0]
    reward_path = Path(reward)
    if not reward_path.is_absolute():
        reward_path = repo / reward_path
    reward_path = reward_path.resolve(strict=True)
    members = {}
    exclude = {"runs", "logs", "images", "models", "data", "reports", "__pycache__", ".git", ".runtime", "venv", ".venv", "node_modules", ".pytest_cache"}
    extensions = {".py", ".yaml", ".yml", ".jinja", ".toml", ".md", ".sh"}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in exclude and not d.startswith(".")]
        for name in files:
            p = Path(root) / name
            if p.suffix in extensions or name in {"requirements.txt", "LICENSE", "NOTICE"}:
                members["repo/" + p.relative_to(repo).as_posix()] = p
    for p in run.iterdir():
        if p.name in {"run_manifest.json", "experiment_config.json", "checkpoint_tracker.json"} or p.name.startswith("resolved") and p.suffix in {".yaml", ".yml"}:
            members["original_run/" + p.name] = p
    for p in reward_path.parent.rglob("*.py"):
        if "__pycache__" not in p.parts:
            members["reward_source/" + p.relative_to(reward_path.parent).as_posix()] = p
    image_root = Path(cfg["data"].get("image_dir") or repo)
    datasets, images = {}, {}
    image_key = cfg["data"].get("image_key", "images")
    for split, key in [("train", "train_files"), ("val", "val_files")]:
        files = cfg["data"][key]
        if isinstance(files, str):
            files = [files]
        datasets[split] = []
        for n, filename in enumerate(files):
            p = Path(filename).resolve(strict=True)
            if p.suffix != ".jsonl":
                raise ValueError("This collector expects canonical JSONL; retain other formats with their original loader")
            count = 0
            for line in p.open(encoding="utf-8-sig"):
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                values = row.get(image_key, []) or []
                if isinstance(values, str):
                    values = json.loads(values) if values.startswith("[") else [values]
                for value in values:
                    img = Path(value)
                    img = img if img.is_absolute() else image_root / img
                    images[str(img)] = {"exists": img.is_file(), "bytes": img.stat().st_size if img.is_file() else None}
            members[f"data_original/{split}-{n}.jsonl"] = p
            datasets[split].append({"source": str(p), "rows": count, "sha256": sha(p)})
    packages = {}
    for name in ["torch", "transformers", "vllm", "flash-attn", "ray", "omegaconf", "datasets", "numpy", "tensordict", "torchdata", "qwen-vl-utils"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "scope": "Original code/config/reward/JSONL evidence. Image bytes, model weights, checkpoints and environment wheels are NOT included. Absolute config paths need relocation.",
        "run_dir": str(run), "repo": str(repo), "manifest_sha256": sha(manifest),
        "reward_entrypoint": cfg["worker"]["reward"]["reward_function"],
        "reward_kwargs": cfg["worker"]["reward"].get("reward_function_kwargs", {}),
        "data": datasets, "images_not_included": images,
        "image_count": len(images), "missing_image_count": sum(not x["exists"] for x in images.values()),
        "model_path_not_included": cfg["worker"]["actor"]["model"]["model_path"],
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "files": {k: {"source": str(v), "sha256": sha(v)} for k, v in members.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        metadata = Path(td) / "collection_manifest.json"
        metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with tarfile.open(out, "x:gz") as tar:
            for name, p in sorted(members.items()):
                tar.add(p, arcname=name, recursive=False)
            tar.add(metadata, arcname="collection_manifest.json")
    return report


def self_check():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = root / "repo/runs/example"
        run.mkdir(parents=True)
        reward = root / "repo/reward.py"
        reward.write_text("def compute_score(rows): return []\n")
        train = root / "train.jsonl"
        val = root / "val.jsonl"
        train.write_text('{"images":["missing.png"]}\n')
        val.write_text('{"images":[]}\n')
        cfg = {"worker": {"reward": {"reward_function": str(reward) + ":compute_score"}, "actor": {"model": {"model_path": "/not/included"}}}, "data": {"train_files": str(train), "val_files": str(val), "image_dir": str(root)}}
        (run / "run_manifest.json").write_text(json.dumps({"config": cfg}))
        out = root / "test.tar.gz"
        report = collect(run, out, sha(run / "run_manifest.json"))
        assert report["data"]["train"][0]["rows"] == 1 and report["missing_image_count"] == 1
        with tarfile.open(out) as tar:
            assert "repo/reward.py" in tar.getnames() and "collection_manifest.json" in tar.getnames()
        try:
            collect(run, root / "bad.tar.gz", "wrong")
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched provenance was accepted")
    print("COLLECTOR_SELF_CHECK_PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--root", type=Path, default=Path("/inspire/hdd/project/continuinglearningtheory/xieyuan-24039/zxx/zwz-stage2"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    else:
        matches = [a.run_dir] if a.run_dir else list(a.root.glob("*/repo/runs/" + RUN))
        if len(matches) != 1 or a.out is None:
            ap.error("Supply --out and an unambiguous historical --run-dir (or --root)")
        r = collect(matches[0], a.out, MANIFEST_SHA)
        print(json.dumps({"archive": str(a.out), "data": r["data"], "image_count_not_included": r["image_count"], "missing_images": r["missing_image_count"]}, ensure_ascii=False))
