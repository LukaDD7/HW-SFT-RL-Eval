#!/usr/bin/env bash
#
# 预下载 lmms-eval 目标 benchmark 数据集到共享 HF 缓存
#   （GPU 节点无外网；lmms-eval 跑的时候以 HF_HOME + HF_HUB_OFFLINE=1 读缓存）
#   需要的 repos（对应 configs/eval/project_vision_opd.yaml 的 task）:
#     lmms-lab/GQA            (testdev_balanced_instructions + testdev_balanced_images)
#     kcz358/DynaMath         (dynamath_reasoning)
#     oscarqjh/ViewSpatial_lmmseval (viewspatial)
#     MMMU/MMMU_Pro           (standard (10 options))
#     CaraJ/MathVerse-lmmseval      (testmini_version_split / vision_intensive)
#     lmms-lab/MMBench        (en / dev)
#   2026-09-03 扩展（SFT-RL 遗忘评测新增 9 项中的 7 项，ReMI/MV-MATH 走本地 replay）:
#     oscarqjh/MindCube_lmmseval   (mindcube_full)
#     lmms-lab/VQAv2              (vqav2_val / validation)
#     lmms-lab/ScienceQA          (ScienceQA-IMG / test)
#     AI4Math/MathVista           (testmini)
#     lmms-lab/MMVet              (mmvet)
#     RunsenXu/MMSI-Bench         (mmsi_bench)
#     BLINK-Benchmark/BLINK       (blink)
#   ReMI 走本地 raw replay，无需下载。
#   注意：snapshot_download 写入 $HF_HOME/hub；随后在线 load_dataset 一次，
#   把 datasets 的解析/arrow 缓存写进 $HF_HOME/datasets（离线 load 必需）。
# 用法（有外网的节点）:
#   bash scripts/sft_rl/download_bench_datasets.sh
set -euo pipefail

HF_HOME="${SFT_RL_HF_CACHE:-${HF_HOME:-${DTOPD_DATASET_ROOT:-.}/hf_cache}}"
export HF_HOME
mkdir -p "${HF_HOME}"

PY="${PYTHON:-python}"

"${PY}" - <<'EOF'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

hf_home = os.environ["HF_HOME"]

def grab(repo, patterns=None):
    print(f"== snapshot {repo} {patterns or ''}")
    snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, cache_dir=hf_home)

grab("lmms-lab/GQA", ["testdev_balanced_instructions/*", "testdev_balanced_images/*"])
grab("kcz358/DynaMath")
grab("oscarqjh/ViewSpatial_lmmseval")
grab("MMMU/MMMU_Pro", ["standard (10 options)/*"])
grab("CaraJ/MathVerse-lmmseval")
grab("lmms-lab/MMBench", ["en/*"])
grab("oscarqjh/MindCube_lmmseval")
grab("lmms-lab/VQAv2")
grab("lmms-lab/ScienceQA")
grab("AI4Math/MathVista")
grab("lmms-lab/MMVet")
grab("RunsenXu/MMSI-Bench")
grab("BLINK-Benchmark/BLINK")

# 在线 load 一次：把 datasets 解析/arrow 缓存写入 $HF_HOME/datasets（离线 load 依赖）
tests = [
    ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
    ("lmms-lab/GQA", "testdev_balanced_images", "testdev"),
    ("kcz358/DynaMath", None, "test"),
    ("oscarqjh/ViewSpatial_lmmseval", None, "test"),
    ("MMMU/MMMU_Pro", "standard (10 options)", "test"),
    ("CaraJ/MathVerse-lmmseval", "testmini_version_split", "vision_intensive"),
    ("lmms-lab/MMBench", "en", "dev"),
    ("oscarqjh/MindCube_lmmseval", None, "train"),
    ("lmms-lab/VQAv2", "vqav2_val", "validation"),
    ("lmms-lab/ScienceQA", "ScienceQA-IMG", "test"),
    ("AI4Math/MathVista", None, "testmini"),
    ("lmms-lab/MMVet", None, "test"),
    ("RunsenXu/MMSI-Bench", None, "test"),
    ("BLINK-Benchmark/BLINK", None, "val"),
]
for repo, cfg, split in tests:
    print(f"== cache-resolve {repo} {cfg or ''} / {split}")
    load_dataset(repo, cfg, split=split)
print("ALL DONE")
EOF

echo "== verify cache =="
for d in \
  "datasets--lmms-lab--GQA" \
  "datasets--kcz358--DynaMath" \
  "datasets--oscarqjh--ViewSpatial_lmmseval" \
  "datasets--MMMU--MMMU_Pro" \
  "datasets--CaraJ--MathVerse-lmmseval" \
  "datasets--lmms-lab--MMBench" \
  "datasets--oscarqjh--MindCube_lmmseval" \
  "datasets--lmms-lab--VQAv2" \
  "datasets--lmms-lab--ScienceQA" \
  "datasets--AI4Math--MathVista" \
  "datasets--lmms-lab--MMVet" \
  "datasets--RunsenXu--MMSI-Bench" \
  "datasets--BLINK-Benchmark--BLINK"; do
  if [ -d "${HF_HOME}/hub/${d}" ]; then
    echo "$(du -sh "${HF_HOME}/hub/${d}" | cut -f1)  ${d}"
  else
    echo "MISSING  ${d}"
  fi
done
echo "datasets cache: $(du -sh "${HF_HOME}/datasets" 2>/dev/null | cut -f1 || echo MISSING)"
