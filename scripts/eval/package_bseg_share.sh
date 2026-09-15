#!/usr/bin/env bash
# Package the B-segment (original 6-bench) eval stack for teammates in
# different environments (assumes NO shared filesystem with this machine).
#
# B 段 = SFT-RL 项目的目标 6-bench 评测（lmms-eval 0.7.1 @ 88b23e2 + 本仓库
#   dual_track_opd.eval 调度层）, 与 MMF 14-bench (vlmeval_share) 是两套并行
# 评测线, 组合在同一 bundle 家族:
#   - MMF 14-bench:  VLMEvalKit @ ce95c4d (A 组, run_eval_example.sh)
#   - B 段 6-bench:  lmms-eval + dual_track_opd (本包, run_bench_example.sh)
#
# 6 项 benchmark:
#   v2 口径 (4 项 rule-based, 无 judge):  gqa / dynamath / viewspatial / mmmu_pro
#   v1 口径 (2 项):                       remi (本地 replay) / mmbench (judge)
#   对表时 = v2 4 项 + v1 2 项 各用各口径 (manuscript 第 0(b) 步)。
#
# Bundle layout (README_B.md inside is the teammate-facing guide):
#   bseg_share/
#     lmms-eval/                upstream source @ 88b23e2 + 主线 3 处 patch
#     Dual-Track-OPD-eval/      eval 调度层代码 (src/dual_track_opd/eval + configs + eval_tasks + tests + scripts)
#     hf_cache/hub/             6 个数据集的 hub snapshot (blob+symlink 结构原样)
#     remi_replay/              ReMI replay 源 jsonl + ReMI parquet (dataset_root)
#     env/                      requirements_bseg_freeze.txt + 关键版本清单
#     run_bench_example.sh      评测启动模板 (改开头变量)
#     README_B.md
#     VERSION.txt
#
# NOT included (teammate provides): model weights under test, judge weights
# (Qwen3-VL-32B-Instruct-FP8 for v1 mmbench), GPUs, CUDA toolkit (nvcc)。
#
# Usage (CPU instance; no GPU involved):
#   bash scripts/eval/package_bseg_share.sh
# Outputs (both kept):
#   $EVAL_ROOT/bseg_share/                 plain dir
#   $EVAL_ROOT/bseg_share_<YYYYmmdd>.tar.gz
set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
EVAL_ROOT="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval"
STAGE="${EVAL_ROOT}/bseg_share"
TARBALL="${EVAL_ROOT}/bseg_share_$(date +%Y%m%d).tar.gz"

LMMS_GIT="${REPO_ROOT}/third_party_runtime/lmms-eval"
LMMS_COMMIT="88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
HF_HUB="${DTOPD_ROOT}/.cache/huggingface/hub"
REMI_PARQUET="${DTOPD_ROOT}/dataset/ReMI"
REMI_RAW="${DTOPD_ROOT}/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl"
EVAL_ENV="${DTOPD_ROOT}/envs/vision-opd-cu128"

for p in "${LMMS_GIT}" "${HF_HUB}" "${REMI_PARQUET}" "${EVAL_ENV}" "${REPO_ROOT}"; do
    [ -e "${p}" ] || { echo "FATAL: missing ${p}" >&2; exit 1; }
done
[ -f "${REMI_RAW}" ] || { echo "FATAL: missing ReMI replay source ${REMI_RAW}" >&2; exit 1; }

GIT_HEAD="$(git -C "${LMMS_GIT}" rev-parse HEAD)"
[ "${GIT_HEAD}" = "${LMMS_COMMIT}" ] || {
    echo "FATAL: lmms-eval HEAD ${GIT_HEAD} != pinned ${LMMS_COMMIT}" >&2
    exit 1
}

echo "== staging ${STAGE} =="
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

echo "== lmms-eval @ ${LMMS_COMMIT:0:8} =="
# 源码 + 主线 3 类未提交 patch (api/task.py 空串容错 + mmbench utils 8 个语言
# 文件的 /chat/completions 后缀修复) —— 直接整树拷贝 (git archive 不含未提交
# 修改, 而 mmbench judge 打分必须用改过的 utils), 并附 patch 文件副本。
mkdir -p "${STAGE}/lmms-eval"
(cd "${LMMS_GIT}" && tar -cf - --exclude=.git --exclude=__pycache__ .) | tar -xf - -C "${STAGE}/lmms-eval"
mkdir -p "${STAGE}/patches"
git -C "${LMMS_GIT}" diff > "${STAGE}/patches/lmms_eval_mainline_uncommitted.patch"
echo "   (含未提交修改 $(grep -c '^diff --git' "${STAGE}/patches/lmms_eval_mainline_uncommitted.patch") 个文件; patch 副本在 patches/ 供 git 干净树重放)"

echo "== Dual-Track-OPD eval 调度层 =="
# 只取评测需要的子树, 不带整个 repo (teammate 用 pip install -e 安装本目录)。
DST="${STAGE}/Dual-Track-OPD-eval"
mkdir -p "${DST}/src/dual_track_opd" "${DST}/configs/eval" "${DST}/tests" "${DST}/scripts"
git -C "${REPO_ROOT}" archive HEAD \
    src/dual_track_opd/eval src/dual_track_opd/__init__.py \
    configs/eval/project_vision_opd.yaml configs/eval/project_vision_opd_v2.yaml \
    eval_tasks/opd_v2 tests/eval_v2 \
    tests/test_benchmark_suite.py tests/test_project_summary.py tests/test_run_vlm_eval.py \
    scripts/eval/run_target_benchmarks.sh scripts/eval/run_target_benchmarks_v2.sh \
    scripts/sft_rl/run_sftrl_benchmarks.sh scripts/sft_rl/download_bench_datasets.sh \
    scripts/sft_rl/remi_reeval.py docs/remi_mv_math_protocol_20260913.md \
    pyproject.toml README.md \
    | tar -xf - -C "${DST}"

# 移植性 sed：bundle 里的 runner 主线写死 DTOPD_ROOT/REPO_ROOT/EVAL_ENV/LOG_DIR
# 四条绝对路径, teammate 机器不存在 —— 改成 ${VAR:-主线默认} 形式, 由
# run_bench_example.sh 传入 bundle 本地路径。v1 链 run_target_benchmarks.sh
# (env 转发层) 只 exec run_sftrl_benchmarks.sh, 无需 sed。
sed -i -E \
    -e 's|^DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy$|DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"|' \
    -e 's|^REPO_ROOT="\$\{DTOPD_ROOT\}/projects/Dual-Track-OPD"$|REPO_ROOT="${REPO_ROOT:-${DTOPD_ROOT}/projects/Dual-Track-OPD}"|' \
    -e 's|^EVAL_ENV="\$\{DTOPD_ROOT\}/envs/vision-opd-cu128"$|EVAL_ENV="${BSEG_EVAL_ENV:-${DTOPD_ROOT}/envs/vision-opd-cu128}"|' \
    -e 's|^LOG_DIR="\$\{DTOPD_ROOT\}/fc-opd-storage/logs"$|LOG_DIR="${BSEG_LOG_DIR:-${DTOPD_ROOT}/fc-opd-storage/logs}"|' \
    "${DST}/scripts/eval/run_target_benchmarks_v2.sh" "${DST}/scripts/sft_rl/run_sftrl_benchmarks.sh"

echo "== HF hub snapshots (6 数据集, 快照平铺结构) =="
# 目标 = teammate 设 HF_HOME=<bundle>/hf_cache 离线可用。结构对齐 hub 规范:
# hf_cache/hub/datasets--<org>--<name>/snapshots/<rev>/<files>（refs/ 保留以
# 定位 rev; blob 已解引用成真实文件, 无 symlink）。lmms-eval 走
# snapshot_download 缓存命中, HF_HUB_OFFLINE=1 不联网。
mkdir -p "${STAGE}/hf_cache/hub"
for ds in datasets--lmms-lab--GQA datasets--kcz358--DynaMath datasets--oscarqjh--ViewSpatial_lmmseval datasets--MMMU--MMMU_Pro datasets--lmms-lab--MMBench; do
    [ -d "${HF_HUB}/${ds}" ] || { echo "FATAL: missing hub cache ${HF_HUB}/${ds}" >&2; exit 1; }
    mkdir -p "${STAGE}/hf_cache/hub/${ds}"
    # refs + snapshots (symlink 解引用为真实文件); 丢弃 blobs/.no_exist/.lock
    cp -a "${HF_HUB}/${ds}/refs" "${STAGE}/hf_cache/hub/${ds}/refs" 2>/dev/null || true
    cp -aL "${HF_HUB}/${ds}/snapshots" "${STAGE}/hf_cache/hub/${ds}/snapshots"
    find "${STAGE}/hf_cache/hub/${ds}" -name '*.lock' -delete 2>/dev/null || true
done

echo "== ReMI replay 源 (strict exact rescore) =="
mkdir -p "${STAGE}/remi_replay/raw_responses" "${STAGE}/remi_replay/dataset/ReMI"
cp "${REMI_RAW}" "${STAGE}/remi_replay/raw_responses/"
cp "${REMI_PARQUET}"/*.parquet "${REMI_PARQUET}/README.md" "${REMI_PARQUET}/LICENSE" "${STAGE}/remi_replay/dataset/ReMI/" 2>/dev/null || true

echo "== env freeze =="
mkdir -p "${STAGE}/env"
# 清洗规则同 vlmeval 包: 去掉 -e editable (repo checkout), file:// 本地 wheel
# 归一为普通 pin (flash_attn wheel 单独说明), +cu128 本地 tag 去掉。
"${EVAL_ENV}/bin/pip" freeze \
    | grep -vE '^-e|^#' \
    | sed -E 's| @ file:.*flash_attn.*|flash-attn==2.8.3|; s|==([0-9][0-9.]*)\+cu128$|==\1|' \
    > "${STAGE}/env/requirements_bseg_freeze.txt"
cat > "${STAGE}/env/KEY_VERSIONS.txt" <<'EOF_KEY'
主线 B 段评测环境 (vision-opd-cu128, Python 3.12.13) 关键版本:
  lmms_eval   0.7.1  @ 88b23e2bfa16a1edbc16e9e238ed82130b3a4f56 (editable, 主线有未提交 patch, 见 patches/)
  vllm        0.18.0
  torch       2.10.0 (cu128 wheel)
  transformers 5.5.0
  datasets    5.0.0
  pyarrow     22.0.0 (ReMI replay 必需)
  openai      2.24.0
  flash_attn  2.8.3 (wheelhouse cu128-torch210-py312; 装不上可跳过, B 段 openai 后端不需要)
评测模型与 judge 都走 vllm serve + OpenAI 兼容 API, lmms-eval 后端 = openai,
不在 lmms-eval 进程内起 vLLM —— GPU 全部让给两个 server。
EOF_KEY

echo "== README_B.md + run_bench_example.sh =="
cat > "${STAGE}/README_B.md" <<'EOF_README_B'
# B 段 6-bench 评测包（lmms-eval + Dual-Track-OPD 调度层）

用途: 在与主线不同的机器上, 跑与 SFT-RL 主线完全一致的 B 段目标 6-bench
评测。与 MMF 14-bench 评测包（VLMEvalKit, 同 bundle 家族）是**两套独立评测线**
, 不要混用采样/判分口径。

## 6 项 benchmark 与两种口径
| 项 | 数据集 (HF repo) | 口径 | 判分 | max_new_tokens |
|---|---|---|---|---|
| gqa | lmms-lab/GQA (testdev_balanced) | **v2** | rule, <answer> 标签优先 | 4096 |
| dynamath | kcz358/DynaMath | **v2** | rule, 官方链 + 标签优先 | 4096 |
| viewspatial | oscarqjh/ViewSpatial_lmmseval | **v2** | rule, 标签优先 | 4096 |
| mmmu_pro | MMMU/MMMU_Pro (standard 10 选项) | **v2** | rule, 标签优先 | 4096 |
| remi | 本地 replay（包内 remi_replay/） | **v1 生成 + exact 重算** | task-aware exact/relaxed, 全 2600 分母 | 2048 (replay 预算) |
| mmbench | lmms-lab/MMBench (en dev) | **v1** | judge（Qwen3-VL-32B-Instruct） | 1024 |

对表规则（manuscript 第 0(b) 步已定）: **v2 4 项 + v1 2 项, 各用各口径**。
v1 的历史 6 项全 v1 口径分数（macro avg 0.3714 等）只作历史对照。

### ReMI 正式口径

runner 自带 `summary.json` 的 ReMI 行是旧 `normalized_exact_diagnostic`，分母只算
可抽取/已匹配样本，会虚高。正式对表必须用 `remi_reval.py --mode exact` 重算；
`run_bench_example.sh` 的 v1/both 模式会自动执行并打印结果。

## 包内容
| 条目 | 说明 |
|---|---|
| lmms-eval/ | EvolvingLMMs-Lab/lmms-eval @ 88b23e2 源码 + 主线未提交 patch（已应用） |
| patches/lmms_eval_mainline_uncommitted.patch | 同上 patch 的 git diff 副本（干净树重放用） |
| Dual-Track-OPD-eval/ | 调度层: src/dual_track_opd/eval, 2 个 suite YAML, eval_tasks/opd_v2, 回归测试, 4 个 runner 脚本 |
| hf_cache/hub/ | 6 数据集 hub snapshot（**解引用后的平铺结构**, 无 blob/symlink） |
| remi_replay/ | ReMI replay 源 jsonl（基线 prompt 集）+ ReMI parquet（图片） |
| env/ | requirements_bseg_freeze.txt + KEY_VERSIONS.txt |
| run_bench_example.sh | 启动模板（改顶部变量） |
| VERSION.txt | 打包日期与源路径 |

## 你还需要自备
1. 被测模型权重（HF 格式目录）。
2. judge 权重（只 v1 mmbench 需要）: Qwen3-VL-32B-Instruct-FP8（主线用的
   本地 FP8 目录; 也可用 BF16 原版 + 对应显存）。v2 4 项 + remi 不需要 judge。
3. GPU: v2 4 项 = 1 张（只起被测模型 server）; v1 2 项 = 2 张（被测 + judge）。
4. CUDA toolkit（nvcc）: 机器没有 /usr/local/cuda 时必须（坑 A, 同 vlmeval 包坑 1）。

## 环境安装（Python 3.12; 有外网机器）
```bash
conda create -y -n bseg python=3.12 pip
conda activate bseg
# 1) torch 家族（cu128 索引; vllm 0.18.0 对应 cu128）:
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# 2) 其余钉死版本（约 370 行; -e/file:// 行已清洗）:
pip install -r env/requirements_bseg_freeze.txt
# 3) lmms-eval + 调度层（editable, 主线同款安装方式）:
pip install -e lmms-eval
pip install -e Dual-Track-OPD-eval
```
flash_attn 2.8.3 若装不上可跳过（B 段走 openai 后端, 不在 lmms-eval 里起
vLLM, 不需要 flash_attn）。非核心包装不上同理, 注释掉再装。
核心链路 = lmms_eval / vllm / torch / transformers / datasets / pyarrow /
openai / pyyaml。

## 坑（主线全踩过）
- A. flashinfer/vLLM JIT 需要 nvcc: 被测模型 server（vllm serve）warmup 会
  JIT 编译, 机器无 /usr/local/cuda 时崩 "Could not find nvcc"。解法 = 启动
  前导出 CUDA_HOME/PATH/LIBRARY_PATH/LD_LIBRARY_PATH 指向你的 CUDA toolkit
  （run_bench_example.sh 顶部有模板）。
- B. ★ lmms-eval 的 mmbench utils 主线有未提交修改（OPENAI_API_URL 自动补
  /chat/completions 后缀 + api/task.py 空响应容错）。本包 lmms-eval/ 目录
  已应用; 若你重新 git clone 上游, 必须先打 patches/ 里的 patch, 否则
  mmbench judge 打分打到 /v1（无后缀）404。
- C. 离线数据: 解包后设 HF_HOME 指向 <bundle>/hf_cache（run 模板已带）,
  HF_HUB_OFFLINE=1。lmms-eval 的 datasets cache（arrow 解析缓存）首次运行
  时会在本地重建（GQA 约 13G 磁盘/十几分钟; 只解析不下载）。
- D. GQA 全量 12544 条很慢（8 workers ~6h）; 冒烟用 SFT_RL_SMOKE=1 或
  EVAL_SMOKE=1（每项 8 条）。
- E. 中断续跑: v2/v1 runner 都自动 --resume-from 同名 run 目录（响应缓存
  逐条回放）; 前台跑被 SIGHUP 杀 = 全损, 必须 nohup/setsid 包裹。
- F. judge（mmbench 用）要全程在线; shell 有 http_proxy 时发 127.0.0.1 的
  请求会被劫持 → 模板已带 no_proxy。
- G. remi 项离线跑必须同时提供: prior_raw_root（remi_replay/raw_responses）
  + dataset_root（remi_replay/dataset）—— 模板已设好。

## 跑
1. 打开 run_bench_example.sh, 改顶部变量（模型路径/GPU/端口/env python/
   CUDA toolkit/数据缓存根）。
2. 分两段跑（口径不同, 与主线对表规则一致）:
   - v2 4 项:   RUN_V2=1 bash run_bench_example.sh   （1 张卡, 无 judge）
   - v1 2 项:   RUN_V1=1 bash run_bench_example.sh   （2 张卡, 含 judge）
   - 全部:      bash run_bench_example.sh            （默认先 v2 后 v1）
3. 产物: ${BUNDLE_DIR}/outputs/<RUN_NAME>_v2 与 ${BUNDLE_DIR}/outputs/<RUN_NAME>_remimmb_nojudge/_judged
   （run_manifest.json + lmms/*/results + replay/remi.jsonl）。
   ReMI 正式分数看 `logs/bseg_remi_exact_<RUN_NAME>.log`; `summary.json` 的 ReMI
   行是旧 normalized_exact 诊断，不能用于对表。

## 可比性
不变项: lmms-eval @ 88b23e2 + 上述未提交 patch + 同一批数据 snapshot +
dual_track_opd.eval 调度层（temperature=0, seed 42, workers 8, batch 1）+
判分链（v2 = <answer> 标签优先 + 确定性 fallback; ReMI = remi_reeval.py
全分母 exact/relaxed; v1 mmbench = 32B judge extractor）。满足即与主线 B 段表
直接可比。
EOF_README_B

cat > "${STAGE}/run_bench_example.sh" <<'EOF_RUN_B'
#!/usr/bin/env bash
# B 段 6-bench eval — teammate template.
# 用法: 改变量 → RUN_V2=1 / RUN_V1=1 / (全跑) bash run_bench_example.sh
set -euo pipefail

# ---- variables (edit these) ----
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CKPT=/path/to/model_hf_dir          # 被测模型 (HF 格式)
JUDGE_MODEL_PATH=/path/to/Qwen3-VL-32B-Instruct-FP8   # 只 v1 mmbench 需要
EVAL_GPU=0
JUDGE_GPU=1
EVAL_PORT=8000
JUDGE_PORT=8001
PYTHON=/path/to/bseg/bin/python           # 你的 env python (装好 lmms-eval + Dual-Track-OPD-eval)
CUDA_TOOLCHAIN=/usr/local/cuda            # nvcc 根; 无 /usr/local/cuda → README_B 坑 A
RUN_NAME=my_ckpt_bseg

# ---- derived (usually no edit) ----
export HF_HOME="${BUNDLE_DIR}/hf_cache"   # 6 数据集离线缓存 (README_B 坑 C)
export SFT_RL_HF_CACHE="${HF_HOME}"
REPO_DIR="${BUNDLE_DIR}/Dual-Track-OPD-eval"
mkdir -p "${BUNDLE_DIR}/outputs" "${BUNDLE_DIR}/logs" "${HF_HOME}/datasets"

# ---- portability roots for the runner scripts (sed 改过的变量在此赋值) ----
export DTOPD_ROOT="${BUNDLE_DIR}"                     # runner 内部只用于默认值兜底
export REPO_ROOT="${REPO_DIR}"                        # 调度层代码根
export BSEG_EVAL_ENV="$(dirname "$(dirname "${PYTHON}")")"   # 你的 conda env 根
export BSEG_LOG_DIR="${BUNDLE_DIR}/logs"              # vllm server 日志目录

# ---- CUDA toolchain exports (flashinfer JIT; README_B 坑 A) ----
export CUDA_HOME="${CUDA_TOOLCHAIN}"
export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${LD_LIBRARY_PATH:-}"
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost

MODE="both"
[ "${RUN_V2:-0}" = "1" ] && MODE="v2"
[ "${RUN_V1:-0}" = "1" ] && MODE="v1"

run_v2() {
    echo "== [v2] 4 rule-based benches (1 GPU, no judge) =="
    env EVAL_CKPT="${MODEL_CKPT}" \
        EVAL_RUN_NAME="${RUN_NAME}_v2" \
        EVAL_GPU="${EVAL_GPU}" \
        EVAL_PORT="${EVAL_PORT}" \
        EVAL_MAX_LEN=65536 \
        EVAL_CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN}" \
        EVAL_HF_CACHE="${HF_HOME}" \
        DTOPD_EVAL_ROOT="${BUNDLE_DIR}/outputs" \
        bash "${REPO_DIR}/scripts/eval/run_target_benchmarks_v2.sh" \
        2>&1 | tee "${BUNDLE_DIR}/logs/bseg_v2_${RUN_NAME}.log"
}

run_v1() {
    echo "== [v1] remi + mmbench (2 GPUs, judge) =="
    env SFT_RL_MODEL_HF="${MODEL_CKPT}" \
        SFT_RL_RUN_NAME="${RUN_NAME}_remimmb" \
        SFT_RL_BENCHMARKS=remi \
        SFT_RL_JUDGE_BENCHMARKS=mmbench \
        SFT_RL_JUDGE_HF="${JUDGE_MODEL_PATH}" \
        SFT_RL_EVAL_GPU="${EVAL_GPU}" \
        SFT_RL_JUDGE_GPU="${JUDGE_GPU}" \
        SFT_RL_EVAL_PORT="${EVAL_PORT}" \
        SFT_RL_JUDGE_PORT="${JUDGE_PORT}" \
        SFT_RL_MAX_LEN=65536 \
        SFT_RL_CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN}" \
        SFT_RL_HF_CACHE="${HF_HOME}" \
        DTOPD_EVAL_ROOT="${BUNDLE_DIR}/outputs" \
        QWEN3VL8B_RAW_ROOT="${BUNDLE_DIR}/remi_replay/raw_responses" \
        DTOPD_DATASET_ROOT="${BUNDLE_DIR}/remi_replay/dataset" \
        bash "${REPO_DIR}/scripts/eval/run_target_benchmarks.sh" \
        2>&1 | tee "${BUNDLE_DIR}/logs/bseg_v1_${RUN_NAME}.log"

    echo "== [ReMI] strict full-denominator rescore =="
    "${PYTHON}" "${REPO_DIR}/scripts/sft_rl/remi_reeval.py" \
        --mode exact \
        --jsonl "${BUNDLE_DIR}/outputs/${RUN_NAME}_remimmb_nojudge/replay/remi.jsonl" \
        --label-jsonl "${BUNDLE_DIR}/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl" \
        2>&1 | tee "${BUNDLE_DIR}/logs/bseg_remi_exact_${RUN_NAME}.log"
}

case "${MODE}" in
    v2) run_v2 ;;
    v1) run_v1 ;;
    both) run_v2; run_v1 ;;
    *) echo "usage: RUN_V2=1 (v2 only) | RUN_V1=1 (v1 only) | bash $0 (both)"; exit 1 ;;
esac
echo "== done. outputs under ${BUNDLE_DIR}/outputs (见各 log 末尾 summary) =="
EOF_RUN_B
chmod +x "${STAGE}/run_bench_example.sh"

cat > "${STAGE}/VERSION.txt" <<EOF_VERSION
packaged_at: $(date '+%Y-%m-%d %H:%M:%S %z')
lmms_eval_commit: ${LMMS_COMMIT}
lmms_eval_source: ${LMMS_GIT} (整树拷贝, 含主线未提交 patch; 副本 patches/lmms_eval_mainline_uncommitted.patch)
scheduler_source: ${REPO_ROOT} git archive HEAD (src/dual_track_opd/eval + configs/eval + eval_tasks/opd_v2 + tests + scripts)
hf_hub_source: ${HF_HUB} (5 dataset repos, symlink 解引用平铺)
remi_replay_source: ${REMI_RAW} + ${REMI_PARQUET} (strict rescore via scripts/sft_rl/remi_reeval.py)
env_source: ${EVAL_ENV} (pip freeze -> env/requirements_bseg_freeze.txt)
EOF_VERSION

echo "== tarball =="
rm -f "${TARBALL}"
tar -czf "${TARBALL}" -C "${EVAL_ROOT}" bseg_share

echo ""
echo "== done =="
du -sh "${STAGE}" "${TARBALL}"
echo "freeze lines: $(wc -l < "${STAGE}/env/requirements_bseg_freeze.txt")"
sha256sum "${TARBALL}" | tee "${TARBALL}.sha256"
echo "bundle dir : ${STAGE}"
echo "tarball    : ${TARBALL}"
