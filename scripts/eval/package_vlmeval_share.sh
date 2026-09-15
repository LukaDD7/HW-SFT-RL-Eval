#!/usr/bin/env bash
# Package the MMF 14-bench VLMEvalKit eval stack for a teammate in a
# different environment (assumes NO shared filesystem with this machine).
#
# Bundle layout (README.md inside is the teammate-facing guide):
#   vlmeval_share/
#     VLMEvalKit/               upstream source @ ce95c4d (git archive, no .git)
#     configs/                  the 2 model config JSONs (model_path needs editing)
#     lmu_data/                 14 benchmark TSVs (~1.6G, offline-ready)
#     env/                      requirements_eval_freeze.txt + environment manifest
#     run_eval_example.sh       judge + run.py command template (edit variables)
#     README.md
#     VERSION.txt
#
# NOT included (teammate provides): model weights under test, judge weights
# (Qwen3-VL-32B-Instruct, HF repo of same name), GPUs, and a CUDA toolkit
# (nvcc) when the machine has no /usr/local/cuda — the flashinfer
# sampling-kernel JIT inside vLLM 0.27.1 needs it (see README "坑 1",
# the same failure the judge hit on 2026-09-07).
#
# Usage (CPU instance; no GPU involved):
#   bash scripts/eval/package_vlmeval_share.sh
# Outputs (both kept):
#   $EVAL_ROOT/vlmeval_share/               plain dir (rsync-friendly)
#   $EVAL_ROOT/vlmeval_share_<YYYYmmdd>.tar.gz
set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
EVAL_ROOT="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval"
STAGE="${EVAL_ROOT}/vlmeval_share"
TARBALL="${EVAL_ROOT}/vlmeval_share_$(date +%Y%m%d).tar.gz"

VLMEVAL_GIT="${DTOPD_ROOT}/third_party_runtime/VLMEvalKit"
VLMEVAL_COMMIT="ce95c4d14829b8ed688fdfaea210c4bb510e1a35"
CFG_DIR="${EVAL_ROOT}/vlmeval_cfg"
LMU_DATA="${DTOPD_ROOT}/fc-opd-storage/lmu_data"
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/va_opd_qwen35_environment_manifest.json"

for p in "${VLMEVAL_GIT}" "${CFG_DIR}" "${LMU_DATA}" "${ENV_PREFIX}"; do
    [ -e "${p}" ] || { echo "FATAL: missing ${p}" >&2; exit 1; }
done
[ -f "${CFG_DIR}/qwen3vl_8b_base_mfr.json" ] || { echo "FATAL: missing base config JSON" >&2; exit 1; }
[ -f "${CFG_DIR}/qwen3vl_8b_tailsft_mfr.json" ] || { echo "FATAL: missing tailsft config JSON" >&2; exit 1; }
[ -f "${ENV_MANIFEST}" ] || { echo "FATAL: missing env manifest ${ENV_MANIFEST}" >&2; exit 1; }

GIT_HEAD="$(git -C "${VLMEVAL_GIT}" rev-parse HEAD)"
[ "${GIT_HEAD}" = "${VLMEVAL_COMMIT}" ] || {
    echo "FATAL: VLMEvalKit HEAD ${GIT_HEAD} != pinned ${VLMEVAL_COMMIT}" >&2
    exit 1
}
if [ -n "$(git -C "${VLMEVAL_GIT}" status --porcelain)" ]; then
    echo "FATAL: VLMEvalKit tree is dirty; refusing to package" >&2
    exit 1
fi

echo "== staging ${STAGE} =="
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

echo "== VLMEvalKit @ ${VLMEVAL_COMMIT:0:8} (git archive, no .git) =="
mkdir -p "${STAGE}/VLMEvalKit"
git -C "${VLMEVAL_GIT}" archive --format=tar "${VLMEVAL_COMMIT}" | tar -x -C "${STAGE}/VLMEvalKit"

echo "== configs (2 JSONs; teammate must edit model_path) =="
mkdir -p "${STAGE}/configs"
cp "${CFG_DIR}/qwen3vl_8b_base_mfr.json" "${CFG_DIR}/qwen3vl_8b_tailsft_mfr.json" "${STAGE}/configs/"

echo "== lmu_data TSVs (14 benchmarks, ~1.6G) =="
mkdir -p "${STAGE}/lmu_data"
cp "${LMU_DATA}"/*.tsv "${STAGE}/lmu_data/"

echo "== env spec (cleaned pip freeze + manifest) =="
mkdir -p "${STAGE}/env"
# -e editable lines (dual-track-opd, verl) dropped: they are repo checkouts,
#   not needed for eval.
# vllm @ file://...  -> vllm==0.27.1: the wheel is installed from the
#   wheels.vllm.ai URL (see README install order); the plain pin keeps the
#   file pip-installable.
# +cu132 local tags stripped: torch family must be installed FIRST from the
#   pytorch cu132 index (README), then this file's pins are already satisfied.
"${ENV_PREFIX}/bin/pip" freeze \
    | grep -vE '^-e|^#' \
    | sed -E 's|^vllm @ file:.*|vllm==0.27.1|; s|==([0-9][0-9.]*)\+cu132$|==\1|' \
    > "${STAGE}/env/requirements_eval_freeze.txt"
cp "${ENV_MANIFEST}" "${STAGE}/env/environment_manifest.json"

echo "== README.md + run_eval_example.sh =="
cat > "${STAGE}/README.md" <<'EOF_README'
# MMF 14-benchmark 评测包（VLMEvalKit 同款环境快照）

用途: 在与主线不同的机器/环境里, 跑与 Dual-Track-OPD 主线完全一致的 MMF
14-benchmark 评测（VLMEvalKit @ ce95c4d, greedy 解码, judge = 本地 vLLM 起的
Qwen3-VL-32B-Instruct）。协议出处: MMF 论文（arXiv:2601.21821）14-bench 组合。

## 包内容
| 条目 | 说明 |
|---|---|
| VLMEvalKit/ | open-compass/VLMEvalKit @ ce95c4d 源码快照（git archive, 无 .git） |
| configs/ | 两个模型 config JSON（base / tailsft）—— ★ 里面 model_path 是打包机的绝对路径, 必须改成你本地的权重路径 |
| lmu_data/ | 14 个 benchmark 的 TSV（题目+图片, ~1.6G）; LMUData 指到这里即完全离线, 不触发在线下载/版本漂移（MD5 由框架自动校验） |
| env/requirements_eval_freeze.txt | 主线跑通环境的全量 pip freeze（-e 行已删, vllm/torch 的本地 tag 已清洗; torch/vllm 按下面顺序单独先装） |
| env/environment_manifest.json | 主线环境 manifest（关键版本钉死 + vllm wheel sha256, 供核对） |
| run_eval_example.sh | judge + 评测启动命令模板（改开头变量即可跑） |
| VERSION.txt | 打包日期与源路径 |

## 你还需要自备（包里不含的大件）
1. 被评测模型权重（HF 格式目录）: base 臂可从 HF 下载 Qwen/Qwen3-VL-8B-Instruct;
   微调 ckpt 从训练侧拷贝。评自己的模型 = 复制一份 configs/*.json, 改 key 名与
   model_path, 其余采样参数别动（改了就和主线不可比）。
2. judge 权重: Qwen3-VL-32B-Instruct（HF 同名 repo, ~63G BF16）。
3. GPU: 被测模型 1 张（80G 级; 8B + 32K 上下文单卡足够）+ judge 1 张
   （32B BF16 建议 H200 141G; 80G 卡装不下 BF16, 可用 FP8 权重并相应调
   gpu-memory-utilization）。
4. CUDA toolkit（nvcc）: 机器没有 /usr/local/cuda 时必须准备一个（见坑 1）。

## 环境安装（有外网的机器; Python 3.12）
```bash
conda create -y -n vlmeval python=3.12 pip
conda activate vlmeval
# 1) torch 家族（必须最先装, 走 cu132 索引）:
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
# 2) vLLM 0.27.1（固定 wheel; sha256 核对 env/environment_manifest.json 的
#    vllm_wheel_sha256 = 98e9fc2a...3a2670）:
pip install "https://wheels.vllm.ai/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl"
# 3) 其余钉死版本（约 270 个包, 与主线逐版本一致）:
pip install -r env/requirements_eval_freeze.txt
```
个别非核心包装不上（如 torch_c_dlpack_ext）: 注释掉那一行再装, 不影响
14-bench 评测; 核心链路 = torch / vllm / transformers 5.12.0 /
qwen-vl-utils 0.0.14 / flashinfer-python 0.6.16.post3 / pandas / openpyxl /
tabulate / math-verify / litellm / openai。

## 坑（都在主线踩过, 原样写给你）
0. ★ configs/*.json 的 data 段必须带 class+dataset 显式映射（如
   "MMMU_DEV_VAL": {"class": "MMMUDataset", "dataset": "MMMU_DEV_VAL"}）——
   不要改回空字典 {}。新版 run.py 走 --config 路径时空字典仅对视频数据集
   合法, 图像 benchmark 会全部抛
   "Empty dataset config <name> is not a supported video dataset shortcut"
   秒失败（汇总表全 "-"）。本包的 configs 已是修复后版本, 照抄即可。
1. ★ flashinfer 采样核 JIT: vLLM 0.27.1 引擎 warmup 会 JIT 编译 top-k/top-p
   采样核, 机器没有 nvcc 时直接
   "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
   doesn't exist" 崩溃。judge（vllm serve）和被测模型（run.py 进程内嵌 vLLM
   引擎）都躲不掉, 与模型注意力类型无关。
   解法 = 启动前导出 4 个变量指向你的 CUDA toolkit（run_eval_example.sh 顶部
   已带模板）: CUDA_HOME / PATH(+bin) / LIBRARY_PATH(+lib64) /
   LD_LIBRARY_PATH(+lib64)。机器自带 /usr/local/cuda 时默认模板直接可用。
2. configs/*.json 的 model_path 是绝对路径 —— 必须改。
3. judge 要全程在线: MathVista/MathVerse/MathVision/CharXiv/DynaMath 在 judge
   掉线时会抛错跳过该 benchmark（不是静默 0 分）; 看到日志末尾 Run Summary
   Report 之前别停 judge。
4. shell 里有 http_proxy 时, 发往 127.0.0.1 的 judge 请求会被代理劫走 ——
   模板已带 no_proxy=127.0.0.1,localhost。
5. 中断续跑: 同命令加 --reuse（框架每 10 行 dump 一次预测, 自动接续）。
6. 首次启动 judge 约 3-4 分钟（含 JIT 编译）, 就绪检查见模板; 连续 5 分钟起
   不来 → tail judge 日志看报错。

## 跑
1. 打开 run_eval_example.sh, 改顶部变量（模型路径 / GPU 卡号 / 端口 / 你的
   env python 路径 / CUDA toolkit 路径）。
2. bash run_eval_example.sh   # 起 judge → 就绪探测 → 起 14-bench 评测
3. tail -f 日志; 结束标志 = Run Summary Report 汇总表（每个 benchmark 一行
   分数）。产物在 ${WORK_DIR}/<模型名>/<eval_id>/（xlsx = 预测, _acc.csv = 评分）。

## 可比性
不变项: VLMEvalKit commit ce95c4d + 同一批 TSV + greedy（T=0, top_p 1.0,
top_k -1）+ repetition_penalty 1.05 + max_new_tokens 32768 + max_pixels
4194304 + judge 同为 Qwen3-VL-32B-Instruct。满足这些, 分数与主线同表直接可比
（vLLM 引擎自身的非确定性带来的偏差通常在 0.x 分以内）。

## 参考
- 上游: https://github.com/open-compass/VLMEvalKit（本包 = 其 ce95c4d 快照;
  也可自行 git clone 后 git checkout ce95c4d 替代 VLMEvalKit/ 目录）
- 主线命令出处: Dual-Track-OPD manuscript-tailsft-gpu 第 6 / 7-A 步
EOF_README

cat > "${STAGE}/run_eval_example.sh" <<'EOF_RUN'
#!/usr/bin/env bash
# MMF 14-bench VLMEvalKit eval — teammate template.
# Usage: edit the variables below, then: bash run_eval_example.sh
set -euo pipefail

# ---- variables (edit these) ----
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this bundle dir
MODEL_CFG=base_or_tailsft                     # base | tailsft, or your copied JSON's basename stem
MODEL_CFG_JSON="${BUNDLE_DIR}/configs/qwen3vl_8b_${MODEL_CFG}_mfr.json"  # or point to your copied JSON
EVAL_GPU=0
JUDGE_GPU=1
JUDGE_PORT=8801
JUDGE_MODEL_PATH=/path/to/Qwen3-VL-32B-Instruct
PYTHON=/path/to/vlmeval/bin/python            # your env python
CUDA_TOOLCHAIN=/usr/local/cuda                # nvcc root; no /usr/local/cuda → see README 坑1
WORK_DIR="${BUNDLE_DIR}/outputs"
LOG_DIR="${BUNDLE_DIR}/logs"

[ -f "${MODEL_CFG_JSON}" ] || { echo "FATAL: config not found: ${MODEL_CFG_JSON} (check MODEL_CFG)"; exit 1; }
mkdir -p "${WORK_DIR}" "${LOG_DIR}"

# ---- CUDA toolchain exports (flashinfer sampling-kernel JIT; README 坑1) ----
# Applies to BOTH the judge server and the run.py in-process vLLM engine.
export CUDA_HOME="${CUDA_TOOLCHAIN}"
export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${LD_LIBRARY_PATH:-}"
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost

# ---- 1) judge (keep online for the whole eval; README 坑3) ----
VLLM_BIN="$(dirname "${PYTHON}")/vllm"
setsid nohup env CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" "${VLLM_BIN}" serve "${JUDGE_MODEL_PATH}" \
    --served-model-name Qwen3-VL-32B-Instruct --host 127.0.0.1 --port "${JUDGE_PORT}" \
    --api-key dummy --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > "${LOG_DIR}/judge_32b_${JUDGE_PORT}.log" 2>&1 &
echo "judge starting on GPU ${JUDGE_GPU} (first boot incl. JIT ~3-4 min); log: ${LOG_DIR}/judge_32b_${JUDGE_PORT}.log"

for i in $(seq 1 20); do
    sleep 15
    if curl -s "http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -H 'Authorization: Bearer dummy' \
        -d '{"model":"Qwen3-VL-32B-Instruct","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}' \
        | grep -q choices; then
        echo "judge ready"
        break
    fi
    echo "waiting for judge (${i}/20)..."
    if [ "${i}" = "20" ]; then echo "FATAL: judge not ready after 5 min; tail ${LOG_DIR}/judge_32b_${JUDGE_PORT}.log"; exit 1; fi
done

# ---- 2) 14-bench eval (model under test; interrupt-resume = rerun with --reuse) ----
env CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
    PYTHONPATH="${BUNDLE_DIR}/VLMEvalKit" \
    LMUData="${BUNDLE_DIR}/lmu_data" \
    "${PYTHON}" "${BUNDLE_DIR}/VLMEvalKit/run.py" \
    --config "${MODEL_CFG_JSON}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    --judge Qwen3-VL-32B-Instruct \
    --judge-base-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge-key dummy \
    2>&1 | tee "${LOG_DIR}/vlmeval_${MODEL_CFG}.log"
# 结束标志 = 日志末尾 Run Summary Report 汇总表。
EOF_RUN
chmod +x "${STAGE}/run_eval_example.sh"

cat > "${STAGE}/VERSION.txt" <<EOF_VERSION
packaged_at: $(date '+%Y-%m-%d %H:%M:%S %z')
vlmevalkit_commit: ${VLMEVAL_COMMIT}
vlmevalkit_source: ${VLMEVAL_GIT} (git archive, tree verified clean)
configs_source: ${CFG_DIR}
lmu_data_source: ${LMU_DATA}
env_source: ${ENV_PREFIX} (pip freeze -> env/requirements_eval_freeze.txt)
env_manifest: env/environment_manifest.json (vllm_wheel_sha256 inside)
EOF_VERSION

echo "== tarball =="
rm -f "${TARBALL}"
tar -czf "${TARBALL}" -C "${EVAL_ROOT}" vlmeval_share

echo ""
echo "== done =="
du -sh "${STAGE}" "${TARBALL}"
echo "freeze lines: $(wc -l < "${STAGE}/env/requirements_eval_freeze.txt")"
sha256sum "${TARBALL}" | tee "${TARBALL}.sha256"
echo "bundle dir : ${STAGE}"
echo "tarball    : ${TARBALL}"
