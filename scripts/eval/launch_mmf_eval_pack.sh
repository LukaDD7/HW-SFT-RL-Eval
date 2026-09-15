#!/usr/bin/env bash
# One-command launcher for the MMF 14-bench eval pack.
#
# Usage:
#   bash scripts/eval/launch_mmf_eval_pack.sh [gpu_base] [arms]
#
# Examples:
#   bash scripts/eval/launch_mmf_eval_pack.sh 0 base,tailsft,ptdpo
#   bash scripts/eval/launch_mmf_eval_pack.sh 4 base,tailsft
#
# Notes:
#   - GPU layout: model arms use gpu_base, gpu_base+1, gpu_base+2; judge uses gpu_base+3.
#   - `--reuse` is passed to launch_mmf_eval.sh so interrupted runs resume.
#   - If the PTD-PO arm is requested and its HF export is missing, this script exports
#     global_step_390 from FSDP shards before starting the eval.

set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
ENV_PREFIX="${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1"
CFG_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/vlmeval_cfg"

GPU_BASE="${1:-0}"
ARMS="${2:-base,tailsft,ptdpo}"

PTDPO_CKPT="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_virl39k_ptd_base_4gpu_20260906_r4/global_step_390"
PTDPO_HF="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390"
PTDPO_CFG="${CFG_DIR}/qwen3vl_8b_ptdpo_r4_step390_mfr.json"

BASE_CFG="${CFG_DIR}/qwen3vl_8b_base_mfr.json"
TAILSFT_CFG="${CFG_DIR}/qwen3vl_8b_tailsft_mfr.json"

mkdir -p "${CFG_DIR}"

# Export PTD-PO step390 HF weights if needed.
if [[ "${ARMS}" == *ptdpo* ]]; then
  if [[ ! -f "${PTDPO_HF}/model.safetensors" ]]; then
    echo "== exporting PTD-PO step390 HF weights =="
    SFT_RL_CKPT="${PTDPO_CKPT}" \
    SFT_RL_OUT="${PTDPO_HF}" \
    SFT_RL_GPUS="$((GPU_BASE + 2))" \
      bash "${REPO_ROOT}/scripts/sft_rl/export_hf_from_grpo_ckpt.sh"
  else
    echo "== PTD-PO HF export already exists: ${PTDPO_HF} =="
  fi

  # Ensure the MMF config points at the exported HF directory.
  "${ENV_PREFIX}/bin/python" - "${PTDPO_HF}" "${PTDPO_CFG}" <<'PY'
import json
import sys

hf_path, out_path = sys.argv[1:]
cfg = {
    "model": {
        "qwen3vl_8b_ptdpo_r4": {
            "class": "Qwen3VLChat",
            "model_path": hf_path,
            "use_vllm": True,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
            "max_new_tokens": 32768,
            "max_pixels": 4194304,
            "post_process": True,
        }
    },
    "data": {
        "MMMU_DEV_VAL": {"class": "MMMUDataset", "dataset": "MMMU_DEV_VAL"},
        "MathVista_MINI": {"class": "MathVista", "dataset": "MathVista_MINI"},
        "MathVision": {"class": "MathVision", "dataset": "MathVision"},
        "MathVerse_MINI": {"class": "MathVerse", "dataset": "MathVerse_MINI"},
        "DynaMath": {"class": "Dynamath", "dataset": "DynaMath"},
        "LogicVista": {"class": "LogicVista", "dataset": "LogicVista"},
        "VisuLogic": {"class": "VisuLogic", "dataset": "VisuLogic"},
        "ScienceQA_VAL": {"class": "ImageMCQDataset", "dataset": "ScienceQA_VAL"},
        "RealWorldQA": {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
        "MMBench_DEV_EN": {"class": "ImageMCQDataset", "dataset": "MMBench_DEV_EN"},
        "MMStar": {"class": "ImageMCQDataset", "dataset": "MMStar"},
        "AI2D_TEST": {"class": "ImageMCQDataset", "dataset": "AI2D_TEST"},
        "CharXiv_descriptive_val": {"class": "CharXiv", "dataset": "CharXiv_descriptive_val"},
        "CharXiv_reasoning_val": {"class": "CharXiv", "dataset": "CharXiv_reasoning_val"},
    },
}
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=4)
PY
fi

# Start the shared judge on the last GPU.
JUDGE_GPU="$((GPU_BASE + 3))"
echo "== starting MMF judge on GPU ${JUDGE_GPU} =="
bash "${REPO_ROOT}/scripts/eval/launch_judge.sh" "${JUDGE_GPU}"

# Launch the requested arms.
IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
for arm in "${ARM_LIST[@]}"; do
  case "${arm}" in
    base)
      cfg="${BASE_CFG}"
      gpu="$((GPU_BASE + 0))"
      ;;
    tailsft)
      cfg="${TAILSFT_CFG}"
      gpu="$((GPU_BASE + 1))"
      ;;
    ptdpo)
      cfg="${PTDPO_CFG}"
      gpu="$((GPU_BASE + 2))"
      ;;
    *)
      echo "FATAL: unknown arm ${arm}" >&2
      exit 1
      ;;
  esac
  echo "== launching MMF arm ${arm} on GPU ${gpu} =="
  bash "${REPO_ROOT}/scripts/eval/launch_mmf_eval.sh" "${cfg}" "${gpu}" --reuse
done

echo "== MMF eval pack launched =="
echo "arms: ${ARMS}"
echo "judge: GPU${JUDGE_GPU} :8801"
echo "logs: ${DTOPD_ROOT}/fc-opd-storage/logs/vlmeval_*.log"
