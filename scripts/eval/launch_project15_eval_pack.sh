#!/usr/bin/env bash
# One-command launcher for the project 15-bench eval pack.
#
# Usage:
#   bash scripts/eval/launch_project15_eval_pack.sh [gpu_base] [arms] [parallel]
#
# Examples:
#   bash scripts/eval/launch_project15_eval_pack.sh 0 base,tailsft,ptdpo 2
#   bash scripts/eval/launch_project15_eval_pack.sh 4 base,tailsft,ptdpo 2
#
# Notes:
#   - Each arm uses 2 GPUs: one for the model server and one for its judge.
#   - `parallel` controls how many arms run at once. On a 4-GPU block use 2.
#   - The 15-bench set is split into 11 rule-based tasks and 4 judge-required tasks.

set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
LOG_DIR="${DTOPD_ROOT}/fc-opd-storage/logs"
STATE_DIR="${LOG_DIR}/project15_state"
PROJECT15_CONFIG="${PROJECT15_CONFIG:-${REPO_ROOT}/configs/eval/project_vision_opd.yaml}"
PROJECT15_RUN_SUFFIX="${PROJECT15_RUN_SUFFIX:-}"

GPU_BASE="${1:-0}"
ARMS="${2:-base,tailsft,ptdpo}"
PARALLEL="${3:-2}"

PTDPO_CKPT="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_virl39k_ptd_base_4gpu_20260906_r4/global_step_390"
PTDPO_HF="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_ptdpo_r4_step390"

NO_JUDGE="viewspatial,mindcube,gqa,vqav2,scienceqa,mv_math,remi,dynamath,mmsi_bench,blink,mmmu_pro"
JUDGE="mathverse,mathvista,mmbench,mmvet"

# Export PTD-PO step390 HF weights if needed.
if [[ "${ARMS}" == *ptdpo* ]]; then
  if [[ ! -f "${PTDPO_HF}/model.safetensors" ]]; then
    echo "== exporting PTD-PO step390 HF weights =="
    SFT_RL_CKPT="${PTDPO_CKPT}" \
    SFT_RL_OUT="${PTDPO_HF}" \
    SFT_RL_GPUS="$((GPU_BASE + 0))" \
      bash "${REPO_ROOT}/scripts/sft_rl/export_hf_from_grpo_ckpt.sh"
  else
    echo "== PTD-PO HF export already exists: ${PTDPO_HF} =="
  fi
fi

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"

launch_arm() {
  local arm="$1"
  local slot="$2"
  local model_gpu="$((GPU_BASE + slot * 2))"
  local judge_gpu="$((GPU_BASE + slot * 2 + 1))"
  local eval_port="$((8000 + slot * 2))"
  local judge_port="$((8001 + slot * 2))"
  local model_path=""
  local served_name=""
  local run_name=""
  local arm_log=""
  local arm_pid_file=""
  local arm_status_file=""

  case "${arm}" in
    base)
      model_path="${DTOPD_ROOT}/models/Qwen3-VL-8B-Instruct"
      run_name="project15_base_qwen3vl8b${PROJECT15_RUN_SUFFIX}"
      served_name="Qwen3-VL-8B-Base"
      ;;
    tailsft)
      model_path="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_tailsft_mmf122k_1ep/global_step_1774/huggingface"
      run_name="project15_tailsft_mmf122k_1ep${PROJECT15_RUN_SUFFIX}"
      served_name="Qwen3-VL-8B-TailSFT"
      ;;
    ptdpo)
      model_path="${PTDPO_HF}"
      run_name="project15_ptdpo_r4_step390${PROJECT15_RUN_SUFFIX}"
      served_name="Qwen3-VL-8B-PTDPO-R4"
      ;;
    *)
      echo "FATAL: unknown arm ${arm}" >&2
      exit 1
      ;;
  esac
  arm_log="${LOG_DIR}/project15_${run_name}.log"
  arm_pid_file="${STATE_DIR}/${run_name}.pid"
  arm_status_file="${STATE_DIR}/${run_name}.status"

  echo "== launching project15 arm ${arm}: model GPU${model_gpu}, judge GPU${judge_gpu} =="
  echo "   log: ${arm_log}"
  echo "   pid: ${arm_pid_file}"
  echo "   status: ${arm_status_file}"

  mkdir -p "${STATE_DIR}"
  printf 'launching\n' > "${arm_status_file}"
  printf 'arm=%s\ngpu=%s\njudge_gpu=%s\nstarted_at=%s\n' \
    "${arm}" "${model_gpu}" "${judge_gpu}" "$(date -u +%FT%TZ)" \
    > "${arm_pid_file}.meta"

  setsid nohup env \
    EVAL_CKPT="${model_path}" \
    EVAL_RUN_NAME="${run_name}" \
    EVAL_SERVED_MODEL="${served_name}" \
    EVAL_GPU="${model_gpu}" \
    EVAL_JUDGE_GPU="${judge_gpu}" \
    EVAL_PORT="${eval_port}" \
    EVAL_JUDGE_PORT="${judge_port}" \
    SFT_RL_BENCHMARKS="${NO_JUDGE}" \
    SFT_RL_JUDGE_BENCHMARKS="${JUDGE}" \
    SFT_RL_EVAL_CONFIG="${PROJECT15_CONFIG}" \
    SFT_RL_KEEP_GOING=1 \
    bash "${REPO_ROOT}/scripts/eval/run_target_benchmarks.sh" \
    > "${arm_log}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${arm_pid_file}"
  printf 'running\n' > "${arm_status_file}"
  echo "   pid=${pid}"
  return 0
}

launch_arm_sync() {
  local arm="$1"
  local slot="$2"
  local run_name
  case "${arm}" in
    base) run_name="project15_base_qwen3vl8b${PROJECT15_RUN_SUFFIX}" ;;
    tailsft) run_name="project15_tailsft_mmf122k_1ep${PROJECT15_RUN_SUFFIX}" ;;
    ptdpo) run_name="project15_ptdpo_r4_step390${PROJECT15_RUN_SUFFIX}" ;;
    *) echo "FATAL: unknown arm ${arm}" >&2; exit 1 ;;
  esac

  launch_arm "${arm}" "${slot}"
  local pid
  pid="$(cat "${STATE_DIR}/${run_name}.pid")"
  if ! wait "${pid}"; then
    printf 'failed\n' > "${STATE_DIR}/${run_name}.status"
    return 1
  fi
  printf 'completed\n' > "${STATE_DIR}/${run_name}.status"
  return 0
}

if [[ "${PROJECT15_BACKGROUND:-0}" == "1" ]]; then
  slot=0
  for arm in "${ARM_LIST[@]}"; do
    launch_arm "${arm}" "${slot}"
    slot="$((slot + 1))"
  done
  echo "== project15 eval pack launched in background =="
  exit 0
fi

# Process arms in batches of `parallel`. `launch_arm` starts a detached runner
# and records the real PID, so wait on the recorded PID rather than the helper.
batch=()
slot=0
for arm in "${ARM_LIST[@]}"; do
  batch+=("${arm}")
  if [[ "${#batch[@]}" -eq "${PARALLEL}" ]]; then
    pids=()
    run_names=()
    for i in "${!batch[@]}"; do
      local_arm="${batch[$i]}"
      launch_arm "${local_arm}" "${i}"
      case "${local_arm}" in
        base) run_name="project15_base_qwen3vl8b${PROJECT15_RUN_SUFFIX}" ;;
        tailsft) run_name="project15_tailsft_mmf122k_1ep${PROJECT15_RUN_SUFFIX}" ;;
        ptdpo) run_name="project15_ptdpo_r4_step390${PROJECT15_RUN_SUFFIX}" ;;
      esac
      pids+=("$(cat "${STATE_DIR}/${run_name}.pid")")
      run_names+=("${run_name}")
    done
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        printf 'completed\n' > "${STATE_DIR}/${run_names[$i]}.status"
      else
        printf 'failed\n' > "${STATE_DIR}/${run_names[$i]}.status"
      fi
    done
    batch=()
    slot=0
  else
    slot=$((slot + 1))
  fi
done

# Handle any remaining arms in a final smaller batch.
if [[ "${#batch[@]}" -gt 0 ]]; then
  pids=()
  run_names=()
  for i in "${!batch[@]}"; do
    local_arm="${batch[$i]}"
    launch_arm "${local_arm}" "${i}"
    case "${local_arm}" in
      base) run_name="project15_base_qwen3vl8b${PROJECT15_RUN_SUFFIX}" ;;
      tailsft) run_name="project15_tailsft_mmf122k_1ep${PROJECT15_RUN_SUFFIX}" ;;
      ptdpo) run_name="project15_ptdpo_r4_step390${PROJECT15_RUN_SUFFIX}" ;;
    esac
    pids+=("$(cat "${STATE_DIR}/${run_name}.pid")")
    run_names+=("${run_name}")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      printf 'completed\n' > "${STATE_DIR}/${run_names[$i]}.status"
    else
      printf 'failed\n' > "${STATE_DIR}/${run_names[$i]}.status"
    fi
  done
fi

echo "== project15 eval pack complete =="
echo "arms: ${ARMS}"
echo "parallel: ${PARALLEL}"
