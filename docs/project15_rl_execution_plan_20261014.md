# Project15-to-RL pinned execution plan — 2026-10-14

## Purpose

This plan pins the next 4-GPU execution order after the ViRL39K VA-OPD run.
The objective is Project15 capability, not a ViRL-only score:

1. Establish the Project15 transfer of the exported VA-OPD best checkpoint.
2. Build the next RL pool from ViRL-medium plus Cauldron-broad.
3. Run a bounded 50-step GRPO pilot.
4. Select checkpoints using Project15 category macro, not ViRL validation alone.

This plan does not redefine the HWEval benchmark protocols. `run_pinned_eval.sh`
remains the only user-facing evaluation entrypoint.

## Fixed paths and 4-GPU allocation

```bash
export HW_EVAL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/HW-SFT-RL-Eval
export DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export HW_EVAL_ENV="${DTOPD_ROOT}/envs/vision-opd-cu128"
export EVAL_CUDA_TOOLCHAIN="${DTOPD_ROOT}/envs/cuda128-toolchain"
export DTOPD_DATASET_ROOT="${DTOPD_ROOT}/dataset"
export EVAL_HF_CACHE="${DTOPD_ROOT}/.cache/huggingface"
export DTOPD_EVAL_ROOT="${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline"

export EVAL_CKPT="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_va_opd_step2175"

# B6-v2 parallel servers: GPU 0 and GPU 1.
export EVAL_GPU=0
export EVAL_V2_SECOND_GPU=1
export EVAL_PORT=8000
export EVAL_V2_SECOND_PORT=8001

# v1 ReMI/MMBench stage: model on GPU 0, judge on GPU 2.
export SFT_RL_EVAL_GPU=0
export SFT_RL_JUDGE_GPU=2
export SFT_RL_EVAL_PORT=8000
export SFT_RL_JUDGE_PORT=8002
export SFT_RL_JUDGE_HF="${DTOPD_ROOT}/models/Qwen3-VL-32B-Instruct"

cd "${HW_EVAL_ROOT}"
```

GPU 3 is deliberately reserved for recovery or a second sequential stage. Do
not start two full evaluation protocols concurrently unless their ports,
run names, and output roots are disjoint.

## Stage 1 — Project15 transfer of VA-OPD step2175

Run B6 first, then the nine-task complement. Keep the same checkpoint and
prompt mode for both runs.

### 1a. B6-mixed

```bash
export EVAL_RUN_NAME=project15_vaopd_step2175_b6_auto_avg4
export EVAL_THINK_MODE=auto
export EVAL_V2_PARALLEL=1

bash scripts/eval/run_pinned_eval.sh \
  --protocol b6-mixed \
  --think-mode auto
```

### 1b. Project15 complement

```bash
export EVAL_RUN_NAME=project15_vaopd_step2175_complement_auto_avg4
export EVAL_THINK_MODE=auto

bash scripts/eval/run_pinned_eval.sh \
  --protocol project15-complement \
  --think-mode auto
```

Reporting rules:

- Combine the rows for reporting only after both runs complete.
- Do not average B6 and Project15-complement into one protocol macro.
- Label each row with its source protocol and task version.
- Compare against the pinned base/PTD-PO Project15-v1 rows only where the task
  version and scoring chain match.

### Stage 1 decision gate

Proceed to RL if at least one of these holds:

- Math/science shows a positive trend on DynaMath, MathVerse, or MathVista.
- Comprehensive shows a positive trend on MMSI-Bench or MMMU-Pro.
- A diagnostic such as ReMI improves without broad VQA/comprehensive loss.

If all Project15 categories regress, fix the checkpoint or evaluation issue
before RL. Do not use a ViRL-only gain as permission to continue.

## Stage 2 — next RL pool: ViRL-medium + Cauldron-broad

The existing `rl_mixed_31k` is not the next default because it is MMF-hard
dominated and contains no ViRL. Keep it archived as the historical mixed-pool
baseline.

Build a new pool, preferably named:

```text
${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/virl_cauldron_rl_v2
```

Target size: 24,000–32,000 rows.

Recommended composition:

| Source | Share | Selection rule |
|---|---:|---|
| ViRL39K medium | 45–55% | Step2175 pass rate between 0.2 and 0.8; source-stratified |
| Cauldron broad | 45–55% | Existing rule-ok pool; retain the 14-subset broad mix |
| MMF hard | 0–10% optional | Only step2175 medium-difficulty rows with a useful signal |

ViRL source priorities:

- MMK12 and Processed
- MMMath and K12
- M3CoT, ScienceQA, AI2D, ChartQA, DVQA
- Retain smaller sources by stratification rather than dropping them entirely

Cauldron priorities:

- AI2D, ScienceQA, ChartQA, IconQA, InfographicVQA
- DocVQA, TextVQA, Visual7W, VQAv2, VSr

The first pilot should use pure GRPO (`PTD_ENABLE=false`). Hints are not
required for this pool. Add PTD only after the GRPO baseline demonstrates the
pool transfers to Project15.

## Stage 3 — 50-step GRPO pilot

Run from the Dual-Track-OPD repository after the new pool exists:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

export SFT_RL_GPUS=0,1,2,3
export SFT_RL_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/hf/qwen3vl_va_opd_step2175
export SFT_RL_TRAIN=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/virl_cauldron_rl_v2/virl_cauldron_train__part_*.parquet
export SFT_RL_NAME=qwen3vl_vaopd_step2175_virl_cauldron_grpo50
export TOTAL_TRAINING_STEPS=50
export TRAINER_SAVE_FREQ=25
export TRAINER_TEST_FREQ=10
export PTD_ENABLE=false

bash scripts/sft_rl/run_grpo_mmf_ptd.sh
```

Do not substitute the old MMF-hard pool unless deliberately running the
historical control arm.

## Stage 4 — evaluate RL checkpoints

Evaluate step25 and step50 with the same Stage 1 commands, using new run names:

```text
project15_vaopd_step2175_grpo25_b6_auto_avg4
project15_vaopd_step2175_grpo25_complement_auto_avg4
project15_vaopd_step2175_grpo50_b6_auto_avg4
project15_vaopd_step2175_grpo50_complement_auto_avg4
```

Primary comparison:

1. Step2175 before RL
2. GRPO step25
3. GRPO step50
4. Pinned Qwen3-VL-8B base

Decision rules:

- Track the three Project15 category macros separately.
- The primary attack categories are math/science and comprehensive.
- VQA may decrease slightly if the overall target is met.
- Prefer the checkpoint with the best balanced Project15 category macro, even
  if its ViRL validation score is not the highest.
- If only ReMI improves while Project15 native rows regress, the pool has
  drifted toward hard long-chain reasoning and must be rebalanced.

## Current interpretation

The strongest Project15 opportunities are:

- Math/science: DynaMath and MathVerse, with MathVista as a secondary check.
- Comprehensive: MMSI-Bench and MMMU-Pro, with MMBench/BLINK as defensive rows.
- VQA: MindCube and GQA are the most useful rows; ScienceQA and VQAv2 are near
  saturated and should be treated as defensive rows.

ViRL remains useful because it contains substantial medium-difficulty math,
science, MCQ, chart, document, and diagram material. MMF-hard is useful only as
a small, signal-filtered supplement; it must not dominate the next pool.
