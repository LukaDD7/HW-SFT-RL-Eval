# 项目 Benchmark 的 Vision-OPD Baseline

## 已确认的仓库与 HPC 线索

- Vision-OPD 上游代码以 gitlink 形式固定在 `third_party/Vision-OPD`，记录的提交为 `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`。官方远端 `https://github.com/VisionOPD/Vision-OPD.git` 的 `main` 与该 commit 一致；本次也恢复了之前缺失的 `.gitmodules` 映射。
- 复现训练在 8xH200 上完成到 `global_step=65`。可直接评测的 checkpoint 是：

  ```text
  ${DTOPD_ROOT}/projects/Dual-Track-OPD/third_party/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65
  ```

- `baselines/vision_opd/logs/vllm_vision_opd_32k_tp1.log` 证明该目录已被 vLLM 0.18.0 以 `Vision-OPD-4B` 成功加载。
- 已完成的 Vision-OPD 论文侧评测不是项目合同套件，而是 VStar、ZoomBench、HRBench-4K/8K、MME-RealWorld/中文。与 Qwen3.5-4B base 的宏平均对比为 `74.39 vs 71.35`，但 HRBench 两项下降，因此不能替代项目验收 benchmark。
- Qwen3-VL-8B 项目 benchmark 的第一轮 raw responses 仍在 HPC：

  ```text
  ${DTOPD_ROOT}/eval_runs/qwen3vl8b_baseline/raw_responses
  ```

  Git 分支 `origin/analysis/qwen3vl8b-baseline-scoring` 保存了 16 个 raw 文件的清单和保守诊断结果。这里的结果只能作为 Tier-3 内部诊断，不能冒充官方 metric。

## 合同清单与评测分类

附件要求理解模型在 VQA、数学/科学 Reasoning、综合三类中各选至少两个 benchmark。附件中的 `MatchVista` 按公开任务 `MathVista` 处理，并在配置中保留别名。

配置文件是 `configs/eval/project_vision_opd.yaml`。其中每项明确记录：

- 合同类别与名称；
- lmms-eval task 或历史 raw replay 来源；
- metric tier；
- 确定性打分、规则打分、LLM choice extractor 或 LLM-as-judge；
- primary metric 和最大生成长度。

主后端固定为 `lmms-eval v0.7.1` / `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`。该版本原生覆盖合同中的 13/15 个任务，包括 ViewSpatial、MindCube、GQA、VQAv2、ScienceQA、DynaMath、MathVerse、MathVista、MMSI-Bench、BLINK、MMBench、MMMU-Pro 和 MMVet。MV-MATH 与 ReMI 使用历史 Qwen3-VL raw rows 回放相同 prompt/图像，且始终标为内部诊断。

需要 judge/choice extractor 的任务：

- MathVerse
- MathVista
- MMBench
- MMVet

默认 `--judge-policy defer` 会明确跳过这些指标；`predict` 只生成并保存回答；`score` 才调用 judge。

## HPC 快速运行

GPU 节点无互联网时，先在可联网 CPU 节点准备 lmms-eval 和 Hugging Face 数据缓存。激活已能运行 Vision-OPD/vLLM 的环境后：

```bash
cd ${DTOPD_ROOT}/projects/Dual-Track-OPD
git submodule update --init third_party/Vision-OPD
bash scripts/setup/setup_project_eval.sh
```

先查看完整计划，不加载模型：

```bash
bash scripts/eval/run_project_vision_opd.sh --profile acceptance_core --limit 8
```

推荐使用已经在历史日志中验证过的 OpenAI-compatible vLLM 路径。终端 1：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/start_vision_opd_server.sh
```

历史日志实际验证的是 `max_model_len=32768`；启动脚本默认使用 `65536` 以覆盖此前项目 raw 中的长样本。若当前 vLLM/显存配置无法以 64K 启动，先设置 `VISION_OPD_MAX_MODEL_LEN=32768` 做 smoke，再单独处理超长样本。

终端 2，先做 3 个任务、每项 8 条的 smoke：

```bash
bash scripts/eval/run_project_vision_opd.sh \
  --profile smoke \
  --limit 8 \
  --run-name smoke_vision_opd_gs65 \
  --execute
```

跑无 judge 的完整清单，并保留 judge 项为 deferred：

```bash
bash scripts/eval/run_project_vision_opd.sh \
  --profile all \
  --judge-policy defer \
  --keep-going \
  --run-name all_no_judge_vision_opd_gs65 \
  --execute
```

若暂时没有 judge，但想先生成 judge 类任务回答：

```bash
bash scripts/eval/run_project_vision_opd.sh \
  --benchmarks mathverse,mathvista,mmbench,mmvet \
  --judge-policy predict \
  --run-name judge_inputs_vision_opd_gs65 \
  --execute
```

接 judge 时显式记录 endpoint、模型和 prompt 所依赖的 lmms-eval commit：

```bash
export JUDGE_API_KEY='...'
export JUDGE_API_URL='https://your-endpoint/v1/chat/completions'
export JUDGE_MODEL='your-judge-model'
bash scripts/eval/run_project_vision_opd.sh \
  --benchmarks mathverse,mathvista,mmbench,mmvet \
  --judge-policy score \
  --run-name judged_vision_opd_gs65 \
  --execute
```

如不希望常驻服务，也可以对 lmms-eval 原生任务使用 `--inference-backend vllm`。MV-MATH/ReMI replay 依赖 API endpoint，在该模式下会被明确标为 deferred。

运行结束后生成统一的逐项与分类摘要：

```bash
bash scripts/eval/summarize_project_vision_opd.sh \
  ${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/all_no_judge_vision_opd_gs65
```

输出为同目录下的 `summary.json` 和 `summary.csv`。分类宏平均默认排除 Tier-3 内部诊断，避免 MV-MATH/ReMI 的临时打分污染验收结论。

## 输出与复现记录

每次真正运行都会在 Git 之外的 output root 新建目录并写入 `run_manifest.json`，其中包含：

- 当前 repo commit、dirty status 和完整 status；
- lmms-eval tag/commit；
- 完整 resolved config；
- 由 task/source 清单和后端 commit 计算的 dataset manifest hash；
- checkpoint 路径；
- raw output 根目录；
- 每个 benchmark 的命令、metric tier、scoring 分类、状态与 return code。

不要把 raw lmms JSONL、judge 输出、数据集或 checkpoint 加入 Git。只同步最终摘要、表格和必要的小型审计样本。
