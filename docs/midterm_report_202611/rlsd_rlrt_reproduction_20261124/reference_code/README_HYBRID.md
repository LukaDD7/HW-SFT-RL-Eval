# ZwZ step150 → MMFineReason RLSD + RLRT-style Stage2

## 实验定义

本目录是独立实验代码；基础来自官方 RLSD commit
`aca313b978de4fb3a509ddf1831bf61fe597a2eb`。保留 FSDP、vLLM、Ray 训练主干。
`hybrid` 是我们对 VLM 的组合适配，不能称为 RLRT 原文的完整复现：
teacher 只看到原始图文题目和独立最终答案；RLRT 原文使用正确 rollout 作为上下文。
筛选数据已接入共享目录，见下方“筛选题单接入”。正式训练必须显式提供 canonical train/val JSONL。

来源：

- RLSD: https://arxiv.org/html/2604.03128v2
- RLSD 官方代码: https://github.com/iie-ycx/RLSD
- RLRT: https://arxiv.org/html/2605.10781v1

## 默认参数（8 × H200）

| 项目 | 设置 | 说明 |
|---|---|---|
| 初始化 | `/zxx/zwz-repro/models/ZwZ-8B-stage1-step150/huggingface` | 新 Stage2，从本阶段 step0 开始 |
| 学习率 | 常数1e-6，无warmup | RLSD 论文设置；不随teacher权重衰减 |
| 每步题目 / 每题采样 | 256 / 8 | 2048 条轨迹；global_batch 内部乘 G |
| actor PPO epochs | 1 | 有非零优势时，一个 rollout pool 对应一次全局优化器更新 |
| prompt / response 上限 | 4096 / 4096 | teacher 也执行长度检查，禁止静默截断 |
| 采样温度 / top_p | 1 / 1 | 同批学生轨迹给 teacher 强制解码评分 |
| PPO clip low/high | 0.2 / 0.28 | RLSD 论文设置 |
| KL / entropy 奖励 | 关闭 | 只用可验证二元正确性 |
| λ | 0.5，50 步线性衰减至 0 | step0=0.5、global_step=1 时=0.49、step50=0 |
| teacher 权重 clip | [0.8,1.2] | 正负分支先使用相同范围 |
| teacher 同步 | 每10步硬同步 | checkpoint 保存实际滞后 teacher |
| 训练长度 | 200 步 | 可通过 `--max-steps` 指定累计本阶段步数 |
| checkpoint | 每50步，全部保留，不自动删除 | `save_limit: -1`；包含 actor/optimizer/scheduler/RNG/teacher/dataloader |
| 验证 | 每50步、结束时；启动前不验证 | 工程调整，正式 benchmark 由用户提供 |
| 预处理 | 64进程；dataloader 8 workers | 过滤超长 student prompt；teacher 溢出明确报错 |
| FSDP / dynamic batching | full shard / 开启，8192 tokens/GPU | H200 配置待实际8卡验收 |
| CPU offload | 默认关闭 | 提升吞吐；`--memory-conservative` 恢复 CPU offload |

8张卡均用于 actor/rollout/teacher；二元判分在 CPU 执行，不另占 judge GPU。

官方公开 shell 的部分覆盖值是 decay60 / sync20；这里选择论文正文的
decay50 / sync10。固定 λ 是额外实验，通过 `--schedule fixed` 显式选择，不能说成论文默认。
默认 paper schedule 在50步后变成 GRPO 更新，代码同时停止 teacher prompt、forward 与同步。
λ 按 global_step（rollout 迭代）计数；全零优势导致跳过优化器更新时，也会消耗衰减步数。
RLRT 原文 instruct 设置使用 decay30、最佳 weight clip0.5；这里四种方法统一使用 RLSD 的
decay50 / clip0.2，属于控制实验，不代表复现 RLRT 最佳超参数。

## 教师持续时间与正分支强度组合实验

新增参数支持同时开启以下设置；不传参数时仍保留原 decay50 / 双分支 clip0.2 基线：

| 参数 | 组合实验设置 | 行为 |
|---|---|---|
| `--teacher-decay-steps` | 100 | λ 从0.5线性衰减，第1步0.495、第50步0.25、第99步0.005、第100步0 |
| `--positive-clip-range` | 0.5 | A>0 的教师权重范围 `[0.5,1.5]` |
| `--negative-clip-range` | 0.2 | A<0 的教师权重范围 `[0.8,1.2]` |

启动前先沿用原来的 `dataset-current-reward.env` 设置 `TRAIN_DATA`、`VAL_DATA` 和图像路径，
然后使用新的 run-name：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_hybrid.sh \
  --run-name hybrid-teacher100-pos05-stage1step150 \
  --teacher-decay-steps 100 \
  --positive-clip-range 0.5 \
  --negative-clip-range 0.2 \
  --preflight-only
```

预检查通过后去掉 `--preflight-only` 即为同一组合实验的训练命令。
正分支的实际优势乘数范围为 `[1-0.5λ,1+0.5λ]`，负分支为 `[1-0.2λ,1+0.2λ]`；
正负分支按原始优势符号选择，教师评分保持停止梯度。
模型、题单、8次采样、学习率、教师每10步同步、PPO clip 与原实验一致。
第100步起教师评分和同步关闭，恢复 GRPO 更新；本次不增加全错蒸馏分支。

`--teacher-decay-steps` 优先于默认 paper/smoke 的衰减步数；
`--schedule fixed` 只允许衰减步数0或不传覆盖值。
两个分支裁剪参数未设置时，回退到公共 `opsd.rlsd_reweight_clip_range`；
公共值也未设置时沿用 actor 的 PPO clip low。分支参数不会改变 PPO current/old ratio 的裁剪。

这是新的组合实验配置，效果尚待验证。比较原 decay50 与新 decay100 时，应从同一个 Stage1 step150
起点另起训练。由已经训练的 Stage2 step150 继续只能算续训实验，无法补回原前100步的教师信号。
旧运行 manifest 的代码/配置校验继续生效，不修改旧 manifest 强行续跑。

## 四种方法

所有方法均使用相同数据、二元 reward、组内标准差归一化、PPO student current/old ratio。
设 δ=stopgrad(logp_teacher−logp_student)，A 为组内优势，最终优势为
`A * ((1-lambda) + lambda * clip(w,0.8,1.2))`。

| `--method` | A>0 | A<0 |
|---|---|---|
| grpo | 普通优势 | 普通优势 |
| rlsd | w=exp(+δ) | w=exp(−δ) |
| rlrt | w=exp(−δ) | 普通优势 |
| hybrid | w=exp(−δ) | w=exp(−δ)，即原 RLSD 负分支 |

`rlrt` 是统一 answer-only 上下文的控制组，不是原论文完整流程。
权重在 log 空间截断，避免 exp 溢出；权重不回传 teacher/student 梯度。
Teacher prompt 每题只构建一次，再按 UID 对齐到8条 response；图像顺序和 mRoPE 重新计算。

## 判分与有效题组

`rlsd_rlrt_binary.py` 不把格式分加到 reward 中；只输出 binary accuracy。
支持明确的数学数值/元组/有限算术、选择题、布尔、显式 exact_match/OCR。
未实现通用语义判断：不同写法且不能可靠解析的符号式、开放文本或超时/异常均为 UNKNOWN，
不能把它们强行当错误。接入最终数据时必须统计 UNKNOWN 比例，并与筛选判分器做一致性审计。

- 同一题8条中任一 UNKNOWN：整组 loss mask 为0。
- 全对 / 全错：组内优势为0；保留可判分 token 的分母，和普通 GRPO 的归一化一致。
- 混合组：有效 RL 题组。
- 全局没有非零优势：所有 rank 一起跳过 actor 更新，避免仅因 Adam 动量/weight decay 改模型。
- 每步日志记录 `groups/mixed_count`、`all_correct_count`、`all_wrong_count`、`unjudgeable_count`，
  以及 `perf/effective_questions_per_second`。

## 数据入口

正式训练只接受 JSONL，每行至少包含：

```json
{"problem_id":"q1","problem":"<image>\n题目","answer":"独立最终答案","images":["/绝对路径/图像.png"],"problem_type":"math","data_type":"image"}
```

`problem_id`（也支持 index/id/question_id）必须唯一，train/val 不得重叠。
图片数须匹配 `<image>` 数。推荐绝对图片路径；相对路径默认基于 train 文件目录，
可通过 `--image-root` 指定双方统一根目录。不要把完整教师思维链放进 answer。
问题 ID 清单需要先转换成上述完整数据行；没有路径时程序拒绝启动，不会回落到旧全量数据。

## 启动（只在8卡 H200 上执行）

```bash
cd /zxx/zwz-stage2/rlsd-rlrt-20261007/repo
export TRAIN_DATA=/你的最终训练集/train.jsonl
export VAL_DATA=/你的独立验证集/val.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_hybrid.sh --run-name hybrid-paper-step150 --preflight-only
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_hybrid.sh --run-name hybrid-paper-step150 >hybrid-paper-step150.log 2>&1 </dev/null &
```

固定 λ 实验使用另一个 run-name：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_hybrid.sh --schedule fixed --run-name hybrid-fixed-step150 >hybrid-fixed-step150.log 2>&1 </dev/null &
```

断点续训需沿用原数据/方法/模型/schedule，明确给 `--resume latest`：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_hybrid.sh --run-name hybrid-paper-step150 --resume latest >hybrid-paper-step150-resume.log 2>&1 </dev/null &
```

日志简要进度：

```bash
grep -aE 'HYBRID_STEP_DONE|Load from checkpoint|Traceback|Error' hybrid-paper-step150.log | tail -n 30
tail -n 1 runs/hybrid-paper-step150/experiment_log.jsonl
cat runs/hybrid-paper-step150/checkpoint_tracker.json
```

## 恢复约束

保存流程：各rank写 actor/optim/extra/teacher → barrier → driver写 dataloader/RNG →
验证文件与大小 → 写 `checkpoint_complete.json` → 原子更新 tracker。
默认 `save_limit: -1`，不清理旧 checkpoint；200步训练会保留 step50、100、150、200。
只有显式设置正数 `save_limit` 时，才在提交成功后按上限清理旧的完整版本。
未完成的目录不会被当作可恢复 checkpoint。缺分片、大小不符、GPU world size 变化、
JSONL 标注、配置、代码或模型分词器元数据变更会拒绝续训。
此检查不是磁盘内容逐字节校验；它不能检测相同文件大小的静默损坏。
引用图片和初始化模型的大权重文件未逐字节计算哈希，必须保持这些共享路径内容不变；
当前 manifest 不能证明同一路径的图片像素/初始化权重没有被替换。

每次启动留存 `run_manifest.json` 和 `resolved-*.yaml`。旧 RLSD 运行不使用新事务格式，
不能直接作为这个新实验的断点；从 Stage1 HF 导出初始化是另一种操作。
新 checkpoint 的 teacher 可以精确恢复，不能用当前 actor 覆盖滞后 teacher。

## 环境与验证范围

Python 和 venv 均在共享 `/zxx/zwz-repro` 内，默认使用 `venv-official/bin/python`。
新代码通过 PYTHONPATH 隔离，不覆盖旧 RLSD。H200 无需联网下载依赖或模型。
仍须保留兼容的实例镜像/系统库/NVIDIA driver，以及同一个共享目录挂载。
本分支将依赖声明对齐已验证的 Transformers 4.57.6；旧 grader 的 `math_verify`
改为可选依赖 `legacy-grader`，新的二元判分器不使用它。旧 RLSD 目录和 venv 未重装。

4090 的集成测试使用2B模型、短序列、小批次；`--synthetic-reward-smoke` 使用显式测试 reward
保证正负分支都经过 backward，不代表模型准确率。单独对真实 step150 8B 做图文推理检查。
这些测试不能证明8卡 H200 的 NCCL、峰值显存和训练吞吐已经通过；第一次 H200 启动仍需短跑验收。

Transformers4.57.6 对保存后 Qwen3-VL tokenizer 发出的 Mistral regex 警告是版本判定误报：
已比较 base 和 step150 的 runtime-normalized tokenizer 及 chat template，完全一致。
不要设置 `fix_mistral_regex=True` 修改 Qwen 的分词正则。

## 2026-10-07 验证结果

- 服务器105项CPU回归测试通过；依赖检查与主要模块导入通过。
- 真实Stage1 step150 8B模型：3张验证图像的加载和生成通过，不是准确率测试。
- 4090上2B集成测试：正负分支均完成反向传播，保存step1，独立重启后恢复到step2。
- teacher滞后权重、optimizer步数、scheduler和dataloader位置恢复正确；训练与rollout RNG均保存。
  这不等于保证整个vLLM采样过程跨重启逐bit一致。
- step2未写完时step1仍保留；完整提交后才删除step1。最终仅留完整step2测试checkpoint。
- λ=0 时 teacher prompt/forward跳过；实际编译log-prob算子的前向与梯度检查通过。
- 8×H200配置解析与四方法入口检查通过；真实8卡分布式训练、长序列峰值显存和吞吐未验证。
- 以上GPU验收早于正式题单接入，使用合成reward；正式训练不会使用合成reward，且尚未启动。

报告位于上一级目录 `validation-summary.json`、`runtime-audit.json`、`environment-freeze.txt`。
完整测试记录在 `smoke-v3-first.log`、`smoke-v3-resume.log`、`unit-tests.log`。
以上集成测试已结束并释放其GPU上下文。

## 筛选题单接入（2026-10-07）

共享数据目录：`/zxx/rlsd-data/selections/step150-mixed-20261007/`。

- `dataset/train.jsonl`：完整34,624题；`dataset/val.jsonl`：来源中独立留出的521题。
- `dataset-current-reward/train.jsonl`：当前判分器支持的33,316题；对应验证集300题。
- 不兼容的1,308道训练题和221道验证题保留在完整数据及子集目录的`excluded-*.jsonl`中。
- 通过`original_index`精确匹配；补齐1道旧数据缺失题；30,550张独立图片检查无损坏。
- 题号无重复、训练/验证无同题同图交集。筛选与训练判分器不同，历史“混合题”不保证新采样仍为混合题。
- 完整版本不能直接通过当前reward预检；本次未修改reward、训练代码或训练参数。

当前兼容数据的显式环境变量：

```bash
source /zxx/rlsd-data/selections/step150-mixed-20261007/dataset-current-reward.env
```

来源、判分差异及排除明细见该数据目录下的`DATA_HANDOFF.md`、`REWARD_COMPATIBILITY.md`、
`full-data-audit.json`。正式训练仍未启动。
