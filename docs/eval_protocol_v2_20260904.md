# 评测协议 v2 — 修正长 CoT 检查点的系统性评测伪影

日期：2026-09-04
关联审计：`docs/mmf_sft_reproduction_audit_20260903.md`（§3 量化了伪影）
配置：`configs/eval/project_vision_opd_v2.yaml`
任务覆盖：`eval_tasks/opd_v2/`（gqa_v2 / viewspatial_v2 / mmmu_pro_standard_v2 / dynamath_reasoning_v2 + `_extraction.py`）
入口脚本：`scripts/eval/run_target_benchmarks_v2.sh`
测试：`tests/eval_v2/test_opd_v2_extraction.py`（19 例，fixture 来自 v1 基线真实响应）

## 1. 为什么需要 v2

v1 评测（`configs/eval/project_vision_opd.yaml` + lmms-eval 官方任务）对长 CoT
模型（MMF-SFT 谱系）有三类系统性不利伪影，审计 §3 量化如下：

| 伪影 | 数据集 | v1 现象 | 量级 |
|---|---|---|---|
| 解码预算截断 | ViewSpatial / GQA / MMMU-Pro | 256/128/2048 上限，CoT 未完 | ViewSpatial mmf 臂 68.5% (3,911/5,712) 行 ≥950 字符 |
| 首字母抽取 | MMMU-Pro | official parser 取 CoT 中首个字母 | 9/1,730 行 answer≠parsed 且标签答案正确 |
| 随机 fallback | MMMU-Pro | 未解析时 `random.choice` 注入噪声 | prose 响应直接得非确定字母 |
| stale 首标签 | DynaMath | `re.search` 取第一个 `<answer>` | 435/5,010 行含 ≥2 个标签 |
| 截断+抽取复合 | ViewSpatial | 0.0940 中约六成是伪影 | 换「停表文本任意字母」口径约 0.25~0.28 |

v2 的目标：**每个数据集的 prompt/解码预算/判分链都按官方口径与数据特点修正**，
让 MMF-SFT 与 base 的对比反映真实能力差，而不是协议差。

## 2. 全局机制修正（四项共享）

### 2.1 解码预算：4096 + 明确的 openai clamp 语义

openai 后端（`lmms_eval/models/simple/openai.py:443`）硬钳制
`max_new_tokens = min(requested, 4096)`。v1 dynamath yaml 写 49152 实际按
4096 解码但配置表里显示 49152；v2 统一声明真实预算 **4096**，配置即行为。

（`until` 停表序列在 openai 后端不生效——lmms-eval 不传 `stop`，截断纯由
max_tokens 决定；v2 不依赖 until。）

### 2.2 `<answer>` 标签优先判分链

MMF-SFT 训练目标以 `<answer>...</answer>` 结束长 CoT（dynamath 官方
SYSTEM_PROMPT 同款约定）。v2 对每个 MCQ/short-answer 任务实现统一的抽取链：

1. `extract_answer_tag`：取**最后一个** `<answer>` 标签内容（后验优先于先验）；
2. `extract_option_letter`：标签内取独立大写选项字母（`"B"` 命中，
   `"therefore"` 的 T 不算——独立成词才有效）；
3. 无标签时回退官方 parser 的候选扫描（`(X)` / `"X "` / `"X."` / 选项文本
   包含），但**移除 `random.choice` 分支**（见 §2.3）；
4. base 检查点的裸字母直接回答（`"A"`、`"Yes"`、`"catcher"`）在无标签路径
   正常命中——v2 对两臂可比。

### 2.3 移除 random.choice（MMMU-Pro 确定性）

官方 `parse_mmmu_multi_choice_response` 在无任何候选命中时
`random.choice(all_choices)` 返回一个**合法字母**（truthy），事后无法区分
「真解析」与「随机」。v2 复刻官方候选扫描（`_deterministic_candidate_found`），
在 parser 会走随机分支时显式返回空预测（记为 unparsed，判错但不注入噪声），
否则照常走官方 parser。判分完全确定可复现。

### 2.4 TaskManager 命名（v2 后缀）

lmms-eval `--include_path` 的目录索引晚于内置任务，同名任务在
`task_index = {**tasks, **task_index}` 合并中**落败**。所有 v2 任务用新名
（`viewspatial_v2` 等），且 `benchmark_suite` 通过 suite defaults 的
`include_task_path` 自动给每条命令带 `--include_path eval_tasks/opd_v2`。

## 3. 分数据集口径

### 3.1 ViewSpatial-Bench（viewspatial_v2）

- **预算**：256 → **4096**。
- **prompt**：保留官方问题渲染；answer 指令改为
  `Answer with the option letter inside <answer></answer> tags.`（贴合
  MMF-SFT 训练约定；base 臂无标签时走 fallback 路径）。
- **判分**：`<answer>` 标签（内取独立字母）优先；无标签回退**最后一行**的
  独立字母（`last_line_option_letter`）；都失败记 unparsed（判错，诚实口径）。
- **证据**：mmf 臂 v1 只有 20/5,712 行含标签；4096 预算下标签得以完整输出，
  复合伪影消除。truncated CoT 行的 0 分保留为真实能力差（模型未给出可判答案）。

### 3.2 GQA（gqa_v2）

- **预算**：128 → **4096**。
- **prompt**：保留官方单词 post_prompt；追加 answer 指令同上（标签式）。
- **判分**：官方 `normalize_word` + exact match 链不变；`short_answer` 抽取
  优先取 `<answer>` 标签内容，无标签时整条响应 strip 后为预测（base 臂
  `"Yes"` 直答可正常对 `yes`）。
- **证据**：mmf 臂 GQA 0.3759 部分受 128 截断影响；v2 量化残余真实差。

### 3.3 DynaMath（dynamath_reasoning_v2）

- **预算**：声明 4096（真实行为，见 §2.1）。
- **prompt**：官方 SYSTEM_PROMPT（`<answer>` 约定 + think 标签）不变，
  doc_to_text/messages/visual 全用官方函数——**不引入新 prompt 逻辑**。
- **判分**：官方 `compute_score` 链（0.9·acc + 0.1·format）原样保留；唯一改动
  是送入前 `_strip_stale_answer_tags` 删除非末位 `<answer>` 块，让官方
  first-match `re.search` 落在**最后一个**（settled）标签上。
  实测上游链对 `<answer>7.1</answer> ... <answer>8.062</answer>` 判 0 分，
  v2 判 1 分。
- **注**：math_verify 无数值容差（verify(8.0, 8.062)=False）——这是官方链行为，
  v2 保持一致以可比；审计 §3.3 的 0.5695/0.3739 数字来自**数值归一化重判**
  （更宽松口径），与 v2 的官方链数字不可直接比较。

### 3.4 MMMU-Pro（mmmu_pro_standard_v2)

- **预算**：2048 → **4096**（v1 mmf 臂 55.1% 响应撞 2048 cap）。
- **prompt**：官方问题/选项渲染保留；answer 指令同 v2 统一标签式（官方
  `"Answer with the option letter from the given choices directly."` 对长 CoT
  模型不友好——模型确实会先推理后给标签，指令不匹配导致标签后仍继续生成）。
- **判分**：`mmmu_pro_v2_extract_pred`：标签 → 独立字母 → in-choices 校验；
  标签内无有效字母记 unparsed（不让 CoT 杂散字母冒领）；无标签时官方 parser
  候选扫描（随机分支已移除，§2.3）。
- **聚合**：官方 subject 分组 instance 级平均（与上游一致）。

## 4. 可比性边界（诚实口径）

- v2 与 v1 结果**并行保留**（`eval_runs/vision_opd_project_v2` vs
  `eval_runs/vision_opd_project_baseline`），不覆盖。跨版本数字不可直接
  对比时以 v2 为准（v1 的伪影单向利好 base 臂）。
- 与 MMFineReason 论文数字对比：论文评测集不含 GQA/ViewSpatial/MMMU-Pro/ReMI；
  唯一同名 DynaMath 论文用 LLM judge（更宽松），v2 用官方规则链——比较时
  注明判分器差异。
- v1 与 v2 均未加 `--reasoning-parser qwen3`（服务端不剥离 think 块）。
  两版一致故可比；但这意味着判分输入是完整响应（含 CoT），标签优先抽取正是
  为此设计。若未来启用 reasoning parser，判分链无需改动（标签仍在）。
- ReMI（replay 口径）与 MMBench（judge 口径）不在 v2 范围：审计未发现其
  判分链有伪影。

## 5. 运行

```bash
# GPU 节点（需 benchmark 数据已在共享 HF 缓存）
bash scripts/eval/run_target_benchmarks_v2.sh

# 每项 8 条冒烟
EVAL_SMOKE=1 bash scripts/eval/run_target_benchmarks_v2.sh

# 指定检查点/名字
EVAL_CKPT=/path/to/hf_model EVAL_RUN_NAME=my_ckpt \
  bash scripts/eval/run_target_benchmarks_v2.sh
```

脚本自管 vLLM eval server（含 FlashInfer JIT 的 CUDA toolchain 导出，
无 judge server），断点续跑走 `--resume-from`，汇总走
`project_summary --config project_vision_opd_v2.yaml`。

## 6. 验证

- 单测：`pytest tests/eval_v2/test_opd_v2_extraction.py -q`（19 例，fixture
  为 v1 基线真实响应形状：tagged/truncated/direct/stale-tag/multi-tag）。
- 套件 dry-run：`benchmark_suite --config configs/eval/project_vision_opd_v2.yaml
  --benchmarks gqa,dynamath,viewspatial,mmmu_pro` 打印的每条命令均带
  `--include_path eval_tasks/opd_v2`、`max_new_tokens=4096`、`temperature=0`、
  v2 任务名。
