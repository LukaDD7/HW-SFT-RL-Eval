# RLSD＋RLRT：Base 混合 30k 与 ISO＋MOPD 后续 15k 的方法与复现说明

版本：2026-11-24。对象：Qwen3-VL-8B-Instruct 架构，单节点 8×H200。更新自 2026-11-21 交接包；保留原算法、评分器和训练参数，只按本次约定调整训练数据。

这份说明描述我们称为 `hybrid` 的组合方法：保留 GRPO 的组内二元优势和 PPO 更新，用答案条件自教师的概率对正确、错误轨迹分别分配 token 信用。它适用于原始 Base，也适用于 ISO＋MOPD 已训练模型。

## 0. 先区分统一的方法、历史实验和已核验材料

**本版两条实验采用相同的 RLSD＋RLRT 算法、OSS v3 训练评分器、提示模板与 8×H200 参数；初始化和训练题单按下表分别设置。**

| 实验 | 学生与自教师初始化 | 本次训练数据 | 合并前条数 |
|---|---|---|---:|
| Base 后接 hybrid | `Qwen/Qwen3-VL-8B-Instruct` | MOPD-route-v2-train-15k ＋ RL-RSTG-ZVP-train-15k，全量拼接 | 30,000 |
| ISO＋MOPD 后接 hybrid | route-v2 MOPD step351 | 仅 RL-RSTG-ZVP-train-15k | 15,000 |

Base 的两份 15k **全量保留，先拼接，不去重、不各抽 7.5k，不做额外来源加权**；交给原 dataloader 按 seed1 均匀打乱。1∶1 是两仓库记录数之比，不要求每个 batch 恰好各 128 题。重复题若存在，保留并在数据清单报告；30,000 指记录数，不预先等同于 30,000 道独立题。

除上述初始化、训练数据及必需的路径/run 名外，其他配置沿用第 8 节。训练评分器仍为 `rstg_oss_verifier_v3`，不切换为旧 Base 的 MMFineReason 评分器。两条实验的数据不同，因此结果用于比较这两条训练路线，不能单独归因于初始化。

**注意数据版本与评分器版本是两件事。** 当前 HF 的 RL 数据卡写明 `OSS v5`、`RSTG + RL-ZVP`；这是该发布数据的来源说明。本次只取其中的题、图、标准答案及规则元数据，仍运行我们之前定好的纯 hybrid＋OSS v3，不启用 RSTG/RL-ZVP 新算法，也不因此自动改用 v5 评分器。该 HF 题单不能仅因同为 15k 就称作旧 OSS v3 run 的逐字节原题单。

### 0.1 历史模型来源

| 项目 | 标准目标 |
|---|---|
| MOPD 初始化模型 | `lzy-vlm-lab-opd/qwen3-vl-8b-iso-mopd-route-v2-step351` |
| 初始化 revision | `f9582f3594ba0a47e513a50b658204e7c322fbbb` |
| 原始训练运行名 | `hybrid-rstg-ossv3-mopd351-teacher200-pos05-full400-h200-20261112-100312-423389` |
| 训练数据 | 模型说明记录为 RSTG 的 **15,000 道训练题** |
| 评分器身份 | `rstg_oss_verifier_v3` |
| 对外评测模型 | `lzy-vlm-lab-opd/rlsd-rlrt-mopd`，各 step 的分支/文件布局须按实际仓库确认 |
| 本次核验样本 | 已导出的 step200：`export_source.json`、`README.md` |
| 原 run manifest SHA256 | `550f5e9b18b7bc59360da44f9fd8338b843752787dc12b0f66517ea09a071bed` |

Base 对照从 `Qwen/Qwen3-VL-8B-Instruct` 初始化；发布可复现实验时还必须锁定其实际下载 revision 或权重 SHA256，不能只写模型名。本次没有凭空填写一个 Base revision。

### 0.2 本交接包的证据边界

前版已读取历史 Base 共用框架中的权重公式、组掩码、actor 更新、教师提示、调度与配置，以及阿里实例上 MOPD step200 的原始导出来源；参考公式和分组的 32 项 CPU 测试通过。本版保留这些来源记录，未将其冒充为新的 H200 训练验证。

**本次新增核验了两个 HF 数据仓库的固定 revision、manifest、字段与首条样本，并保留原 32 项 CPU 检查记录；未重新取得旧 MOPD run 的完整 resolved 配置、原 OSS v3 评分器源码和原验证集。新 HF 题单与历史原题单的身份需分开。因此：**

- 下文给出可独立实现的完整算法规范，以及与我们共用 hybrid 配置对齐的参数配方。
- 不宣称已经逐字段验证了 MOPD 历史运行的全部工程参数，也不把目录名当成完整运行配置。
- `reference_code/` 是共用 hybrid 实现的核对用快照，包含未启用的其他实验代码；**不是原 MOPD 训练仓库的完整源码归档**。
- 文件夹是“方法、环境步骤、数据转换工具与核验材料交接包”，不是包含完整训练仓库、模型权重与原 OSS v3 的一键运行包。数据可按锁定 HF revision 下载；原 OSS v3 适配器及依赖、原验证集、完整 hybrid 仓库仍须补齐。环境新装步骤未在本机 Windows 上冒充 H200 实测。
- 旧 `README_HYBRID.md` 中 decay50/decay100、200 步等默认值属于旧实验；当前标准配方是总 400 步、teacher decay200、正 clip0.5、负 clip0.2。

## 1. 输入、输出与模型角色

每道训练题记为 `(I, x, a*)`：

- `I` 是一张或多张图，顺序固定；
- `x` 是题目文本以及需要保留的选项、任务类型等信息；
- `a*` 是独立的正确最终答案，不是教师生成的推理链；
- 学生是可训练策略 `πθ`；
- `πold` 是当前 rollout 池开始前的行为策略；
- 教师 `πφ` 是学生同架构、同初始化的独立滞后副本，每 10 个 RL global step 硬同步一次；
- 评分器 `V` 是 RSTG 的 OSS v3 验证逻辑，经训练接口返回正确性和可判分性。

训练输出是更新后的学生权重。教师只在训练中用于计算信用权重，评测、部署只加载导出的学生。

**教师是自教师，不是外部 32B 模型。** 教师拥有的额外信息是 `a*`。这既不同于“完全相同条件下的学生自评”，也不保证教师在错误前缀后一定能生成正确推理。

ISO＋MOPD 后接 RL 时，学生和自教师都从 MOPD step351 初始化。Base 后接 RL 时，两者都从 Base 初始化。不要出现学生用 MOPD、教师却仍用 Base 的混合初始化。

## 2. 学生采样：每道题生成 8 条回答

每个 RL global step 抽取 `B=256` 道题。每道题独立采样 `G=8` 条学生轨迹：

\[
y_{ij}\sim\pi_{\mathrm{old}}(\cdot\mid q_i),\quad i=1,\ldots,256,\ j=1,\ldots,8.
\]

`q_i` 表示经过任务模板、图像插入和模型 chat template 处理后的学生输入。学生输入中不放 `a*`。

每步共生成 `256×8=2048` 条轨迹。采样温度 1.0，top-p 1.0，top-k -1；训练 seed 为 1。**seed43 是我们评测使用的 seed，不是这里的训练 seed。**

学生提示最长 4096 token，回答最长 4096 token。遇 EOS 正常停止，`ignore_eos=false`。视觉 token 计入模型上下文；多图题不能只保留第一张图。

### 2.1 提示格式

共用 hybrid 参考实现的学生模板按任务类型要求：

```text
You FIRST think about the reasoning process as an internal monologue
enclosed within <thought> </thought> tags.
The final answer MUST be enclosed within <answer> </answer> tags.
```

选择题还要求 `<answer>` 中只放选项字母；数值题要求最终数值。完整模板见 `reference_code/examples/visual_rl/format_prompt/unified.jinja`，实施复现时应保留整份文件，不能只复制上面的摘要。

这里需要区分两件事：我们通过文字要求模型输出推理；历史共用配置没有设置 `enable_thinking=True` 的额外 chat-template 参数，也没有通过配置强制加 `<think>` prefill。这和 B15 的 Think 评测协议不是同一个概念。忠实复现应保留实际训练模板；若改成 `<think>`、增加 prefill 或换 system prompt，必须作为新变体记录。

## 3. 判分与题组处理

对每条学生回答调用同一份 RSTG＋OSS v3 训练评分接口，至少得到：

```python
{
    "overall": 0.0 or 1.0,
    "accuracy": 0.0 or 1.0,
    "judgeable": 0.0 or 1.0,
}
```

上面是 hybrid 所需接口合同，不能据此猜测 OSS v3 内部的具体数值容差、单位规则或字符串抽取方式。这些规则必须交付评分器原文件及依赖。尤其不能直接套用 MMFineReason 的 `numeric_tol=1e-3`，或拿 ReMI/MathVerse 的 benchmark 判分脚本来训练。

同一道题的 8 条回答通过题目 UID 归组，而不是假设“当前相邻的 8 行一定来自同一道题”；分布式重排后尤其需要保留 UID。

| 组别 | 定义 | 训练处理 |
|---|---|---|
| 混合组 | 8 条均可判分，正确条数为 1～7 | 产生正、负优势，进入 hybrid 更新 |
| 全对组 | 8 条均可判分且全对 | 组内优势为零，无本组策略梯度贡献 |
| 全错组 | 8 条均可判分且全错 | 组内优势为零，无本组策略梯度贡献 |
| 不可判分组 | 至少一条 `judgeable=0` | 整题 8 条回答的训练 loss mask 置零 |

**当前纯 hybrid 没有专门救活全错组。** 正确、错误 token 权重都只是乘在原优势上，`0×权重` 仍然是零。CAST、额外答案条件 rollout、负样本蒸馏、DCSD 等都不在本配方中。

“没有规范答案”“预测无法解析”“预测明确错误”并不总是同一类结果，应尊重原评分器的 verdict。有些格式失败在明确数值/选择题上可以可靠判错，有些开放文本不匹配只能是 unknown。不能一律掩码，也不能一律算错。

是否截断后直接判零、最终答案抽取位置、数值与符号等价、容差、单位以及多答案处理，都应由 **同一份 OSS v3 训练适配器** 决定。原评分文件未到位时，不补写一个猜测版本。

### 3.1 GRPO 组内优势

对可判分组，令 `r_ij∈{0,1}`：

\[
\mu_i=\frac1G\sum_j r_{ij},\qquad
s_i=\sqrt{\frac1{G-1}\sum_j(r_{ij}-\mu_i)^2},
\]

\[
A_{ij}=\frac{r_{ij}-\mu_i}{s_i+10^{-6}}.
\]

参考实现采用 PyTorch 默认样本标准差，即分母 `G−1`，不是总体标准差 `G`。将同一轨迹的 `A_ij` 广播到所有有效回答 token。prompt、padding 不参与 loss。

不可判分组整组置零。可判分的全对、全错组保留有效回答 mask，但优势为零：**它们在参考实现中仍计入 token 平均的分母**。如果改成“仅对混合组 token 求平均”，会改变梯度尺度，应作为新实现差异披露。

若完整全局 batch 没有任何非零优势，所有 rank 一起跳过 actor optimizer 更新，避免 Adam 动量或 weight decay 在无训练信号时仍改动参数。global step 仍推进，teacher 衰减仍计步。

## 4. 教师输入：题目＋独立答案；评分对象仍是学生原轨迹

教师得到与学生相同的图像、问题，以及正确最终答案 `a*`。共用模板完整结构为：

```text
{{ question }}

Here is a reference solution:
{{ hint }}

After understanding the reference solution, please try to solve this problem
using your own approach below. You FIRST think about the reasoning process
as an internal monologue enclosed within <thought> </thought> tags.
The final answer MUST be enclosed within <answer> </answer> tags.
```

这里 `hint` 实际只填独立最终答案。虽然模板叫 reference solution，**不能把数据集中完整解析或教师思维链塞入该字段**。`question` 在参考实现中优先使用已格式化的学生问题文本，因此不能随意改成遗漏选项/格式指令的另一段问题。

教师不另行采样新答案，不先生成一条“正确教师 CoT”，也不把教师自行输出当作 reward。对每条原学生轨迹 `y_ij`，教师做 teacher forcing：

\[
\ell^T_{ijt}=\log\pi_\phi(y_{ijt}\mid q_i^+,y_{ij,<t}),
\]

其中 `q_i^+` 是答案增强的教师输入。学生则计算：

\[
\ell^S_{ijt}=\log\pi_\theta(y_{ijt}\mid q_i,y_{ij,<t}).
\]

两侧评分的 **回答 token ID 和回答前缀完全相同**；不同的是 prompt 内容和可能滞后的模型参数。错误回答前缀也原样保留。教师看到正确答案并不等于教师概率就是可靠的过程真值，因此后面只让它有界调整 RL 信用。

每道题的教师 prompt 只构建一次，再按 UID 映射到 8 条回答。教师 prompt 更长，attention mask 和多模态位置编码要重新计算；可以复用图像像素张量，但不能直接照搬学生 position IDs。

教师 prompt 也做长度检查，不应静默截去题目、图像标记或答案。回答概率需要正确的 next-token shift；必须评分 `y_t` 而非误取 `y_(t+1)`。

## 5. hybrid 的核心：正 RLRT 风格、负 RLSD 分支

首先定义停止梯度的教师—学生 log 概率差：

\[
\delta_{ijt}=\operatorname{sg}(\ell^T_{ijt}-\ell^S_{ijt}).
\]

`sg` 表示 stop-gradient。参考代码的 `student_log_probs` 来自当前 actor forward，而不是直接拿缓存的 `old_log_probs` 代替；在只有一次全局更新且尚未 optimizer.step 的阶段，两者可能接近，但实现上要区分。

### 5.1 正优势轨迹：RLRT 风格

对 `A_ij>0`：

\[
u_{ijt}=e^{-\delta_{ijt}}
=\frac{\pi_\theta(y_{ijt}\mid q_i,y_{ij,<t})}
{\pi_\phi(y_{ijt}\mid q_i^+,y_{ij,<t})},
\]

\[
\bar u_{ijt}=\operatorname{clip}(u_{ijt},0.5,1.5).
\]

也就是正确轨迹中，教师相较学生给某 token 更低概率时，奖励乘数趋向上调；给更高概率时，乘数趋向下调。它不等于“教师越喜欢哪个 token，就越奖励哪个 token”。

### 5.2 负优势轨迹：RLSD 的失败分支

对 `A_ij<0`，仍然使用 `u=exp(-δ)`，但裁剪范围更窄：

\[
\bar u_{ijt}=\operatorname{clip}(u_{ijt},0.8,1.2).
\]

错误轨迹中，学生比答案条件教师更偏好的 token 会受到更强惩罚；教师相较更认可的 token 惩罚稍弱。

**因此当前 hybrid 的正负分支指数形式相同，差别是优势符号与裁剪半径。** “正用 RLRT、负用 RLSD”是组合来源的名称，不能想当然给两侧写成相反指数。

### 5.3 与原始 GRPO 优势插值

\[
f_{ijt}=(1-\lambda_k)+\lambda_k\bar u_{ijt},
\]

\[
\widetilde A_{ijt}=A_{ij}\operatorname{sg}(f_{ijt}).
\]

权重和教师不反向传播；只有学生策略损失更新学生。权重始终为正，因此不会把“失败”变成正奖励，也不会把“成功”翻转成惩罚。

当 `λ=0.5` 时：

| 分支 | 原始权重裁剪 | 最终优势乘数 |
|---|---|---|
| 正优势 | `[0.5,1.5]` | `[0.75,1.25]` |
| 负优势 | `[0.8,1.2]` | `[0.9,1.1]` |

例如一个失败 token 的 `A=-1`，插值乘数为 1.1，则其新优势为 -1.1；乘数为 0.9，则为 -0.9。正分支对应的正优势同理。

数值实现优先在 log 空间裁剪：`log(u)=-δ`，裁到 `log(1−ε)` 与 `log(1+ε)` 再取 exp，避免指数溢出。参考代码还把有限概率差饱和至 ±60；真正参与 PPO 的是裁剪后的有界权重。

### 5.4 四个方法开关的区别

| method | 正优势未裁剪权重 | 负优势未裁剪权重 |
|---|---|---|
| `grpo` | 1 | 1 |
| `rlsd` | `exp(+δ)` | `exp(-δ)` |
| `rlrt` | `exp(-δ)` | 1 |
| **`hybrid`** | **`exp(-δ)`** | **`exp(-δ)`** |

本交接只使用 `hybrid`。`rlrt` 在我们的代码中是统一答案上下文的控制版本；没有逐字复现 RLRT 原文的正确轨迹上下文流程，所以文中使用“RLRT 风格正分支”这一准确称谓。

## 6. PPO 更新保持原来的 current/old 概率比

教师权重只修改优势，不替代 PPO 的重要性比：

\[
\rho_{ijt}=\exp(\ell^S_{ijt}-\ell^{old}_{ijt}),
\quad
\bar\rho_{ijt}=\operatorname{clip}(\rho_{ijt},0.8,1.28).
\]

设：

\[
b_{ijt}=\max(-\widetilde A_{ijt}\rho_{ijt},
-\widetilde A_{ijt}\bar\rho_{ijt}).
\]

参考实现还保留 dual clip 系数 3：

\[
L_{ijt}=\begin{cases}
b_{ijt},&\widetilde A_{ijt}\ge0,\\
\min(b_{ijt},-3\widetilde A_{ijt}),&\widetilde A_{ijt}<0.
\end{cases}
\]

按有效回答 token 求全局平均：

\[
\mathcal L=\frac{\sum_{i,j,t}m_{ijt}L_{ijt}}
{\sum_{i,j,t}m_{ijt}}.
\]

这里 `m` 是回答 token mask 乘题组可判分 mask，排除 prompt、padding 和整组 unknown。可判分全对/全错组按第 3 节所述仍在分母中。

本配方不加额外 KL-to-base 惩罚，不加熵奖励，不加格式奖励或专门长度奖励，不加独立 full-vocabulary 蒸馏损失。日志中的 KL、entropy 可以是诊断指标，不代表参与了训练目标。

两个裁剪不能混淆：

- 教师信用权重：正 `[0.5,1.5]`、负 `[0.8,1.2]`；
- PPO current/old 比：两分支都用 `[0.8,1.28]`，并保留负优势 dual clip3。

## 7. 自教师刷新与 200 步衰减

每个 RL 阶段都重新从 `global_step=0` 开始计数。MOPD 的 step351 不计入这 400 步；MOPD351 后的 RL step200 表示额外完成了 200 个 RL global step。

\[
\lambda_k=0.5\max(1-k/200,0).
\]

| RL global step | λ |
|---:|---:|
| 0，初始化定义值 | 0.5 |
| 1，第一次实际更新 | 0.4975 |
| 50 | 0.375 |
| 100 | 0.25 |
| 150 | 0.125 |
| 199 | 0.0025 |
| 200～400 | 0 |

λ 衰减的是教师信用的混合比例，**不是学习率**。学习率一直保持 `1e-6`。

教师在 step0 与学生同初始化。只要 teacher 激活，每逢 global step10、20、30……的 actor 更新结束后，将学生当前权重硬复制给教师。step10 的评分使用复制前的滞后教师，step11 起才用 step10 刷新后的教师。刷新不是 EMA，不是再训练教师，也不是重新从磁盘载入 Base。

按上述衰减与实际判断顺序，step200 的 λ 已经为零，不再做教师 prompt、前向和周期同步；最后一次激活的周期同步通常是 step190。step200～400 的更新目标回到相同配置的 GRPO，保留相同数据与可判分组处理。

这不意味着 step200 以后的模型等于“从 Base 做 GRPO200”：它沿用了前 199 步 hybrid 学到的参数以及 Adam 状态。

## 8. 参数配方

以下是上一版已定好的共用 hybrid 配方，两条实验完全共用。第 8.2 节的训练题单按本次用户约定更新；其余参数不随数据量扩大而改变。历史 MOPD 的完整 resolved 配置仍应归档复核，下表不替代它。

### 8.1 方法与优化

| 参数/含义 | 值 |
|---|---|
| 方法 | `opsd.grpo_opsd_mode=hybrid` |
| simple hybrid 路由 | `enable_grpo_opsd_simple=true` |
| 二元题组合同 | `binary_reward_groups=true` |
| 独立滞后教师 | `freeze_teacher_model=true` |
| 教师上下文 | `teacher_context_mode=hint`，`use_gt_as_hint=true`，gold-only |
| 优势估计 | GRPO，二元 outcome，样本标准差，eps `1e-6` |
| 初始教师混合权重 | `rlsd_lambda=0.5` |
| 教师权重 warmup | 0 |
| 教师权重 decay | 200 global step |
| 正权重 clip 半径 | 0.5 |
| 负权重 clip 半径 | 0.2 |
| 教师同步间隔 | 10 global step，硬同步 |
| 仅对 thought span 加权 | false：全部有效回答 token |
| 仅对负轨迹加权 | false：正、负分支都启用 |
| 正负信用权重反向传播 | 关闭，stop-gradient |
| AdamW 学习率 | `1e-6`，constant |
| betas / weight decay | `(0.9,0.999)` / `0.01` |
| 学习率 warmup | 0 |
| max grad norm | 1.0 |
| PPO clip low / high | 0.2 / 0.28 |
| PPO dual clip | 3.0 |
| PPO epochs | 1 |
| loss 平均 | token 级全局平均 |
| KL 正则 / 熵奖励 | 关闭 |
| EAPO | `eapo_negative_kappa=0` |
| RISE / DCSD / SDPO | 均关闭 |
| 负样本 off-policy 蒸馏 | 关闭 |
| LoRA | 不使用；全参数更新 |
| 冻结视觉塔 | false |

字段名中保留 `opsd`、`stgca` 是工程历史：它们不代表当前启用了 OPSD-LoRA、RISE 或 DCSD。`stgca_*` 是 `rlsd_*` 的兼容别名；两套字段同时存在时必须同值，否则应报错。

### 8.2 数据、采样和视觉输入

| 参数 | 值 |
|---|---|
| 训练数据 | Base：两个 HF 15k 全量合并 30k；MOPD：仅 HF RL-RSTG-ZVP 15k，固定 revision 见第 9 节 |
| 训练评分器 | RSTG OSS v3，必须交付实际适配器与依赖 |
| 每步题数 | 256 |
| 每题回答数 | 8 |
| 每步轨迹数 | 2048 |
| `actor.global_batch_size` 配置值 | 256，worker 内部再乘 `rollout.n=8` |
| temperature / top-p / top-k | 1.0 / 1.0 / -1 |
| 训练 seed | 1 |
| prompt / response 上限 | 4096 / 4096 |
| `ignore_eos` | false |
| 图片最小/最大像素预算 | 262144 / 1048576，按实际 image_* 配置 |
| 模态 | 保留所有图片及原顺序 |
| 超长 student prompt | 过滤并记录数量 |
| 超长 teacher prompt | 明确检查，禁止默默截掉答案 |
| dataloader workers | 8 |
| 预处理过滤 workers | 64 |
| 在线混合题动态重采样 | 本配方不启用 |

`rollout.n=8` 才是这个分支实际使用的采样数；`opsd.student_rollout_n` 等其他分支字段不应被误当成此处的 G。

筛选时一题 8 次“1～7 对”不保证训练时新采样仍然是混合组。只离线筛一次并不意味着之后每个 batch 都有 256 道有效题。

### 8.3 8×H200 工程配方

| 参数 | 值 |
|---|---|
| 节点 / GPU | 1 / 8×H200 |
| 后端 | Ray＋FSDP＋vLLM，hybrid engine |
| FSDP | full shard |
| 混合精度 | BF16 计算，FP32 reduce/buffer；参数主副本以实际模型构建代码为准 |
| gradient checkpointing | 开启 |
| actor microbatch 配置 | 1 / GPU；动态 batching 开启时实际拼包由 token 预算决定 |
| actor/ref token budget | 8192 / GPU，共用 H200 配置值 |
| padding-free | 开启 |
| Ulysses size | 1 |
| torch compile | H200 共用配置为 true |
| CPU 参数/optimizer offload | H200 共用配置关闭 |
| vLLM tensor parallel size | 1，即 8 卡数据并行生成 |
| vLLM gpu_memory_utilization | 0.6 |
| vLLM max_num_batched_tokens | 32768 |
| vLLM chunked prefill | true |
| vLLM enforce_eager | false |
| 总 RL global step | 400 |
| checkpoint 间隔 | 50，全保留 |
| 验证间隔 | 50；启动前不验证 |

同一批 GPU 分阶段承担 rollout、学生更新和滞后教师评分；不要自行改成一半卡固定教师、一半卡固定生成的方案。rollout 前同步更新后的学生权重，生成结束后释放/休眠其 KV cache，为训练阶段让显存。

vLLM 的 0.6 不是“训练阶段只能用 60% 显存”，也不保证 GPU 利用率 60%。动态 microbatch 数不等于优化器更新次数；典型完整 batch 对应一次全局 optimizer.step。

阿里 PPU 版曾因后端支持采用 eager、不同 token budget、不同生成进程管理和内存策略。它与 H200 版可以共享数学方法，但不能称为执行配置逐项相同。跨设备只恢复 HF 学生权重属于新训练初始化；完整 optimizer/teacher/dataloader 恢复需要匹配原 world size 与代码合同。

### 8.4 步数和数据曝光

代码在显式设置 `trainer.max_steps=400` 时，以该值作为训练步数；此时 `total_epochs` 不负责截断训练，dataloader 用完后继续循环。不要误读成只跑 3 轮就一定结束，也不要同时把“400 步”和“数据只遍历一次”写在复现说明中。

总步数仍为 400，不因 Base 题库扩大到 30k 而自动翻倍。不计过滤和 drop_last 等影响：

| 项目 | Base 混合 30k | MOPD 后续 15k |
|---|---:|---:|
| 每步抽取问题 / 生成轨迹 | 256 / 2048 | 256 / 2048 |
| step200 累计问题曝光 | 51,200 | 51,200 |
| step200 平均记录曝光 | 约 1.71 次 | 约 3.41 次 |
| step400 累计问题曝光 / 轨迹 | 102,400 / 819,200 | 102,400 / 819,200 |
| step400 平均记录曝光 | 约 3.41 次 | 约 6.83 次 |

按记录均匀抽样，Base 的两仓库在 400 步中各约 51,200 次问题曝光，实际受每轮打乱和最后不完整 batch 影响。这里是记录曝光，并非独立题曝光，也不等于非零优势的有效训练信号数。超长 prompt 过滤后，记录数和来源比例可能改变，必须保存过滤前后计数，不应自行补采样改变旧配置。

## 9. 数据与源码交接：达到可复现必须一起给出的资产

只给算法名称和模型权重，不能让别人严格复现我们的历史实验。交接时至少包括：

1. **初始化模型**：MOPD351 指定 revision，或者 Base 指定 revision；包含权重、config、tokenizer、processor、chat template。
2. **完整 train/val 题单**：训练题使用下方两个固定 HF 版本；保留来源、原始 sample_id、行号、独立最终答案、选项和规则元数据，并保存转换后 JSONL 的 SHA256。验证集沿用原实验的独立 val，不从这两个 train 中暗中切走样本，也不把 train 兼作 val。两个 HF 仓库只发布 train，原 val 需另外交付。
3. **全部图片**：与 JSONL 引用一一对应，包含多图顺序；交付路径映射和图片清单。不要只发 ID 列表或软链接到对方看不到的共享盘。
4. **评分器**：原 RSTG OSS v3 训练适配器、调用到的 verifier 模块、配置/容差、运行依赖及文件 SHA256。评分器名字不等于评分器源码。
5. **实际训练源码**：包括 canonical data loader、teacher prompt 构造、GRPO 计算、credit helper、actor、FSDP-vLLM 同步、checkpoint 逻辑及启动器。
6. **运行配置和环境**：`resolved-*.yaml`、`run_manifest.json`、运行时 `experiment_config.json`（如有）、依赖版本、CUDA/driver、GPU 数量、环境变量覆盖值。
7. **验收记录**：第一步分组计数、有效 lambda、teacher refresh、梯度、checkpoint 完整标记与一次真实恢复验证。

JSONL 接口示例（只展示格式，不是实际 RSTG 样本）：

```json
{"problem_id":"example-001","problem":"<image>\n题目文本","answer":"B","images":["images/example.jpg"],"problem_type":"multiple choice","options":["选项1","选项2","选项3","选项4"]}
```

必须同时保留评分器所需的真实字段，不能以示例中的最少字段覆盖原数据。`problem_id` 应唯一，train/val 不得重叠。图片数应匹配 `<image>` 标记数。相对图片路径应绑定明确 image root，绝对路径迁移时必须重定位。

### 9.1 固定的训练数据版本

| 数据 | HF revision | manifest 条数 | MMFineReason | Cauldron |
|---|---|---:|---:|---:|
| [MOPD-route-v2-train-15k](https://huggingface.co/datasets/lzy-vlm-lab-opd/MOPD-route-v2-train-15k) | `6374961a4945f5ee7462520ecd54bcc93ee9d0bc` | 15,000 | 5,486 | 9,514 |
| [RL-RSTG-ZVP-train-15k](https://huggingface.co/datasets/lzy-vlm-lab-opd/RL-RSTG-ZVP-train-15k) | `df005b52aead28acf57b05a1f30487a220d398bb` | 15,000 | 6,000 | 9,000 |
| Base 全量合并 | 两者各一次 | **30,000** | **11,486（38.29%）** | **18,514（61.71%）** |
| MOPD 后续 | 第二份一次 | **15,000** | **6,000（40%）** | **9,000（60%）** |

两者均为 20 个 Parquet 分片，每片 750 条，内嵌原图片字节；发布分片的 SHA256 已列于各自 `manifest.json`。源类别数来自锁定 manifest，本包保留其原文。未下载全部图片做新的全量交集审计，故不声称两份数据互不重合。MOPD 版不再加入第一份蒸馏训练题，也不另加旧 Base 筛选的 32,676 题。

### 9.2 HF 字段转换与训练合同

| HF 字段 | hybrid 训练用途 |
|---|---|
| `question` | 映射到 `problem`，保留 `<image>` 和完整选择题选项 |
| `ground_truth` | 映射到独立 `answer`；teacher hint 只读此答案 |
| `image` / `image_sha256` | 导出原始字节到图片文件；JSONL 写 `images` 列表并核验图片哈希 |
| `sample_id` | 原样保留为来源 ID；另用仓库＋原始行号生成唯一 `problem_id`，避免拼接后 UID 相撞 |
| `rule_solution_json` | 原字符串保留，并解析为 `rule_solution`；选择题 options、references、容差及 metadata 均保留，不能丢给默认文本比较 |
| `dataset/source/subset/domain/subtask/answer_type/reward_type` | 保留用于路由、判分与分来源统计 |
| `prompt/enable_thinking/min_pixels/max_pixels/teacher_id` | 作为 `published_*` 来源记录，不用来覆盖本次固定训练设置 |

**不要直接把 HF `prompt` 整段训练对话喂给旧模板。** 发布数据带有 `<think>` system 指令，而当前 hybrid 沿用 `unified.jinja` 的 `<thought>` 指令；直接拼接会变成双重模板。使用原始 `question` 加一次旧学生模板，图像预算仍为 262144～1048576。数据中的三教师路由标签不是本次教师选择器，本次只有同初始化的滞后自教师。

附 `prepare_hf_data.py`：按固定 revision 下载、逐分片校验、导出图片和上述两份训练 JSONL；不会启动训练，不自行重判、不切 val、不去重、不按 teacher_id 训练外部教师。唯一 ID 只解决工程分组冲突，不意味着重复内容变成独立题。

**评分接口接线不可省略：** 原 OSS v3 如何读取结构化规则，要以原适配器为准。必须确认 reward manager 确实把 `rule_solution_json`、options、references/容差等送到它；光是 JSONL 中保留字段还不够。包内旧共用 reward manager 快照不能代表原 OSS v3 适配已经完成。不要把规则 JSON 整串当作答案去比较，也不要把它塞进 teacher hint。两份数据都用同一旧 OSS v3，分别做正确/错误/不可判分的回放检查；遇到 v3 不支持的任务要明确记录，不能静默切换 v5 或 MMFineReason 判分。

原 val 大小、筛选细节、OSS v3 的具体容差和异常处理没有原文件就不编造。当前两份新数据的发布记录不能代替历史评分器源码。

交接包中的 `collect_original_run.py` 可在启智原实例上运行，按准确 run 名归档配置、代码、评分器同目录依赖、题单和图片路径清单。它不启动训练、不修改原 run，也不复制几十 GB 的 checkpoint。图片本体和模型仍应单独交付，脚本生成的清单会明确标记这些依赖。

## 10. 从收到资产到启动训练

以下流程使用**交付的原 MOPD hybrid 仓库**，不是随便安装最新版 veRL，也不是直接运行本包的 `reference_code/`。

### 10.1 环境和资产准备

详见附带的 **`环境与数据准备.md`**，其中给出 Linux＋8×H200、Python3.11、CUDA12.8 软件栈、依赖安装顺序、FlashAttention ABI 检查、HF 数据下载转换、模型版本记录、原评分器接线和启动前验收。

核心版本保持 Python3.11、PyTorch2.8.0+cu128、torchvision0.23.0、Transformers4.57.6、vLLM0.11.0、FlashAttention2.8.3、Ray2.53.0、OmegaConf2.3.1。`environment/requirements-cuda.txt` 固定直接依赖；`environment/reference-pip-freeze.txt` 是旧环境的取证记录，含历史本地 wheel 路径，不能原样在新机器执行安装。

优先复用原 CUDA 镜像/环境；若重建，按附文安装并执行 `pip check`、`check_environment.py` 与实际 8 卡启动检查。参考快照不含完整可安装仓库或原 OSS v3。Windows 本机的数据/文档检查不能等同于 H200 完整训练通过。

### 10.2 两份配置如何生成

以同一份原 hybrid resolved 配置为母版，复制两份。下面是需要改动的字段示意，不是丢掉其余参数的最小完整 YAML：

```yaml
data:
  train_files: /workspace/hybrid/data/base_mix30k.train.jsonl  # Base
  # MOPD 版改为 /workspace/hybrid/data/mopd_rstg15k.train.jsonl
  val_files: /workspace/hybrid/original_val/val.jsonl
  image_dir: /workspace/hybrid/data
  prompt_key: problem
  answer_key: answer
  image_key: images
  format_prompt: /workspace/hybrid/repo/examples/visual_rl/format_prompt/unified.jinja
  shuffle: true
  seed: 1
worker:
  actor:
    model:
      model_path: /workspace/hybrid/models/Qwen3-VL-8B-Instruct  # Base
      tokenizer_path: /workspace/hybrid/models/Qwen3-VL-8B-Instruct
      # MOPD 版的这两项改为 /workspace/hybrid/models/MOPD-route-v2-step351
  reward:
    reward_function: /实际原OSS_v3适配器路径.py:实际原函数名
    # reward_function_kwargs 完整沿用原值，不用发布数据里的 OSS v5 覆盖。
trainer:
  max_steps: 400
  total_epochs: 3
  n_gpus_per_node: 8
  experiment_name: base-mix30k-hybrid-h200  # MOPD 版用独立名称
  save_checkpoint_path: /workspace/hybrid/runs/base-mix30k-hybrid-h200
  load_checkpoint_path: null
  find_last_checkpoint: false
  save_freq: 50
  save_limit: -1
```

原 val 图片若位于另一 image root，应先按清单重定位成绝对路径或统一图片根，不能只改 train 图片路径。验证集的题目 ID 及内容还要与两份 train 检查交集。

第 8 节全部字段从母版保留，特别是 batch256、n8、lr1e-6、teacher sync10、decay200、lambda0.5、clip0.5/0.2、PPO0.2/0.28、dual3、4096 response、seed1 和 50 步保存。`opsd_main` 的 simple hybrid 路由需启用，不能误走纯 GRPO 或全分布蒸馏。`configs/experiment_recipe.json` 提供可机读的两套数据身份与核心参数，不冒充完整 runtime YAML。

对两份最终 resolved 配置做差异检查：允许初始化模型/分词器、train_files、run 名和输出/缓存绝对路径不同；其余算法、优化、采样、图像预算、评分器文件哈希及 kwargs、提示模板和验证设置保持一致。数据发布附带的 system prompt、teacher_id、像素设置不能偷偷形成第二套配置。禁止沿用旧 load_checkpoint_path。原始 2026-11-21 包和历史实验不改写。

### 10.3 前台与后台启动

在已经安装好原训练仓库及对应环境、路径重定位完成后，底层训练入口是：

```bash
cd /接收方/原训练仓库
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_TMPDIR=/tmp/hybrid-rstg
export TMPDIR=/tmp/hybrid-tmp
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# PYTHON 必须是已经安装对应 CUDA 软件栈的 Python 绝对路径。
# CONFIG 必须是已补齐路径、与原评分器绑定的新 run YAML。
"$PYTHON" -u -m verl.trainer.opsd_main "config=$CONFIG"
```

需要后台运行时，将最后一条换为：

```bash
mkdir -p "$LOG_DIR"
nohup "$PYTHON" -u -m verl.trainer.opsd_main "config=$CONFIG" \
  >"$LOG_DIR/$RUN_NAME.log" 2>&1 </dev/null &
echo "PID=$!"
echo "LOG=$LOG_DIR/$RUN_NAME.log"
```

`PYTHON`、`CONFIG`、`LOG_DIR`、`RUN_NAME` 需预先设置；Ray临时目录使用短路径，避免 AF_UNIX socket 路径超过 107 字节。一次只启动一个占用该组 GPU 的训练任务。

若在平台“分布式训练任务”中运行，主命令应保持前台；后台 launcher 自己退出可能让平台认为任务完成，从而杀掉后台训练进程。nohup 主要针对交互式实例。

底层入口是否自动写原启动器的 manifest 取决于交付代码。因此正式实验优先使用交付的原 launcher 完成数据预检、manifest 和并发锁；不能因为底层模块能启动就跳过这些身份记录。

### 10.4 第一阶段和后续训练的边界

本说明只复现 RL 阶段。ISO＋MOPD 起点可直接下载指定 revision，无需先重新训练其第一阶段。若要复现整条流水线，则还需另交付 ISO＋MOPD 的教师、路由、数据和训练配置，本说明不代替那部分。

## 11. 一步更新的精确顺序

```text
初始化 student θ 与独立 teacher φ；φ ← θ。
设 RL global_step=0，创建 AdamW、dataloader、随机数状态。

while global_step < 400:
    global_step += 1
    将最新 student θ 同步到 rollout 引擎，记为 πold
    取 256 道题；每题生成 8 条回答，不给学生看 gold
    评分器返回二元正确性和 judgeable
    按题目 UID 计算 GRPO 优势（样本标准差）
    任一 unknown 的题整组 mask=0
    λ = 0.5 * max(1 - global_step/200, 0)

    如果全局存在非零优势:
        如果 λ>0:
            每题构造一次“同图题目＋独立答案”的 teacher prompt
            对学生原轨迹 teacher forcing，计算 teacher logp
        计算 student 当前 logp
        如果 λ>0:
            对正/负优势分别用 exp(student_logp - teacher_logp)
            正 clip[.5,1.5]；负 clip[.8,1.2]；与1按λ插值
            weighted_A = A * stopgrad(weight)
        否则:
            weighted_A = A；不访问teacher logp
        用原 student_current/student_old 比计算 PPO dual-clip loss
        全局按有效回答token归一化，反传、grad_clip1.0、AdamW step
    否则:
        所有rank一起跳过optimizer

    如果λ>0且global_step%10==0:
        teacher φ ← 更新后的 student θ
    按50步频率验证、保存完整checkpoint
```

计算 `teacher_logp` 和 `student_logp` 的实际微批前向可以交错执行，但都必须在对应 optimizer.step 之前完成；每个微批获得相同 pre-update 阶段的含义。这里不缓存来自已更新学生的第二套信用信号。

## 12. 训练监控和验收

不要只看 loss 是否下降。至少记录：

| 项目 | 解释/预期 |
|---|---|
| `groups/total_count` | 完整主 batch 应为256 |
| `groups/mixed_count` | 有非零正确/错误对比的题目数 |
| `groups/all_correct_count`、`all_wrong_count` | 都不产生本组优势梯度 |
| `groups/unjudgeable_count` | 被整组屏蔽的题数 |
| `groups/effective_trajectory_count` | 混合题数×8，不是全部采样条数 |
| `grpo_opsd/stgca_effective_lambda` | step1=.4975，step100=.25，step200=0 |
| 正负最终乘数均值/裁剪比例 | 正、负分支的信用权重行为 |
| `actor/update_skipped_zero_advantage` | 全局无有效优势时为1 |
| `grpo_opsd/teacher_synced` | 激活期逢10步同步 |
| `grpo_opsd/teacher_forward_skipped` | λ归零后应跳过 |
| response长度/截断比例 | 识别回答变长、不收束、格式失败 |
| reward accuracy / judgeable | 必须一起看，不能把有mask的统计混为有效题正确率 |
| grad norm、非有限数检查 | 确认真的完成了更新且无 NaN/Inf |
| rollout / teacher / update / save 用时 | 区分生成、教师评分和更新瓶颈 |

四类组的计数必须加起来等于 total_count。若一个 batch 只有100道混合题，其直接有效轨迹数为800，并不能说2048条全部提供了组内优势信号；同时注意同质组保留分母的梯度缩放影响。

验收应覆盖：真实一轮 rollout→评分→非零优势→反传、step10教师刷新、一次完整保存与重启恢复。CPU公式测试不能替代8卡长序列/NCCL/峰值显存验收。

## 13. Checkpoint、恢复和导出

完整 checkpoint 应包含学生、AdamW、scheduler、滞后教师、dataloader、trainer/RNG、各 rank extra state；只有 `checkpoint_complete.json` 等提交标记存在且分片完整，才可用作精确续训点。

恢复时保留原代码、数据、评分器、world size、提示与调度。不要把滞后教师重新覆盖成当前学生；不要把从 HF 模型重新建优化器叫作完整断点恢复。HF 导出用于评测或新阶段初始化，通常不含 optimizer/dataloader 状态。

导出学生为标准 HF safetensors，并复制 tokenizer、processor、chat template、generation_config；执行严格模型加载、权重有限性检查与一次真实多模态输入前向/生成检查。不要把分片 FSDP checkpoint 直接当 HF 模型目录。

## 14. 与历史 Base 版的准确关系

| 维度 | 历史 Base 版 | 本版 Base | 本版 MOPD 后续 |
|---|---|---|---|
| 初始化 | Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Instruct | ISO＋MOPD route-v2 step351 |
| 数学更新 | hybrid 正 RLRT / 负 RLSD | 共用 hybrid | 同左 |
| 数据 | 曾用 MMFineReason 筛选题 | 两个 HF 15k 全量拼接 30k | 仅 HF RL-RSTG-ZVP 15k |
| 评分器 | 曾用 MMFineReason 规则适配器 | 原 OSS v3 | 同一原 OSS v3 |
| 硬件与后端 | 曾有 H200、PPU 变体 | 8×H200，FSDP＋vLLM | 同左 |
| 阶段计步 | RL 从 0 起 | RL 从 0 起，总 400 | MOPD351 后新 RL 从 0 起，总 400 |
| 外部强教师 | 无 | 无，自身滞后副本 | 无，自身滞后副本 |

这次明确比较“Base＋全量 30k RL”和“先 MOPD＋后续 15k RL”两条路线。它们的 RL 算法与超参数相同，但数据与初始化不同；不能把分数差完全归到初始化上。若另做仅初始化消融，可额外固定同一题单，两者都用 15k 或都用 30k；这不是本次指定配置。

## 15. 复现说明中的常见错误

- 把每步2048条轨迹写成2048道题，或把microbatch次数写成optimizer更新次数。
- 正分支误用纯RLSD的 `exp(+δ)`，导致跑成另一算法。
- 教师权重分母错用old策略，未披露改变了参考实现。
- 教师权重未detach，反向传播了信用权重本身。
- 教师看到完整正确CoT，而实际answer-only配方只提供最终答案。
- 用教师新生成文本替换学生轨迹，变成了另一个蒸馏流程。
- 同步顺序放错，使第10步提前使用更新后教师给同一步轨迹评分。
- 把200步teacher衰减误写成学习率衰减。
- λ=0后仍做纯hybrid不需要的teacher forward，浪费计算。
- 全错组加了辅助蒸馏/CAST，却继续称为当前纯hybrid。
- 偷换unknown处理、混合组分母、评分容差、格式奖励或长度奖励。
- 把Base seed43评测配置误当成seed1训练配置。
- 将本版指定的新 HF 30k/15k 方案说成已核验的历史原题单；或把两份 15k 各抽 7.5k、悄悄去重，违反本次全量拼接约定。
- 直接采用 HF 数据里的 system prompt、teacher_id 或 OSS v5 名称，改变原提示、教师机制或训练评分器。
- 转换数据时丢掉 rule_solution_json，或者未把规则元数据传入真实 OSS v3 就开始训练。
- 仅靠run名确认配置，未交付resolved配置、评分器与题单哈希。

## 16. 可直接转述的方法摘要

我们在Qwen3-VL-8B上采用答案条件自教师指导的GRPO。每步对256道多模态题分别采样8条学生回答，由固定的RSTG OSS v3训练评分器给出二元正确性及可判分标记；按题计算标准化组内优势，屏蔽包含不可判分回答的整组。教师是学生的独立滞后副本，额外读取独立最终答案，在同一学生回答前缀上计算已采样token的概率。对正优势采用RLRT风格权重、对负优势采用RLSD失败分支，两者均使用停止梯度的学生/教师概率比，分别裁剪至[0.5,1.5]和[0.8,1.2]，与原优势按λ插值。λ从0.5在200个RL global step内线性降至0；激活期教师每10步硬同步。学生以lr1e-6、PPO clip0.2/0.28和dual clip3进行全参数更新，总400步，不叠加KL蒸馏损失、EAPO、RISE、DCSD或全错组辅助目标。本版 Base 初始化时使用两个固定版本的 15k 训练集全量拼接、均匀打乱，共 30,000 条记录；ISO＋MOPD step351 初始化时仅用 RL-RSTG-ZVP 的 15,000 条记录。两者其余 RL 参数及 OSS v3 评分器一致，属于数据与初始化同时指定的两条训练路线。

