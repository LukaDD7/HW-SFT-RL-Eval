---
pretty_name: RL (RSTG + RL-ZVP) training questions (15k)
task_categories:
- visual-question-answering
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---
# RL (RSTG + RL-ZVP) · 训练题目

RL 阶段。从 route-v2 MOPD step351 初始化，训练方法为 RSTG + RL-ZVP，训练数据版本为 OSS v5。

本仓库收录该阶段实际使用的 15,000 条训练题，保留原始行序。每题包含原图、原始 system/user 消息、含选项的题面（如有）、标准答案、判分规则、题目来源和教师路由标签。图片以原始文件字节嵌入 Parquet。仓库中的答案来自训练数据标注。

| 来源 | 训练题数 |
| --- | ---: |
| MMFineReason | 6,000 |
| the_cauldron | 9,000 |

`question` 是原始 user 消息，`prompt` 包含完整训练对话输入。题目可以通过 `sample_id` 追踪，图像可以通过 `image_sha256` 核验。`ground_truth` 是训练记录中的答案字符串，`rule_solution_json` 保留该题的结构化判分说明。`teacher_id` 表示数据记录中的路由标签。原始训练 Parquet 的 SHA-256 和发布分片的校验值见 [manifest.json](manifest.json)。

另一个阶段的题目见 [MOPD-route-v2-train-15k](https://huggingface.co/datasets/lzy-vlm-lab-opd/MOPD-route-v2-train-15k)。

## 原始数据与授权

MMFineReason 来自 OpenDataArena 发布的 MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking。其数据卡标注 Apache-2.0。the_cauldron 中各子数据集适用各自条款，其整理者对所拥有权利的提示词采用 CC BY 4.0。使用时请按 `source` 核对相应条款。[MMFineReason](https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking) · [the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
