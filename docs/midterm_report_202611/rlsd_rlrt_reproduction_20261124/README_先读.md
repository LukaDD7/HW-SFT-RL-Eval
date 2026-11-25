# RLSD＋RLRT 复现材料（2026-11-24）

这版只按新约定调整训练题单。先读 `RLSD_RLRT_方法与复现说明.md`；环境、下载及数据转换步骤见 `环境与数据准备.md`。

| 实验 | 初始化 | 数据 |
|---|---|---|
| Base 后接 hybrid | Qwen3-VL-8B-Instruct | 两个 HF 15k 全量拼接，30k |
| MOPD 后接 hybrid | route-v2 MOPD step351 | 仅 RL-RSTG-ZVP 15k |

共同配置：8×H200、400 RL steps、teacher decay200、teacher sync10、lambda0.5、batch256×8、lr1e-6、正/负 clip0.5/0.2、4096 回答上限、seed1、每50步完整保存、原 OSS v3 评分器；不启用 LoRA/EAPO/RISE/DCSD/RL-ZVP。

Base 全量保留各15k，不各抽7.5k，不去重；在原 dataloader 按 seed1 均匀打乱。数据量与初始化不同，不是仅初始化消融。两个 HF 数据卡的旧 system prompt 和路由标签只作来源记录，不能覆盖 hybrid 的固定提示与自教师。

## 文件

- `RLSD_RLRT_方法与复现说明.md`：算法公式、精确更新顺序、全部核心参数、分组和断点合同。
- `环境与数据准备.md`：Linux/CUDA 安装顺序、数据字段映射、模型与评分器准备、启动和检查。
- `configs/experiment_recipe.json`：两套固定 HF revision、全量合并策略与不变参数；不是完整 runtime YAML。
- `prepare_hf_data.py`：固定版本下载、校验分片和图片、导出 Base 30k/MOPD 15k JSONL；不会启动训练。
- `check_environment.py`、`environment/`：环境检查与重建直接依赖；原 freeze 单独保留，非可直接移植的完整锁文件。
- `reference_code/`：原共用 hybrid 核心源码快照（未修改）；不是完整训练仓库。
- `evidence/`：旧实验来源记录、原32项CPU检查、新HF manifests/字段、更新审计。`delivery_audit_20261121.json` 是旧版交付状态；本版以 `update_audit_20261124.json` 为准。
- `collect_original_run.py`：从启智原目录只读收集旧 run 的代码、题单与评分器，不抓模型/图片本体。

## 还需随项目交付的原资产

完整 hybrid 训练仓库、OSS v3 适配器及实际依赖/kwargs、原独立验证集和图片。公开 HF 题目仓库不包含这些代码；有数据、能 import 模型，并不表示训练评分器接线已完成。尤其要确认两份数据的结构化规则传入同一个旧 OSS v3，而非套一个默认字符串评分器。

旧文档、21日ZIP及历史运行均保留不改。这次没有启动GPU训练、上传云端或提交Git；没有把新HF题单冒充旧run逐字节原文件，也没有声称在Windows本机重建了H200 CUDA环境。
