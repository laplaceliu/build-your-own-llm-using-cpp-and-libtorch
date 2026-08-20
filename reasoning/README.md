# reasoning/ — 推理模型项目

基于本仓库 `gpt-sft` 的 GPT-2 实现，应用《附录 F 理解推理大语言模型：构建与优化推理模型的方法和策略》全部四类方法构建真实可运行的推理模型，并通过对比实验观测方法与规模的效果差异。

## 三大方法

| 方法 | 附录 F 对应 | 路径 | 形态 |
|------|-------------|------|------|
| 推理时间扩展 | 小节 1（CoT/投票/束搜索） | `serve/reasoning_chat` `--strategy` | 纯推理层，无需训练 |
| 纯 SFT 与蒸馏 | 小节 4（Sky-T1/旅程学习） | `train/sft_train` | 微调 + 推理数据 |
| 纯强化学习 (GRPO) | 小节 2（DeepSeek-R1-Zero/TinyZero） | `train/rl_train` | 规则奖励 + 策略梯度 |
| SFT+RL 流水线 | 小节 3（DeepSeek-R1 蓝图） | `rl_train --init sft` | 在 SFT 基础上继续 RL |

## 双规模对比

`small` (124M) 与 `medium` (355M) 全链路产出：`{size}-{base|sft|rl|sft_rl}.pth`，由 `eval/eval_math` 统一评测。

## 目录结构

```
reasoning/
├── CMakeLists.txt            # 顶层构建
├── core/                     # 核心库（模型、训练、奖励、生成工具）
├── train/                    # sft_train / rl_train CLI
├── serve/                    # reasoning_chat / reasoning_serve
├── eval/                     # eval_math 评测器
├── scripts/                  # download-data / build / run / run-compare
├── data/                     # 下载数据与 .pth 产物（gitignore）
└── docs/compare-report.md    # 对比实验结果
```

## 依赖

- C++17 + GCC 11（与 gpt-sft 一致）
- LibTorch（默认 `/opt/libtorch`，可用 `TORCH_ROOT` 覆盖）
- CUDA 13.0 + 至少一张 GPU（GPT-2 medium 训练推荐 ≥12GB）
- Python 3 + `huggingface_hub` / `pandas` / `pyarrow`（仅数据下载阶段）

## 快速开始

```bash
# 1. 下载数据（SFT 用 Sky-T1-17k，RL/评测用 GSM8K）
./scripts/download-data.sh

# 2. 编译
./scripts/build.sh

# 3. 训练 SFT（small）
./scripts/run.sh sft_train --data data/sft-train.json --size small --epochs 2 --out data/small-sft.pth

# 4. 训练 RL（small，从基座）
./scripts/run.sh rl_train --data data/rl-train.json --weights data/gpt2-small.safetensors \
  --size small --max-steps 60 --out data/small-rl.pth

# 5. 推理（CoT + 多数投票）
./scripts/run.sh reasoning_chat --model data/small-sft.pth --size small --strategy vote --samples 5

# 6. 评测对比
./scripts/run-compare.sh
```

详细对比报告见 [`docs/compare-report.md`](docs/compare-report.md)。
