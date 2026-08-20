# gpt-toolcall/data

本目录存放训练数据 + 模型权重，按设计**不入库**（见仓库根 `.gitignore`）。

```
data/
├── raw/
│   └── hermes-train.jsonl       # Hermes-function-calling-v1 原始 (~26 MB, 不入库)
├── toolcall-data.json           # Alpaca 格式训练数据 (训练生成, 不入库)
├── toolcall-sft-medium.pth      # SFT 后 medium 模型 (~1.7 GB, 不入库)
└── smoke/
    └── toolcall-smoke.json      # 12 条手写 smoke 数据 (已入库, 用于 build 验证)
```

## 文件说明

### `raw/hermes-train.jsonl`
- 来源：[NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- 格式：每行 JSON，`{"conversations": [{"from": "system|human|gpt|tool", "value": "..."}, ...]}`
- 大小：约 26 MB / 12,565 条多轮对话
- **不入库**；用 `./scripts/run.sh download` 重新拉取

### `toolcall-data.json`
- 来源：`scripts/convert_data.py` 把 Hermes JSONL 拍平为 Alpaca
- 格式：每行 JSON `{"instruction": "<system + tools + user + 历史>", "input": "", "output": "<assistant 该写的内容>"}`
- 大小：取决于 max-len 过滤后保留多少（实测 ~30 MB / ~25k 条样本）
- **不入库**；用 `./scripts/run.sh convert` 重新生成

### `toolcall-sft-medium.pth`
- 来源：`scripts/train-background.sh` 训练生成
- 大小：约 1.7 GB（gpt2-medium fp32 权重）
- **不入库**；用 `./scripts/run.sh download-weights && ./scripts/run.sh train` 重新训练

### `smoke/toolcall-smoke.json`（已入库）
- 12 条手写对话，覆盖 3 个工具（`get_weather` / `get_time` / `calc`）+ 单轮 / 多轮 / 自然语言回答 / 中文指令
- 用于 `scripts/run.sh smoke` 快速验证 chat 二进制能跑通（不需要训练权重）

## 数据格式示例

### Hermes 原始（输入）
```json
{
  "conversations": [
    {"from": "system", "value": "<tools>\n<tool>{\"name\":\"get_weather\",\"description\":\"天气\",\"parameters\":{...}}</tool>\n</tools>\nYou are helpful."},
    {"from": "human",  "value": "北京天气?"},
    {"from": "gpt",    "value": "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"北京\"}}\n</tool_call>"},
    {"from": "tool",   "value": "{\"temperature\":28}"},
    {"from": "gpt",    "value": "北京今天 28°C。"}
  ]
}
```

### Alpaca 拍平后（输出）
```json
{"instruction": "You are a helpful assistant with access to tools...\n<tools>\n<tool>{\"name\":\"get_weather\",\"description\":\"天气\",\"parameters\":{...}}</tool>\n</tools>\nUSER: 北京天气?\nASSISTANT:", "input": "", "output": "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"北京\"}}\n</tool_call>"}
{"instruction": "...上一条 + ASSISTANT: <tool_call>... + TOOL: {...} + ASSISTANT:", "input": "", "output": "北京今天 28°C。"}
```

每个 assistant turn 展开为一条训练样本（含工具描述的 system prompt 也重复贴进去，因为 SFT 不会跨样本记忆）。

## 重新生成数据

```bash
cd gpt-toolcall
./scripts/run.sh download          # 拉 Hermes 原始
./scripts/run.sh convert           # 生成 toolcall-data.json

# 可选：限制大小（避免超出 gpt2-medium 1024 上下文）
python3 scripts/convert_data.py \
    --max-len 2048 \
    --shuffle \
    --limit 10000
```

## 不入库的考虑

- **隐私 / 合规**：HuggingFace 上 Hermes 数据是 CC-BY-NC-4.0（NC = 非商业），不适合进仓库 license
- **体积**：原始 26 MB + Alpaca ~30 MB + 模型 1.7 GB，加起来 2 GB 进 git 仓库是个灾难
- **可重现**：`download-data.sh + convert_data.py` 是确定性的（除非上游变更）

需要还原时直接 `./scripts/run.sh download && convert`，两份脚本都是幂等的（已存在就跳过）。