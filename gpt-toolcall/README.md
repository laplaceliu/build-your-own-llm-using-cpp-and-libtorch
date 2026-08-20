# gpt-toolcall

自然语言 → Function Calling / Tool Use 的最小 SFT 项目。

> 在 gpt-bash（自然语言 → bash） 之上，加一个 Hermes XML 风格的工具调用能力。

## 背景

Function calling 是 LLM 当下的核心能力之一。但主流实现（OpenAI Function Calling、Anthropic Tool Use）都是闭源 API。我们用同一套 GPT-2 backbone（与 `gpt-bash` 共享）训一个能生成 `<tool_call>{...}</tool_call>` 的小模型，跑通整个 SFT 闭环。

**关键认知**：模型本身不执行函数。Function calling 本质是"在特定 trigger 下输出结构化文本 + 客户端 dispatcher 解析执行 + 回喂"的循环。

## 设计取舍

| 决策 | 选择 | 理由 |
|---|---|---|
| Chat template | Hermes XML（`<tool_call>` / `<tool_response>`） | 不用特殊 token，纯 ASCII，BPE 友好；JSON 比 XML 容错差（多/少一个引号就炸） |
| 多轮处理 | **Python dispatcher**（不动 C++） | gpt-sft 训练代码硬编码 Alpaca 格式，改 C++ 大动干戈；dispatcher 让 C++ 当"单轮 worker" |
| 数据格式 | **Hermes → Alpaca 拍平** | 多轮对话中每个 assistant turn 是一条训练样本（history → assistant response） |
| 训练 | 复用 `gpt-sft/train/sft_train` | 一字不改 C++ 训练代码 |
| 推理 | `tool_chat` C++ + `dispatch.py` | C++ 只做"prompt in → text out"，dispatcher 做多轮循环 + 工具注册 |

## 数据 + 权重 一站式下载

```bash
cd gpt-toolcall
./scripts/run.sh download          # 拉 NousResearch/hermes-function-calling-v1 (~26 MB)
./scripts/run.sh download-weights  # 拉 gpt2-medium.safetensors (~1.4 GiB)
./scripts/run.sh convert           # Hermes JSONL → Alpaca JSON (~/data/toolcall-data.json)
```

镜像切换（与 gpt-bash 同约定）：

| 变量 | 取值 | 说明 |
|---|---|---|
| `HF_MIRROR` | `hf-mirror` (默认) | 中国大陆加速，~10 MB/s |
| `HF_MIRROR` | `huggingface` | HF 官方，需海外网络 |
| `ALL_PROXY` | `socks5h://host:port` | 走代理（如 `localhost:30000`）|

## 构建

```bash
./scripts/run.sh build             # cmake → build/tool_chat
```

产物：`build/tool_chat`（单轮 worker，prompt in → text out）。

## 训练

```bash
SIZE=medium EPOCHS=3 BATCH=4 ./scripts/train-background.sh
# 输出: data/toolcall-sft-medium.pth  (~1.7 GB)
```

> 与 `gpt-bash` 完全相同的训练参数与显存预算（batch=4 是为了避开 medium 在 16 GiB 卡的 OOM）。

## 推理

### CLI 多轮 dispatcher
```bash
./scripts/run.sh chat --query "北京今天天气怎么样?"
./scripts/run.sh chat --query "(1+2)*3 等于多少?"
```

### Python REPL
```bash
./scripts/run.sh chat
# 输入 "北京天气?"、"(1+2)*3" 等，回车得到答案
# 输入 quit 退出
```

### 直接用 dispatcher（写自己的工具）

```python
from chat.dispatch import register_tool, run_loop

@register_tool(
    name="my_tool",
    description="...",
    parameters={...},
)
def my_tool(arg: str) -> str:
    return json.dumps({"result": ...})

print(run_loop("用户问题", binary="./build/tool_chat",
               model="./data/toolcall-sft-medium.pth", size="medium"))
```

## 评测

```bash
./scripts/run.sh build
./scripts/run.sh download-weights    # 训后用自己训练的 .pth
python3 eval/tool_eval.py \
    --data data/smoke/toolcall-smoke.json \
    --binary build/tool_chat \
    --model data/toolcall-sft-medium.pth \
    --size medium \
    --no-cuda \
    --out logs/smoke-eval.json
```

输出 4 个维度的命中率：
- `valid_tc`：期望 tool_call 时，输出含合法 `<tool_call>{...}</tool_call>`
- `correct_tool`：工具名一致
- `correct_args`：arguments JSON keys 一致（值不查）
- `natural_language`：期望自然语言时，输出不含 tool_call

## smoke 验证（不需训练权重）

```bash
./scripts/run.sh build
./scripts/run.sh smoke
# 用预训练 gpt2-medium + 12 条手写 prompt 跑一遍
# 期望：模型输出接近乱码但流程跑通（不崩、JSON 解析正常）
```

## 架构图

```
┌──────────────────────────────────────────────────────────┐
│  用户: "北京天气?"                                       │
│         ↓                                                │
│  chat/dispatch.py:                                       │
│    1. 渲染 prompt (system + tools + user)                │
│    2. 写临时文件                                         │
│    3. 调 tool_chat --json --stop-on "</tool_call>"      │
│         ↓                                                │
│  chat/src/tool_chat.cpp:                                 │
│    prompt → BPE encode → GPT-2 → argmax → decode          │
│    (stop-on "</tool_call>" 时立即返回)                  │
│         ↓ {"text": "<tool_call>{...}</tool_call>",       │
│             "stopped_on": "</tool_call>"}                  │
│    4. 正则提取 JSON → 真执行工具                          │
│    5. 把结果拼成下一轮 prompt，循环                       │
│         ↓                                                │
│  最终: "北京今天 28°C，多云。"                            │
└──────────────────────────────────────────────────────────┘
```

## 关键文件

| 文件 | 角色 |
|---|---|
| `chat/chat_template.py` | **训练/推理唯一模板源**（1 字节都不能差） |
| `chat/src/tool_chat.cpp` | C++ 单轮 worker（prompt → text） |
| `chat/dispatch.py` | Python 多轮 dispatcher + 工具注册表 |
| `scripts/convert_data.py` | Hermes 多轮 → Alpaca 单条 |
| `scripts/train-background.sh` | 调 gpt-sft 训练 |
| `eval/tool_eval.py` | 4 维评分 |
| `data/smoke/toolcall-smoke.json` | 12 条手写 smoke |

## 与 gpt-bash / gpt-sft 的边界

- `gpt-sft/core/*`：完全不动（共享 BPE + safetensors + GPTModel）
- `gpt-bash/`：完全不动（与本项目平行）
- `gpt-toolcall/`：**全新**，独立训练 + 独立 dispatcher，独立 chat 二进制