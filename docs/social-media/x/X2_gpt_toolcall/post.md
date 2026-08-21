# X.2 gpt-toolcall：让模型学会调工具

> 扩展项目第 2 帖｜⭐⭐ 重点实战｜阅读 8 分钟｜口播 5 分钟
> 平台：★ 知乎 + ★★★ B 站

---

## 钩子

**让 LLM 调用 calculator / weather / shell？自己训练一个。**

gpt-toolcall：基于 Hermes-function-calling-v1 数据集训练。

---

## 项目结构

```
gpt-toolcall/
├── README.md
├── chat/
│   ├── tool_chat.cpp         # C++ 单轮推理
│   ├── dispatcher.py         # Python 多轮调度
│   └── tools/
│       ├── calculator.py
│       ├── weather.py
│       ├── shell.py
│       ├── search.py
│       └── datetime.py
├── data/
│   └── hermes_function_calling.jsonl  # 16k 样本
└── scripts/
    └── train.sh
```

---

## 训练数据：Hermes 风格

```json
{
  "conversations": [
    {"role": "user", "content": "北京今天多少度？"},
    {"role": "assistant", "content": "我需要查一下北京的天气"},
    {"role": "tool", "name": "weather", "content": "{\"temp\": 25, \"condition\": \"晴\"}"},
    {"role": "assistant", "content": "北京今天 25 度，天气晴朗。"}
  ],
  "tools": "[{\"name\": \"weather\", \"description\": \"查天气\", \"parameters\": {\"city\": {\"type\": \"string\"}}}]"
}
```

**特殊 token**：
- `<|tool_call|>{...}</|tool_call|>` 包裹工具调用 JSON

---

## 训练输出

```
[tool_chat] loaded gpt2-medium
[training] 16672 samples × 2 epoch = 33344 steps
epoch 0 loss=0.93  [38m]
epoch 1 loss=0.42  [37m]
Training done. Final loss: 0.42
```

**关键数据**（来自 gpt-toolcall 实测）：
- 13× 数据 + 少 epoch > 1× 数据 + 多 epoch（6 倍效果）
- 2 epoch 33k 步 ≈ 75 分钟（RTX 4080）

---

## 5 个内置工具

| 工具 | 功能 | 参数 |
|---|---|---|
| **calculator** | 数学计算 | `expression: string` |
| **weather** | 查天气 | `city: string` |
| **shell** | 执行命令（沙箱） | `command: string` |
| **search** | 联网搜索 | `query: string` |
| **datetime** | 当前时间 | `timezone: string?` |

---

## 多轮对话示例

```python
from dispatcher import ToolDispatcher

dispatcher = ToolDispatcher(
    model_path="./checkpoints/gpt2-medium-toolcall",
    tools=["calculator", "weather", "shell", "search", "datetime"]
)

# 多轮调用
result = dispatcher.run([
    {"role": "user", "content": "(25 + 17) * 3 是多少？"},
])

# 输出：
# 1. 模型思考：需要算 (25+17)*3
# 2. 调用 calculator: {"expression": "(25+17)*3"}
# 3. 工具返回: 126
# 4. 模型回答: (25+17)*3 = 126
```

**第二轮**：模型还能记住上下文继续追问：
```python
result = dispatcher.run([
    {"role": "user", "content": "(25+17)*3 是多少？"},
    {"role": "assistant", "content": "126"},
    {"role": "user", "content": "再除以 2 呢？"},
])

# 输出：调用 calculator: {"expression": "126 / 2"} → 63
```

---

## 关键代码：dispatcher 核心

```python
class ToolDispatcher:
    def __init__(self, model_path, tools):
        self.model = load_model(model_path)
        self.tools = {t.__name__: t for t in tools}
    
    def run(self, messages, max_rounds=5):
        for round_idx in range(max_rounds):
            # 1. 模型推理
            prompt = self.format_prompt(messages)
            response = self.model.generate(prompt)
            
            # 2. 解析 tool_call
            tool_call = parse_tool_call(response)
            
            if tool_call is None:
                # 没有 tool_call → 直接返回
                messages.append({"role": "assistant", "content": response})
                return response
            
            # 3. 执行工具
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            tool_result = self.tools[tool_name](**tool_args)
            
            # 4. 把结果喂回模型
            messages.append({
                "role": "assistant",
                "content": f"<|tool_call|>{json.dumps(tool_call)}</|tool_call|>"
            })
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
        
        return "Max rounds reached"
```

---

## 实测：5 个工具 5/5 匹配

| 用户 | 模型输出 | 工具调用 | 结果 |
|---|---|---|---|
| 「25+17」 | 思考... | calculator | 42 ✅ |
| 「北京天气」 | 思考... | weather(city=北京) | 25 度 ✅ |
| 「ls 当前目录」 | 思考... | shell(cmd=ls) | 文件列表 ✅ |
| 「今天几号」 | 思考... | datetime() | 2026-08-21 ✅ |
| 「Python 是谁发明的」 | 思考... | search(query=Python creator) | Guido ✅ |

**5/5 工具名匹配，端到端 dispatcher 跑通多轮**。

---

## C++ 单轮 + Python 多轮的分工

```
              ┌─ C++ 单轮推理（tool_chat.cpp）
              │  - 高效、无 Python 依赖
              │  - 输入 prompt，输出 text
              │
模型 ◄────────┤
              │
              └─ Python dispatcher
                 - 多轮对话循环
                 - 工具注册和调度
                 - 上下文管理
```

**优势**：
- C++ 推理快（适合热路径）
- Python 调度灵活（适合业务逻辑）

---

## 部署

```bash
# 1. 启动 HTTP 服务
$ python -m gpt_toolcall.serve --port 8000
[gpt-toolcall] serving on :8000

# 2. 调用（OpenAI 兼容协议）
$ curl -X POST http://localhost:8000/v1/chat \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt2-medium-toolcall",
        "messages": [{"role": "user", "content": "今天北京多少度"}]
    }'

# {"content": "我需要查天气", "tool_call": {"name": "weather", ...}}
```

---

## 下一步

**X.3 gpt-sft 框架**：CLI + HTTP 服务化

---

## 互动
你用 Function Calling 做过什么应用？
- ChatGPT 插件
- 自建 Agent
- 工具自动化

评论告诉我。
