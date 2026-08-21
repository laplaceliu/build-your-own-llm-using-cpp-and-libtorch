# 7.1 指令数据格式：Alpaca / ChatML 怎么选

> 第 7 章第 1 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**同一个数据集，不同格式模板，训练效果能差 10%。**

选模板比选模型还重要。

---

## 3 种主流格式

### 格式 1：Alpaca（Stanford）

```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is really nice today."
}
```

训练时拼成：
```
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
将以下句子翻译成英文

### Input:
今天天气真好

### Response:
The weather is really nice today.
```

### 格式 2：ChatML（OpenAI 风格）

```xml
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
将以下句子翻译成英文：今天天气真好<|im_end|>
<|im_start|>assistant
The weather is really nice today.<|im_end|>
```

### 格式 3：LLaMA-2 Chat（Meta）

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

将以下句子翻译成英文：今天天气真好 [/INST] The weather is really nice today. </s>
```

---

## 对比表

| 格式 | 优势 | 劣势 | 代表模型 |
|---|---|---|---|
| **Alpaca** | 简单、通用、易改 | 缺少对话结构 | Alpaca, Vicuna |
| **ChatML** | 对话清晰、OpenAI 风格 | 需要特殊 token | Qwen, InternLM |
| **LLaMA-2** | 经典、社区资源多 | 标签复杂 | LLaMA-2 Chat |

**本项目用 Alpaca**：简单、容易理解、GPT-2 默认词表都能跑。

---

## Alpaca 模板实现

```cpp
std::string format_alpaca(const std::string& instruction,
                          const std::string& input,
                          const std::string& output = "") {
    std::string prompt = "Below is an instruction that describes a task. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n" + instruction + "\n\n";
    if (!input.empty()) {
        prompt += "### Input:\n" + input + "\n\n";
    }
    prompt += "### Response:\n";
    
    if (!output.empty()) {
        prompt += output;
    }
    return prompt;
}

// 训练时只对 Response 部分算 loss
std::vector<int> tokenize_with_mask(const std::string& prompt,
                                     const std::string& response,
                                     BPETokenizer& tok) {
    auto prompt_ids = tok.encode(prompt);
    auto response_ids = tok.encode(response);
    
    std::vector<int> ids = prompt_ids;
    ids.insert(ids.end(), response_ids.begin(), response_ids.end());
    
    // loss mask: 0 for prompt, 1 for response
    std::vector<float> mask(prompt_ids.size(), 0.0f);
    mask.insert(mask.end(), response_ids.size(), 1.0f);
    
    return ids;
}
```

**关键**：只对 `### Response:\n` 之后的 token 算 loss，前面的 prompt 部分 mask 掉。

---

## 训练数据示例

```json
[
  {
    "instruction": "用 Python 写一个快速排序",
    "input": "",
    "output": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    ..."
  },
  {
    "instruction": "解释量子计算",
    "input": "",
    "output": "量子计算是一种利用量子比特（qubit）进行计算的新型计算范式..."
  },
  ...
]
```

---

## 评估数据：要不要分 prompt 和 response？

```cpp
// ❌ 错误：算整个序列的 loss
auto logits = model(ids);
auto loss = cross_entropy(logits, ids);  // 把 prompt 也当 label

// ✅ 正确：只算 response 部分
auto response_start = prompt_ids.size();
auto logits = model(ids);
auto response_logits = logits.slice(1, response_start);
auto response_targets = ids.slice(0, response_start);
auto loss = cross_entropy(response_logits, response_targets);
```

**原因**：模型应该学的是「如何回答」，不是「如何复述 prompt」。

---

## 一个常见坑

```cpp
// ❌ 训练时 prompt 和 output 顺序混乱
prompt = "### Response:\n" + output;
prompt += "### Instruction:\n" + instruction;  // 顺序反了

// ✅ 保持 Alpaca 标准顺序
prompt = "### Instruction:\n" + instruction +
         "\n\n### Response:\n" + output;
```

否则模型学到错乱的模板对应关系。

---

## 本项目用哪个？

本项目选 **Alpaca 风格**：
- ✅ GPT-2 词表原生支持（无特殊 token）
- ✅ 中文友好（不用 ChatML 的 `<|im_start|>` 特殊 token）
- ✅ 代码直接读懂（不用 XML）

---

## 下一步

**7.2 GPT-2 medium 355M**：能否在 C++ 跑得动

---

## 互动

你训练时用哪种格式？
- Alpaca（简单）
- ChatML（标准）
- 自定义（业务定制）

评论告诉我你的应用场景。
