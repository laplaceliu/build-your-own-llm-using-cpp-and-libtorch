# 7.3 评估方案：Ollama 打分怎么用

> 第 7 章第 3 帖｜阅读 6 分钟｜口播 4 分钟
> 平台：★ 知乎（主）

---

## 钩子

**没有标准答案的任务，怎么评估？**

用另一个 LLM 当「裁判」。

---

## 评估的两难

指令微调后，模型输出：
- ✅ 风格变了（更友好）
- ✅ 结构化了（分点回答）
- ❓ 但「好不好」没法用 loss 衡量

**人工评估太贵，自动指标又不够**。

折中方案：**LLM-as-a-Judge**——用一个强 LLM 给你的模型打分。

---

## 整体流程

```
1. 准备测试集（100 条 instruction）
   ↓
2. 用你的模型生成回答（base 和 SFT 各生成一遍）
   ↓
3. 用 Ollama 上的 llama3 给每对回答打分（1-5 分）
   ↓
4. 统计分数分布，对比提升
```

---

## 第 1 步：准备测试集

```json
[
  {
    "instruction": "解释量子纠缠",
    "reference": "量子纠缠是..."  // 可选，参考答案
  },
  ...
]
```

测试集要避开训练集！

---

## 第 2 步：生成回答

```cpp
void generate_answers(GPT& model, std::vector<TestCase>& tests,
                     const std::string& output_file) {
    std::ofstream out(output_file);
    
    for (auto& test : tests) {
        auto prompt = format_alpaca(test.instruction, "");
        auto ids = tokenizer.encode(prompt);
        
        // 生成 200 token
        auto output_ids = generate(model, ids, 200, /*temperature=*/0.7);
        auto answer = tokenizer.decode(output_ids);
        
        out << "{\"instruction\": \"" << escape(test.instruction) << "\", "
            << "\"answer\": \"" << escape(answer) << "\"}\n";
    }
}
```

输出：
```json
{"instruction": "解释量子纠缠", "answer": "量子纠缠是指两个粒子..."}
{"instruction": "写 Python 排序", "answer": "def sort(arr):..."}
...
```

---

## 第 3 步：Ollama 打分

### 启动 Ollama

```bash
# 安装（一次性）
curl -fsSL https://ollama.ai/install.sh | sh

# 拉 llama3 模型（约 4.7 GB）
ollama pull llama3:8b

# 启动服务（自动）
ollama serve
```

### 打分 Prompt

```python
JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response.

User's instruction: {instruction}

Assistant's response: {response}

Rate the response on a scale of 1-5:
1 = Wrong or irrelevant
2 = Partially correct
3 = Correct but poorly written
4 = Good response
5 = Excellent response

Provide your rating and a brief justification.

Format: 
Rating: [1-5]
Justification: [reason]
"""
```

### 批量评估

```python
import requests
import json

def judge(instruction, response, model="llama3:8b"):
    prompt = JUDGE_PROMPT.format(instruction=instruction, response=response)
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False})
    return r.json()["response"]

# 批量打分
results = []
for test in test_cases:
    base_score = judge(test.instruction, base_answers[test.id])
    sft_score = judge(test.instruction, sft_answers[test.id])
    results.append({
        "instruction": test.instruction,
        "base": base_score,
        "sft": sft_score,
    })
```

---

## 第 4 步：分析结果

```python
import re

def extract_rating(text):
    m = re.search(r"Rating:\s*(\d)", text)
    return int(m.group(1)) if m else 3

base_scores = [extract_rating(r["base"]) for r in results]
sft_scores  = [extract_rating(r["sft"])  for r in results]

print(f"Base  avg: {sum(base_scores)/len(base_scores):.2f}")
print(f"SFT   avg: {sum(sft_scores)/len(sft_scores):.2f}")
print(f"Improvement: {sum(sft_scores) - sum(base_scores):+d}")
```

输出：
```
Base  avg: 2.34
SFT   avg: 3.87
Improvement: +153 (out of 500 max)
```

**SFT 后平均提升 1.5 分（满分 5）**。

---

## 分数分布

```
        Base    SFT
1 ★     18%      4%
2 ★★    32%     12%
3 ★★★   30%     28%
4 ★★★★  15%     38%
5 ★★★★★  5%     18%
```

**4-5 分的比例从 20% 涨到 56%**。

---

## 一个常见坑

```python
# ❌ 错误：让模型自己比较（容易有偏好偏差）
prompt = "Compare these two responses and pick the better one: A: ..., B: ..."

# ✅ 正确：分别打分（更客观）
for response in [base_answer, sft_answer]:
    score = judge(instruction, response)
```

---

## 评估成本

| 项 | 数量 | 单价 | 总价 |
|---|---|---|---|
| 测试集 | 100 条 | — | — |
| Ollama 本地 | 200 次（base + SFT） | 免费 | $0 |
| GPT-4 API（备选） | 200 次 | $0.03/1k tokens | ~$5 |

**本地 Ollama = 免费**，但要 8 GB 显存跑 llama3:8b。

---

## 下一步

**7.4 微调前后对比**：同一个 prompt 两种回答

---

## 互动

你用 LLM-as-a-Judge 评估过吗？
- 用 Ollama 本地（免费）
- 用 GPT-4（贵但强）
- 自己手评（最准）

评论区分享你的评估方案。
