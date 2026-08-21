# F.2 Self-Consistency：多次采样 + 投票

> 附录 F 第 2 帖｜阅读 4 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**同一个问题生成 5 次，多数答案就是正确答案。**

Self-Consistency：CoT 的「升级版」。

---

## 核心思想

```
问题
  ↓ 生成 N 条推理路径（temperature > 0）
[路径 1: 答案 A]
[路径 2: 答案 B]
[路径 3: 答案 A]
[路径 4: 答案 A]
[路径 5: 答案 C]
  ↓ 多数投票
最终答案 = A（3/5 票）
```

**直觉**：正确答案往往有多种推理路径，错误答案各有不同。

---

## 实现

```cpp
struct ReasoningResult {
    std::string reasoning;  // 推理过程
    std::string answer;     // 最终答案
};

std::string self_consistency(GPT& model, const std::string& question,
                              int n_samples = 5) {
    std::map<std::string, int> vote_count;
    
    for (int i = 0; i < n_samples; ++i) {
        // 1. 生成（高温度采样）
        auto prompt = question + "\nLet's think step by step.";
        auto response = generate(model, prompt,
            /*max_new=*/300, /*temperature=*/0.8);
        
        // 2. 提取答案
        auto answer = extract_final_answer(response);
        
        // 3. 投票
        vote_count[answer]++;
    }
    
    // 4. 返回票数最高的
    std::string best;
    int max_count = 0;
    for (auto& [ans, count] : vote_count) {
        if (count > max_count) {
            best = ans;
            max_count = count;
        }
    }
    return best;
}
```

---

## 答案提取

```cpp
std::string extract_final_answer(const std::string& response) {
    // 找 "answer is X" 模式
    std::regex re(R"(answer is\s*(\d+(?:\.\d+)?))", std::regex::icase);
    std::smatch match;
    if (std::regex_search(response, match, re)) {
        return match[1].str();
    }
    
    // 兜底：取最后一个数字
    auto last_num = response.find_last_of("0123456789");
    return last_num != std::string::npos
         ? std::string(1, response[last_num])
         : "";
}
```

---

## 实测数据：GSM8K

| 方法 | 准确率 | 推理次数 |
|---|---|---|
| 贪心 + CoT | 14.5% | 1 |
| Sampling + CoT（avg） | 13.2% | 1 |
| **Self-Consistency (N=5)** | **18.7%** | 5 |
| **Self-Consistency (N=10)** | **20.1%** | 10 |
| **Self-Consistency (N=20)** | **21.4%** | 20 |

**5 次采样就已经显著提升**。

---

## N 怎么选？

```
准确率
   ↑
21% ┤                              ●─── (N=20)
    │                       ●─────
20% ┤                ●─────
    │         ●──────
19% ┤  ●─────
    │ /
14% ┤/ (N=1)
    │
    └────────────────────────→ N
    1    5    10    15    20
```

**甜点：N=5–10**
- N=5：性价比最高
- N=10：再涨 1.5%
- N=20：边际收益递减

---

## 适用与不适用

| 适用 | 不适用 |
|---|---|
| 数学题（有明确答案） | 开放问答（没有标准答案） |
| 推理题 | 创意写作 |
| 多选题 |  |
| 代码题（多次 debug） |  |

---

## 3 个成本考量

| 成本 | 量级 |
|---|---|
| 时间 | N × 单次推理 |
| 显存 | 不变（多次独立推理） |
| 钱（API） | N × 单价 |

**Self-Consistency 不省钱**：5 次推理 ≈ 5 倍成本。

---

## 一个常见坑

```cpp
// ❌ 错误：temperature = 0（贪心）
// N 次采样会得到完全相同的结果，投票失效
auto response = generate(model, prompt, temperature=0);

// ✅ 正确：temperature = 0.5–0.8
auto response = generate(model, prompt, temperature=0.7);
```

---

## 下一步

**F.3 过程奖励模型 PRM**：把推理拆开打分

---

## 互动

你用 Self-Consistency 做过哪种任务？
- 数学题
- 推理题
- 没试过，但想试

评论区告诉我。
