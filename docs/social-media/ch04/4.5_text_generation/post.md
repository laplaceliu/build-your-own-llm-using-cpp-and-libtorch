# 4.5 文本生成：贪心 vs 采样 vs top-k

> 第 4 章第 5 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★★ 小红书（主）

---

## 钩子

**同一个 prompt，三种生成方式差别有多大？**

---

## 3 种解码策略对比

测试 prompt：「Once upon a time」

| 策略 | 输出 | 特点 |
|---|---|---|
| **贪心** | "Once upon a time there was a little girl who..." | 重复、保守 |
| **温度 0.8** | "Once upon a time, in a kingdom far away, a young prince..." | 多样、有创意 |
| **Top-k 50** | "Once upon a time when dinosaurs roamed the earth..." | 兼顾多样和合理 |

---

## 1. 贪心解码（Greedy）

```cpp
int64_t generate_greedy(GPT& model, torch::Tensor ids, int max_new) {
    for (int i = 0; i < max_new; ++i) {
        auto logits = model.forward(ids);  // [batch, seq, vocab]
        auto last_logits = logits.select(1, -1);  // [batch, vocab]
        auto next_id = last_logits.argmax(-1);    // 取最大
        
        ids = torch::cat({ids, next_id.unsqueeze(0)}, 1);
    }
    return ids;
}
```

**问题**：
- 永远选最大概率的 token → 输出无聊、循环
- 「the the the the」就是因为贪心选了同一个词

---

## 2. 温度采样（Temperature Sampling）

```cpp
int64_t generate_temperature(GPT& model, torch::Tensor ids,
                              int max_new, float temperature) {
    for (int i = 0; i < max_new; ++i) {
        auto logits = model.forward(ids).select(1, -1);
        logits = logits / temperature;     // ← 关键：除以温度
        auto probs = torch::softmax(logits, -1);
        auto next_id = torch::multinomial(probs, 1);  // 按概率采样
        ids = torch::cat({ids, next_id}, 1);
    }
    return ids;
}
```

**温度的影响**：
| temperature | 分布 | 输出 |
|---|---|---|
| 0.1 | 极度集中（≈ 贪心） | 重复、保守 |
| 0.7 | 适度分散 | 多样但合理 |
| 1.0 | 原生分布 | 平衡 |
| 1.5 | 极度分散 | 混乱、胡言乱语 |

**直觉**：temperature 越小 = 越「确定」，越大 = 越「随机」。

---

## 3. Top-k 采样

```cpp
int64_t generate_topk(GPT& model, torch::Tensor ids,
                       int max_new, int k, float temperature) {
    for (int i = 0; i < max_new; ++i) {
        auto logits = model.forward(ids).select(1, -1);
        logits = logits / temperature;
        
        // 只保留 top-k 的概率
        auto topk_vals, auto topk_idx = torch::topk(logits, k, -1);
        auto mask = torch::full_like(logits, -1e9);
        mask.scatter_(-1, topk_idx, topk_vals);
        
        auto probs = torch::softmax(mask, -1);
        auto next_id = torch::multinomial(probs, 1);
        ids = torch::cat({ids, next_id}, 1);
    }
    return ids;
}
```

**核心**：先把概率排序，只在前 k 个里采样。

| k | 效果 |
|---|---|
| k=1 | ≈ 贪心 |
| k=10 | 保守但不死板 |
| k=50 | GPT-2 默认 |
| k=50257 | ≈ 全 vocab 采样 |

---

## 4. Top-p（Nucleus）采样

最常用，比 top-k 更智能：
- 不固定 k，而是「累计概率到 p」为止
- 长尾时 k 小，高峰时 k 大

```cpp
auto sorted_probs = torch::sort(probs, /*dim=*/-1, /*descending=*/true);
auto cumsum = sorted_probs.cumsum(-1);
auto mask = cumsum <= 0.9;  // p = 0.9
mask.scatter_(-1, sorted_idx, mask.gather(-1, sorted_idx));
probs = probs * mask;
```

---

## 实战对比

prompt：「The capital of France is」

| 策略 | 输出 |
|---|---|
| 贪心 | "Paris."（稳定正确） |
| T=0.7 | "Paris." 或 "Lyon." |
| Top-k 50 | "Paris."（50 选 1，Paris 概率最高） |
| Top-p 0.9 | "Paris."（累计概率到 90% 时包含 Paris） |

**实测**：对于知识问答，贪心/Top-p 都靠谱；
对于创作类（写故事），Top-k 50 + T=0.8 最好。

---

## 推荐配置

| 任务 | temperature | top_k | top_p |
|---|---|---|---|
| 知识问答 | 0.0–0.3 | — | — |
| 翻译 | 0.0 | — | — |
| 摘要 | 0.3–0.5 | 50 | 0.9 |
| 创意写作 | 0.7–1.0 | 50 | 0.9 |
| 代码生成 | 0.2–0.4 | 40 | 0.9 |

---

## 关键 API

```cpp
// 贪心
auto next = logits.argmax(-1);

// 按概率采样
auto next = torch::multinomial(probs, 1);

// 温度
logits = logits / temperature;

// Top-k
auto [vals, idx] = torch::topk(logits, k, -1);
```

---

## 下一步

**5.1 损失函数**：交叉熵 + 困惑度

---

## 互动

你做生成时用哪种解码？
- 贪心（QA / 翻译）
- 采样（创作）
- Beam search（论文里见过）

评论告诉我你的应用场景。
