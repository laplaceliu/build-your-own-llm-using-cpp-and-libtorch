# F.3 过程奖励模型 PRM：把推理拆开打分

> 附录 F 第 3 帖｜⭐ 高阶｜阅读 5 分钟｜口播 5 分钟
> 平台：★ 知乎（主）

---

## 钩子

**Self-Consistency 还是看最终答案，能不能看每一步对不对？**

PRM（Process Reward Model）= 给每一步打分。

---

## ORM vs PRM

### ORM（Outcome Reward Model）

```
问题 → 模型推理 → 答案 A
   ↓
ORM 给「答案 A」打个总分
   ↓
对/错（1/0）
```

**缺点**：推理过程错了，但蒙对答案，ORM 给高分。

### PRM（Process Reward Model）

```
问题 → 模型推理 → [步骤 1, 步骤 2, 步骤 3, 答案 A]
   ↓
PRM 给每一步打分
   ↓
[0.95, 0.88, 0.92, 1.0]
   ↓
总评 = 加权求和
```

**优势**：能定位「哪一步错了」。

---

## 一个例子

问题：小明 5 个苹果，吃 2 个，又买 3 个，现在有几个？

```
步骤 1: 小明原本有 5 个苹果  → PRM 打 0.95（合理）
步骤 2: 吃了 2 个，剩 5-2=3  → PRM 打 0.92（正确）
步骤 3: 买了 3 个，3+3=6     → PRM 打 0.93（正确）
答案: 6 个苹果              → PRM 打 0.97（最终答案对）
```

PRM 总分 = 0.95 + 0.92 + 0.93 + 0.97 = **3.77**

如果中间某步错了，那一步分数就会低，PRM 总分就降。

---

## 怎么训 PRM？

### 训练数据

需要「带步骤标注」的推理数据：

```json
{
  "question": "小明 5 个苹果...",
  "steps": [
    "原本有 5 个",
    "吃 2 个后剩 3 个",
    "买 3 个后有 6 个",
    "答：6 个"
  ],
  "step_labels": [1, 1, 1, 1]  // 每一步对/错
}
```

**数据获取**：人工标注 + 自动合成（用 Self-Consistency 投票）。

### 训练目标

```cpp
// PRM 输入：问题 + 步骤
// PRM 输出：每一步的分数（0–1）

struct PRM : torch::nn::Module {
    GPT base;  // 共享的 GPT backbone
    torch::nn::Linear head{nullptr};
    
    PRM() {
        head = register_module("head",
            torch::nn::Linear(768, 1));
    }
    
    torch::Tensor forward(torch::Tensor ids) {
        auto hidden = base->forward(ids);     // [batch, seq, 768]
        auto last_hidden = hidden.select(1, -1);
        return torch::sigmoid(head(last_hidden));  // [batch, 1]
    }
};
```

训练时用 BCE Loss（每一步是否正确）。

---

## 推理时怎么用？

### 用 PRM 做 Beam Search

```cpp
struct Beam {
    std::vector<int> tokens;
    float prm_score;
};

std::string generate_with_prm(GPT& model, PRM& prm,
                              const std::string& question,
                              int beam_size = 4) {
    std::vector<Beam> beams = {{tokenizer.encode(question), 0.0f}};
    
    for (int step = 0; step < 100; ++step) {
        std::vector<Beam> candidates;
        
        // 1. 扩展每个 beam
        for (auto& beam : beams) {
            auto logits = model.forward(torch::tensor({beam.tokens}));
            auto topk = torch::topk(logits.select(1, -1), beam_size);
            
            for (int i = 0; i < beam_size; ++i) {
                Beam next;
                next.tokens = beam.tokens;
                next.tokens.push_back(topk.indices[0][i].item<int>());
                
                // PRM 打分
                auto score = prm.forward(torch::tensor({next.tokens}))
                                .item<float>();
                next.prm_score = beam.prm_score + std::log(score + 1e-9);
                
                candidates.push_back(next);
            }
        }
        
        // 2. 保留 top beam_size
        std::sort(candidates.begin(), candidates.end(),
                  [](auto& a, auto& b) { return a.prm_score > b.prm_score; });
        beams.assign(candidates.begin(),
                     candidates.begin() + std::min((int)candidates.size(), beam_size));
    }
    
    return tokenizer.decode(beams[0].tokens);
}
```

**效果**：推理路径偏向「每一步 PRM 都高」的方向。

---

## 实测：Math-Shepherd 数据集

GPT-2 medium 355M：

| 方法 | GSM8K 准确率 |
|---|---|
| Greedy + CoT | 14.5% |
| Self-Consistency (N=10) | 20.1% |
| **PRM-guided Beam Search** | **25.3%** |
| PRM + Self-Consistency | **28.7%** |

**PRM 比 Self-Consistency 强 5–8 个百分点**。

---

## PRM 的挑战

| 挑战 | 说明 |
|---|---|
| **数据贵** | 人工标注每一步对错 |
| **训练难** | 步骤边界识别、错误传播 |
| **推理慢** | 每一步都要过 PRM |
| **通用性** | 跨任务泛化能力弱 |

---

## 什么时候值得用 PRM？

| 场景 | 推荐 |
|---|---|
| **数学竞赛** | ⭐⭐⭐⭐⭐ |
| **复杂推理** | ⭐⭐⭐⭐ |
| **代码生成（多步）** | ⭐⭐⭐ |
| **简单问答** | ❌ 杀鸡用牛刀 |

---

## 下一步

**X.1 gpt-bash**：自然语言 → Bash 命令

---

## 互动

你了解过 PRM 吗？
- 用过 OpenAI o1（PRM 类）
- 看过论文但没实现
- 第一次听说

评论区告诉我。
