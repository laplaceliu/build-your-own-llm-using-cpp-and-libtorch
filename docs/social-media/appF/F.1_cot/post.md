# F.1 Chain-of-Thought：让模型「自言自语」

> 附录 F 第 1 帖｜阅读 4 分钟｜口播 3 分钟
> 平台：★★ 小红书（主）

---

## 钩子

**让模型一步步想，数学题正确率翻倍。**

---

## 一个实验

测试 prompt：「小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？」

### 无 CoT

```
答案：5 - 2 + 3 = 6
```
模型直接给答案，**答对但靠「直觉」**。

### 有 CoT

```
让我一步步算：
1. 小明有 5 个苹果
2. 吃了 2 个 → 5 - 2 = 3 个
3. 又买了 3 个 → 3 + 3 = 6 个
答：小明现在有 6 个苹果。
```
**模型展示推理过程，正确率大幅提升**。

---

## CoT 三种形式

### 1. Few-shot CoT（最稳）

```
Q: 小李有 3 支笔，给了小红 1 支，又买了 2 支，现在有几支？
A: 让我想：
   - 原本 3 支
   - 给了小红 1 支 → 3 - 1 = 2 支
   - 买了 2 支 → 2 + 2 = 4 支
   答：小李现在有 4 支笔。

Q: 小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？
A:
```
**关键**：先给 1 个带推理过程的示例，模型学到「套路」。

### 2. Zero-shot CoT（最简单）

```
Q: 小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？
Let's think step by step.
```

**关键**：一句「let me think step by step」就能让模型自动展开推理。

### 3. Auto-CoT（自动构造示例）

```cpp
// 1. 用 LLM 自动生成示例
auto examples = generate_cot_examples(test_questions, /*k=*/8);

// 2. 加到 prompt
prompt += format_few_shot(examples);
```

---

## 实测数据：GSM8K 数据集

GPT-2 medium 355M 不同 prompt 策略：

| Prompt 策略 | GSM8K 准确率 |
|---|---|
| 直接答题 | 2.1% |
| Few-shot（无 CoT） | 3.8% |
| **Zero-shot CoT** | **8.7%** |
| **Few-shot CoT** | **14.5%** |
| Self-Consistency（5 样本） | **19.3%** |

**CoT 让小模型也能做多步推理**。

---

## 为什么 CoT 有用？

### 直觉 1：分解复杂任务

```
「A + B + C」
   ↓ 分解
「先算 A + B = X」→「再算 X + C」
每一步都简单，整体就不难。
```

### 直觉 2：模型可以自我检查

```
答案 6
↓ 检查
「5 - 2 = 3, 3 + 3 = 6, 对的」← 不展示这个过程会出错
```

### 直觉 3：训练数据中推理痕迹

- LLM 训练语料里有大量「让我想一下...」
- Few-shot 给了示范，模型模仿

---

## 3 个使用场景

| 场景 | 推荐 |
|---|---|
| **算术 / 数学题** | ⭐⭐⭐⭐⭐（必备） |
| **逻辑推理 / 多步规划** | ⭐⭐⭐⭐⭐（必备） |
| **代码生成 / Debug** | ⭐⭐⭐⭐（显著提升） |
| 简单事实问答 | ⭐⭐（不必，反而慢） |
| 创意写作 | ⭐（破坏流畅性） |

---

## 在 LibTorch 中实现 Zero-shot CoT

```cpp
std::string apply_cot(const std::string& question) {
    return question + "\n\nLet's think step by step.";
}

// 训练时：让模型学会以 CoT 形式回答
auto prompt = format_alpaca(question, "") + "\nLet's think step by step.";
auto response = model.generate(prompt);
```

---

## 一个常见误区

```cpp
// ❌ 错误：CoT 用在所有任务
prompt = "What's the capital of France? Let's think step by step.";
// → 浪费 token 还降低准确率

// ✅ 正确：只在需要推理的任务用
prompt = "小明有 5 个苹果... Let's think step by step.";
// → 数学题、推理题才用
```

---

## 下一步

**F.2 Self-Consistency**：多次采样 + 投票

---

## 互动

你用 CoT 提升过哪种任务？
- 数学题
- 推理题
- 代码题
- 没试过

评论区告诉我。
