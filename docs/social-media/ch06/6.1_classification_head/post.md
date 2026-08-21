# 6.1 分类头：把生成模型改成分类器

> 第 6 章第 1 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**「会写文章」的模型，怎么改成「判断垃圾邮件」？**

加一个分类头，冻住主干，只训练它。

---

## 改造前后对比

### 改造前（生成模型）

```
输入文本 → 12 层 Transformer → 词表投影 [vocab] → 下一个 token
输出：每个位置的下一个 token 概率
```

### 改造后（分类模型）

```
输入文本 → 12 层 Transformer → 取最后 token → 分类头 [num_classes]
输出：类别概率（比如 spam/ham）
```

**关键变化**：
1. 不再预测整个序列，只用最后 token 的隐藏状态
2. 输出维度：50,257（vocab）→ 2（spam/ham）

---

## 为什么用最后一个 token？

输入："This message is spam"
- 位置 0 (This) 的隐藏状态：只知道 "This"
- 位置 4 (spam) 的隐藏状态：看到了整句

**最后一个 token 的隐藏状态 = 整个句子的摘要**。

可视化：
```
"This"    →  h_0 (不完整)
" is"     →  h_1
" spam"   →  h_4 (看到全部)
分类头(h_4) → [0.95, 0.05] (95% spam)
```

---

## 实现

```cpp
struct GPTClassifier : torch::nn::Module {
    GPT backbone;          // 12 层 Transformer
    torch::nn::Linear head; // 分类头
    
    GPTClassifier(int d_model, int num_classes) {
        head = register_module("head",
            torch::nn::Linear(d_model, num_classes));
    }
    
    torch::Tensor forward(torch::Tensor ids) {
        // backbone 输出 [batch, seq, d_model]
        auto hidden = backbone.forward(ids);
        
        // 取最后一个 token
        auto last_hidden = hidden.select(1, -1);  // [batch, d_model]
        
        // 分类头
        return head(last_hidden);  // [batch, num_classes]
    }
};
```

---

## 冻结主干 vs 全量微调

### 方案 1：冻结主干（推荐入门）

```cpp
// 1. 冻结所有 backbone 参数
for (auto& p : backbone.parameters()) {
    p.set_requires_grad(false);
}

// 2. 只有 head 是可训练的
auto optimizer = Adam(head->parameters(), 1e-3);

// 优点：训练快（只更新 1,536 个参数 = 768×2）
// 缺点：效果上限低
```

### 方案 2：全量微调

```cpp
// 所有参数都可训练
auto optimizer = Adam(model->parameters(), 1e-4);

// 优点：效果更好
// 缺点：训练慢、显存大、容易过拟合
```

### 方案 3：部分微调（平衡）

```cpp
// 最后 2 层 Block + head 可训练，其余冻结
for (int i = 0; i < 10; ++i) {
    for (auto& p : backbone.blocks[i].parameters()) {
        p.set_requires_grad(false);
    }
}
auto params = std::vector<torch::Tensor>();
for (int i = 10; i < 12; ++i) {
    for (auto& p : backbone.blocks[i].parameters()) {
        params.push_back(p);
    }
}
for (auto& p : head->parameters()) {
    params.push_back(p);
}
auto optimizer = Adam(params, 1e-4);
```

---

## 训练配置

| 参数 | 推荐值 |
|---|---|
| `lr` | 5e-5（全量）或 1e-3（只 head） |
| `epochs` | 5 |
| `batch_size` | 8 |
| 优化器 | AdamW |
| 损失函数 | CrossEntropy |

---

## 数据准备

```cpp
struct SMSExample {
    std::string text;
    int label;  // 0 = ham, 1 = spam
};

// 平衡数据：spam/ham 各占 50%
// 划分：80% train, 10% val, 10% test
```

---

## 一个常见坑

```cpp
// ❌ 错误：忘了 backbone.eval()（虽然 batchnorm 在 GPT 里没有，但 dropout 影响）
model->train();  // OK，但 backbone 也会启用 dropout

// ✅ 更好：分开设模式
backbone->eval();   // 冻结的 backbone 用 eval
head->train();      // 只训练 head
```

**原因**：如果 backbone 还在 dropout，隐藏状态会有噪声，影响 head 学习。

---

## 下一步

**6.2 SMS Spam 微调**：完整流程跑一遍

---

## 互动

你做过分类微调吗？
- 情感分析
- 垃圾邮件
- 意图识别
评论区告诉我你的数据集。
