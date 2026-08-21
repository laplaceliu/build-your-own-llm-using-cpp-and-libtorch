# 5.1 损失函数：交叉熵 + 困惑度

> 第 5 章第 1 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**loss 11 是什么水平？loss 3 呢？困惑度能告诉你答案。**

---

## 交叉熵损失

### 公式

```
Loss = -mean(log(p[target]))
```

直觉：**模型给「正确答案」的概率越大，loss 越小**。

### 实现

```cpp
torch::Tensor compute_loss(torch::Tensor logits, torch::Tensor targets) {
    // logits: [batch, seq, vocab] = [8, 1024, 50257]
    // targets: [batch, seq] = [8, 1024]
    
    // 1. 展平
    auto flat_logits = logits.view({-1, logits.size(-1)});  // [8*1024, 50257]
    auto flat_targets = targets.view({-1});                 // [8*1024]
    
    // 2. 交叉熵
    auto loss = torch::nn::functional::cross_entropy(
        flat_logits, flat_targets);
    
    return loss;
}
```

**重要细节**：LibTorch 的 `cross_entropy` **自动包含 softmax**，不要再额外 softmax！

---

## 文本生成任务的标签

```cpp
// 输入:  [我, 爱, 吃, 苹果]
// 输出:  [爱, 吃, 苹果, 。]  ← 右移一位
// 损失: 预测「爱」用「我」、预测「吃」用「爱」...
```

代码：
```cpp
auto inputs  = torch::tensor({1, 2, 3, 4});  // 我 爱 吃 苹果
auto targets = torch::tensor({2, 3, 4, 5});  // 爱 吃 苹果 。

auto logits = model.forward(inputs);  // [1, 4, 50257]
auto loss = compute_loss(logits, targets);
```

---

## 困惑度（PPL）

```
Perplexity = e^Loss
```

| Loss | PPL | 含义 |
|---|---|---|
| 0 | 1 | 完美预测 |
| 1 | 2.7 | 在 2–3 个候选中犹豫 |
| 3 | 20 | 像在 20 个候选里猜 |
| 11 | 60,000 | 随机初始化的 GPT-2 |
| 7 | 1,100 | 训练一半 |
| 5 | 148 | 训练好（小型 GPT） |

**困惑度 = 模型「等效」在多少个词里选一个**。
PPL=20 意味着模型相当于在 20 个候选里选，难度不大。

---

## GPT-2 训练曲线参考

```
初始化 loss = 11（= ln(50257)）
训练 1 epoch:  loss = 7.5
训练 10 epoch: loss = 5.0
训练 100 epoch: loss = 3.5（overfit 前）
```

---

## 实战代码

```cpp
// 训练循环里
for (int epoch = 0; epoch < 10; ++epoch) {
    auto batch = loader.next_batch();  // {x, y}
    auto logits = model.forward(batch.first);
    auto loss = compute_loss(logits, batch.second);
    
    std::cout << "epoch " << epoch
              << " loss=" << loss.item<float>()
              << " ppl=" << std::exp(loss.item<float>())
              << std::endl;
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

输出：
```
epoch 0 loss=10.97 ppl=58600
epoch 1 loss=8.32  ppl=4123
...
epoch 9 loss=5.10  ppl=164
```

---

## 一个常见坑

```cpp
// ❌ 错误：先 softmax 再算 cross_entropy
auto probs = torch::softmax(logits, -1);
auto loss = -torch::log(probs[targets]).mean();  // 数值不稳定

// ✅ 正确：让 cross_entropy 内部做 softmax
auto loss = torch::nn::functional::cross_entropy(logits, targets);
```

**原因**：手动 softmax + log 在极端概率时数值不稳定（log(0) = -inf）。

---

## 下一步

**5.2 AdamW 优化器**：为什么不用 SGD

---

## 互动

你做训练时见过最低 loss / PPL 是多少？
- PPL < 10（极好）
- PPL 10–50（良好）
- PPL 50–200（一般）
- PPL > 200（需要调参）

评论告诉我。
