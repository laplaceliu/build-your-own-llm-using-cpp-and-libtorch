# 5.2 AdamW 优化器：为什么不用 SGD

> 第 5 章第 2 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**训练 LLM 不用 SGD，用 AdamW。差别在哪？**

---

## 4 种优化器对比

```
SGD:    θ = θ - lr * grad
Momentum: θ = θ - lr * (β * v + grad)
              v = β * v + grad
RMSProp: θ = θ - lr * grad / √(mean(g²))
Adam:    = Momentum + RMSProp
AdamW:   = Adam + 正确的权重衰减
```

---

## Adam = Momentum + RMSProp

```cpp
auto optimizer = torch::optim::Adam(
    model.parameters(),
    torch::optim::AdamOptions(2.5e-4)   // lr
        .betas(std::make_tuple(0.9, 0.95))  // (β1, β2)
        .eps(1e-8)
);
```

**两个矩估计**：
- 一阶矩 `m`：梯度的移动平均（Momentum）
- 二阶矩 `v`：梯度平方的移动平均（RMSProp）

更新规则：
```
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
θ = θ - lr * m_hat / (√v_hat + eps)
```

---

## AdamW：解耦的权重衰减

```cpp
auto optimizer = torch::optim::AdamW(
    model.parameters(),
    torch::optim::AdamWOptions(2.5e-4)
        .weight_decay(0.1)  // ← 关键！
);
```

**问题**：Adam 的 L2 正则化有问题。
梯度 `grad = L' + λθ`，但 Adam 除以 `√v`，L2 项被「缩小」了。

**解决**：AdamW 把权重衰减**从梯度中解耦**：
```
θ = θ - lr * (m_hat / √v_hat + λθ)  // Adam：λ 项被缩放
θ = θ - lr * (m_hat / √v_hat) - lr * λθ  // AdamW：λ 项独立
```

**效果**：权重衰减更稳定，泛化更好。

---

## GPT-2 训练推荐配置

| 参数 | 值 | 来源 |
|---|---|---|
| `lr` | 2.5e-4 | GPT-2 论文 |
| `betas` | (0.9, 0.95) | GPT-2 论文 |
| `eps` | 1e-8 | 默认 |
| `weight_decay` | 0.1 | GPT-2 论文 |
| `batch_size` | 512（实际可能更小） | GPT-2 论文 |

**注意**：小数据集上 lr 建议降到 5e-5 ~ 1e-4。

---

## 学习率调参 4 个经验

1. **先小后大**：从 1e-5 开始，能收敛再往上加
2. **观察 loss 曲线**：loss 抖动 → lr 太大；loss 平 → lr 合适
3. **warmup 必要**：前 100–2000 步线性升 lr
4. **微调用更小 lr**：1e-5 ~ 5e-5（不要用预训练 lr）

---

## 一个常见误区

```cpp
// ❌ 错误：所有参数用同一个 lr
auto opt = Adam(model.parameters(), 1e-3);

// ✅ 更好：embedding 层用更小 lr（防过拟合）
std::vector<torch::optim::OptimizerParamGroup> groups;
groups.push_back({model.tok_emb.parameters(), 1e-4});
groups.push_back({model.blocks.parameters(), 1e-3});

auto opt = Adam(groups);
```

本项目暂不实现这个，保持简单。

---

## 完整优化器初始化代码

```cpp
// 包含 bias 在内全部参数
auto params = model.parameters();

auto optimizer = torch::optim::AdamW(
    params,
    torch::optim::AdamWOptions(2.5e-4)
        .betas(std::make_tuple(0.9, 0.95))
        .eps(1e-8)
        .weight_decay(0.1)
);

// 训练循环
for (auto& batch : dataloader) {
    optimizer.zero_grad();   // 关键！否则梯度累积
    
    auto logits = model.forward(batch.x);
    auto loss = cross_entropy(logits, batch.y);
    loss.backward();
    
    optimizer.step();
}
```

---

## 下一步

**5.3 训练循环**：从 batch 到 GPU 10 行代码

---

## 互动

你训练时遇过的 optimizer 坑：
- loss 不下降（SGD → Adam 试试）
- loss 爆炸（lr 太大）
- 过拟合（weight_decay 太小）

评论区聊聊。
