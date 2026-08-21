# D.1 学习率调度：Warmup + Cosine

> 附录 D 第 1 帖｜阅读 4 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**lr 设好了就行？训练中后期要不要变？**

要，而且分两段：先升后降。

---

## 为什么要调度？

**固定 lr 的问题**：
- 训练初期：lr 太大 → loss 震荡、可能发散
- 训练后期：lr 太大 → 在最优解附近跳来跳去

**解决方案**：
- 前期：lr 从小升到大（warmup）
- 后期：lr 从大降到 0（cosine decay）

---

## 两阶段曲线

```
lr
│
│      /\
│     /  \___________
│    /                \
│   /                   \____
│  /                          \___
│ /________________________________
└──────────────────────────────── time
   warmup      cosine decay
   (前 1-5%)    (剩余 95-99%)
```

---

## Warmup 实现

```cpp
float get_lr_warmup(int step, int warmup_steps, float peak_lr) {
    if (step < warmup_steps) {
        return peak_lr * step / warmup_steps;  // 线性升
    }
    return peak_lr;
}
```

**前 N 步**：lr 从 0 线性升到 peak。
**为什么**：避免早期大 lr 破坏随机初始化的权重。

---

## Cosine Decay 实现

```cpp
float get_lr_cosine(int step, int warmup_steps, int total_steps,
                    float peak_lr, float min_lr = 0.0f) {
    if (step < warmup_steps) {
        return peak_lr * step / warmup_steps;
    }
    
    // 余弦衰减
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    progress = std::min(progress, 1.0f);  // 防止越界
    
    return min_lr + (peak_lr - min_lr) * 0.5f * (1.0f + std::cos(M_PI * progress));
}
```

**曲线**：从 peak 沿余弦曲线降到 min_lr（一般 = 0.1 × peak）。

---

## GPT-2 推荐配置

| 参数 | 值 |
|---|---|
| `peak_lr` | 2.5e-4 |
| `min_lr` | 2.5e-5（peak 的 10%） |
| `warmup_steps` | 2000 |
| `total_steps` | 600,000（10 epoch × 60k steps） |

**warmup 占比 ≈ 0.3%**，cosine 占比 ≈ 99.7%。

---

## 在 LibTorch 中使用

LibTorch 提供 `LRScheduler`：

```cpp
#include <torch/optim/lr_scheduler.h>

auto optimizer = AdamW(params, 2.5e-4);

auto scheduler = torch::optim::CosineAnnealingLR(
    optimizer,
    /*T_max=*/total_steps - warmup_steps,
    /*eta_min=*/2.5e-5
);

// 训练循环
for (int step = 0; step < total_steps; ++step) {
    // ... forward / backward / step ...
    optimizer.step();
    
    // warmup：手动设 lr
    if (step < warmup_steps) {
        float lr = peak_lr * step / warmup_steps;
        for (auto& group : optimizer.param_groups()) {
            group.options().set_lr(lr);
        }
    } else {
        scheduler.step();  // 余弦衰减
    }
}
```

---

## 3 个调参经验

1. **warmup 不能省**：尤其是大模型（>1B），不 warmup 必崩
2. **cosine 比 step decay 平滑**：最终收敛更好
3. **min_lr 不要设 0**：保留一点学习能力，有助于跳出局部最优

---

## 下一步

**D.2 梯度裁剪 + 混合精度**

---

## 互动

你训练时用哪种 lr 调度？
- 固定 lr（最简单）
- Step decay（每 N 步降一半）
- Warmup + Cosine（GPT 标配）
- 其他

评论告诉我。
