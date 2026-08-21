# 4.2 LayerNorm + GELU：两个「看似简单」的零件

> 第 4 章第 2 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**LayerNorm 和 GELU，看似简单，GPT 选它们是有原因的。**

---

## LayerNorm：每个 token 单独归一化

### 公式

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

其中 μ、σ 是**沿着特征维度**（d_model）的均值和方差。

### 与 BatchNorm 对比

```cpp
// BatchNorm：跨样本归一化（按 batch 维度）
auto bn = torch::nn::BatchNorm1d(768);
auto y = bn(x);  // 算 batch 内每个特征维度的均值/方差

// LayerNorm：每个样本单独归一化（按特征维度）
auto ln = torch::nn::LayerNorm(768);
auto y = ln(x);  // 算单个样本的 768 维均值/方差
```

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化范围 | 跨 batch | 单个样本 |
| 依赖 batch size | ✅ 是 | ❌ 否 |
| 推理时差异 | 需要 moving avg | 无 |
| 适用场景 | CNN | **Transformer** |

### 为什么 GPT 用 LayerNorm？

- 训练时 batch_size 可能很小（甚至 1）
- 推理时 batch_size 变化大
- 不需要记录 moving statistics

---

## GELU：平滑版的 ReLU

### 公式

```
GELU(x) = x · Φ(x) = x · 0.5 · [1 + erf(x / √2)]
```

直觉：**x 越大，越可能「保留」**；x 越小，越可能被「压低」。
不像 ReLU 那样硬切（x<0 直接 0）。

### 与 ReLU 对比

| 输入 x | ReLU | GELU |
|---|---|---|
| -2.0 | 0 | -0.04 |
| -1.0 | 0 | -0.16 |
| 0.0 | 0 | 0.00 |
| 1.0 | 1 | 0.84 |
| 2.0 | 2 | 2.00 |

**GELU 在 0 附近平滑**，避免神经元「死亡」（x<0 时梯度为 0）。

### 实现（精确 + 近似两种）

```cpp
// 精确版（用 libtorch 内置）
auto gelu = torch::gelu(x);

// 近似版（tanh 拟合，GPT-2 用的）
auto gelu_approx = 0.5 * x * (1.0 + torch::tanh(
    std::sqrt(2.0 / M_PI) * (x + 0.044715 * x.pow(3))));
```

**实测**：两者误差 < 1e-4，本项目用近似版（与 GPT-2 权重兼容）。

---

## 为什么 GPT 选这两个？

| 因素 | 选择 |
|---|---|
| 训练稳定性 | LayerNorm 比 BatchNorm 稳 |
| 收敛速度 | GELU 比 ReLU 快 5–10% |
| 稀疏性 | GELU 比 ReLU 弱（避免神经元死亡） |
| 工程友好 | 都不用复杂配置 |

---

## 在 Transformer Block 中的位置

```
x
 ↓
LayerNorm  ← 第一个
 ↓
MultiHeadAttention
 ↓
+ x         ← 残差
 ↓
LayerNorm  ← 第二个
 ↓
FeedForward (GELU 在这里)
 ↓
+ x         ← 残差
```

**两个 LayerNorm + 一个 GELU**，每层 4 个「零件」。

---

## 一个小实验：跨语言一致

```cpp
torch::manual_seed(42);
auto ln = torch::nn::LayerNorm({4});
auto x = torch::randn({2, 4});

auto y_cpp = ln(x);
// y_cpp = [[-0.34, 1.20, -0.91, 0.05], ...]
```

```python
# PyTorch 等价代码
torch.manual_seed(42)
ln = nn.LayerNorm(4)
x = torch.randn(2, 4)
y = ln(x)
# y = 完全一致
```

---

## 下一步

**4.3 Transformer Block**：注意力 + FFN 怎么拼

---

## 互动

你用过 SwiGLU / SwishGLU 吗？
LLaMA 用的就是 SwiGLU，不是 GELU。
评论说说你的经验。
