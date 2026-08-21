# 3.3 多头注意力：为什么是「多头」而不是单头

> 第 3 章第 3 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**为什么是「多头」而不是「一个更强的头」？**

答案：每个头关注不同的子空间，并行计算更高效。

---

## 一个直觉类比

一个医生看病 vs 多个专家会诊：
- **单头**：一个医生看 X 光、看验血、看症状——容易顾此失彼
- **多头**：放射科医生看 X 光 + 检验科医生看验血 + 内科医生看症状——各司其职

每个「头」就是一个「专科医生」，专注于不同特征。

---

## 数学定义

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

其中 head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
```

**关键**：把 d_model 维度拆成 num_heads 份，每份独立做注意力。

---

## GPT-2 small 实例

| 参数 | 值 |
|---|---|
| d_model | 768 |
| num_heads | 12 |
| 每个 head 维度 | 768 / 12 = **64** |
| 总参数 | 12 × 3 × 768 × 64 + 768² = **2.36M** |

**每个头 64 维，独立计算注意力**。

---

## 实现（两种风格）

### 风格 1：循环每个 head（简单但慢）

```cpp
class MultiHeadAttention : public torch::nn::Module {
    std::vector<CausalAttention> heads;
    torch::nn::Linear W_out{nullptr};
    
public:
    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> outputs;
        for (auto& head : heads) {
            outputs.push_back(head(x));  // 每个头独立算
        }
        return W_out(torch::cat(outputs, /*dim=*/-1));  // 拼接 + 投影
    }
};
```

### 风格 2：批量并行（高效）

```cpp
// 把 batch 维度扩展：让所有 head 一起算
auto Q = x @ W_q;       // [batch, seq, 768]
Q = Q.view({batch, seq, 12, 64});   // [batch, seq, 12, 64]
Q = Q.permute({0, 2, 1, 3});        // [batch, 12, seq, 64]
// 现在第 2 维是「头」，所有头一起算注意力
auto scores = Q @ K.transpose(-2, -1) / std::sqrt(64);
// ... softmax, mask, dropout ...
auto out = (attn @ V).permute({0, 2, 1, 3}).contiguous();
return W_out(out.view({batch, seq, 768}));
```

**关键操作**：`view` + `permute`，相当于把「12 个头」当成 batch 维度处理。

---

## 为什么要多头？3 个原因

### 原因 1：捕捉不同子空间

可视化 4 个头的注意力：
```
头 1：关注相邻词（语法）
头 2：关注长距离依赖（指代）
头 3：关注特定词性（动词）
头 4：关注整句（语义）
```

### 原因 2：并行计算效率

12 个头 × 64 维 = 12 × 64 = 768 维。
如果 1 个头 × 768 维 = 768 维，**参数一样多，但少了一层并行**。
GPU 对 batch 维度非常友好，多头天然适合并行。

### 原因 3：缓解过拟合

单头容易陷入「只关注某一类信息」，
多头强制模型在不同子空间都学点东西。

---

## GPT-2 配置速查

| 模型 | 层数 | 头数 | 维度 |
|---|---|---|---|
| **GPT-2 small** | 12 | 12 | 768 |
| GPT-2 medium | 24 | 16 | 1024 |
| GPT-2 large | 36 | 20 | 1280 |
| GPT-2 XL | 48 | 25 | 1600 |

**规律**：head_dim = 64 保持不变，靠加 depth 和 width 扩模型。

---

## 下一步

**3.4 注意力可视化**：你的模型在盯着哪个词

---

## 互动

你试过多头数量对最终效果的影响吗？
评论说说你的经验：12 头够用吗？
