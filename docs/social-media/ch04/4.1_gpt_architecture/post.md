# 4.1 GPT 整体架构：一张图看懂 124M 参数

> 第 4 章第 1 帖｜阅读 7 分钟｜口播 4 分钟
> 平台：★ 知乎（主）

---

## 钩子

**GPT-2 small 124M 参数到底是怎么分布的？**

一张图 + 一个饼图，看完就懂。

---

## 整体架构（自上而下）

```
输入文本 "Hello world"
       ↓ Tokenize
Token IDs [15496, 995]
       ↓ Embedding
[batch, seq, 768]   ← 词向量 + 位置向量
       ↓
┌──────────────────────────────────┐
│ Transformer Block × 12            │  ← 12 层堆叠
│  ├─ LayerNorm                    │
│  ├─ MultiHeadAttention (12 heads) │
│  ├─ 残差连接                      │
│  ├─ LayerNorm                    │
│  ├─ FeedForward (768→3072→768)   │
│  └─ 残差连接                      │
└──────────────────────────────────┘
       ↓
[batch, seq, 768]   ← 最终隐藏状态
       ↓ LayerNorm
       ↓ Linear(768, 50257)   ← 词表投影
       ↓
Logits [batch, seq, 50257]
       ↓ Softmax
下一个 token 的概率分布
```

---

## GPT-2 small 配置（124M 参数）

| 参数 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 50,257 | 词表大小 |
| `n_positions` | 1,024 | 最大序列长度 |
| `n_embd` (d_model) | 768 | 隐藏维度 |
| `n_layer` | 12 | Transformer Block 数量 |
| `n_head` | 12 | 注意力头数 |
| 总参数 | **124M** | |

---

## 参数分布饼图

```
Embedding (38.7M, 31%)
├─ tok_emb: 50,257 × 768 = 38.60M
└─ pos_emb: 1,024 × 768 = 0.79M

Per-Block × 12 = 85.1M (69%)
├─ LayerNorm: 1,536 × 2 = 3,072
├─ Attention:
│   ├─ W_qkv: 768 × (3×768) = 1.77M
│   ├─ W_out: 768 × 768 = 0.59M
│   └─ bias: 3×768 + 768 = 3,072
├─ FeedForward:
│   ├─ W1: 768 × 3072 = 2.36M
│   ├─ W2: 3072 × 768 = 2.36M
│   └─ bias: 3072 + 768 = 3,840
└─ Block 合计 ≈ 7.10M
× 12 = 85.1M

输出 LayerNorm: 1,536
```

**3 个关键洞察**：
1. **Embedding 占 31%**：词表大，但只算一次
2. **每个 Block 占 7.1M**：12 层堆叠 = 85M（最大块）
3. **FFN ≈ Attention × 2.7 倍**：FFN 是参数大户

---

## 每一层 Transformer Block 内部

```cpp
struct TransformerBlock : torch::nn::Module {
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    MultiHeadAttention attn{nullptr};
    FeedForward ffn{nullptr};
    
    torch::Tensor forward(torch::Tensor x) {
        // Pre-Norm + 残差
        x = x + attn(ln1(x));   // 注意力子层
        x = x + ffn(ln2(x));    // FFN 子层
        return x;
    }
};
```

**Pre-Norm**（GPT 用）：先 LayerNorm，再算注意力，最后残差。
**对比 Post-Norm**（原 Transformer）：先算注意力，再 LayerNorm，最后残差。

| 方案 | 训练稳定性 | 性能 |
|---|---|---|
| **Pre-Norm** (GPT) | ✅ 不用 warmup | 略低 |
| Post-Norm (原论文) | ❌ 需要 warmup | 略高 |

GPT 选 Pre-Norm = 工程友好。

---

## 数据流形状追踪

```
输入: ids [batch=2, seq=8]   (int64)
  ↓ Embedding(50257, 768)
[2, 8, 768]   (float32)
  ↓ + pos_emb
[2, 8, 768]
  ↓ 12 × TransformerBlock
[2, 8, 768]   ← 形状不变，每层做特征变换
  ↓ Final LayerNorm
[2, 8, 768]
  ↓ Linear(768, 50257)
[2, 8, 50257]   ← 每个位置对 50257 个词的预测分数
```

**关键**：所有中间层都是 `[batch, seq, d_model]`，形状不变。

---

## 下一步

**4.2 LayerNorm + GELU**：两个「看似简单」的零件

---

## 互动

你觉得 124M 参数够用吗？
- 124M（小项目 / 学习）
- 355M（中等任务）
- 1.5B+（严肃产品）

评论区告诉我你的应用场景。
