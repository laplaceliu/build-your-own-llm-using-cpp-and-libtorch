# 4.3 Transformer Block：注意力 + FFN 怎么拼

> 第 4 章第 3 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**Transformer Block = 注意力子层 + FFN 子层 + 2 个残差。**

拼起来就 4 个步骤。

---

## 4 步组装

```cpp
torch::Tensor transformer_block(torch::Tensor x) {
    // 1. 注意力子层（Pre-Norm + 残差）
    auto shortcut = x;
    x = ln1(x);
    x = attn(x);
    x = shortcut + x;  // 残差
    
    // 2. FFN 子层（Pre-Norm + 残差）
    shortcut = x;
    x = ln2(x);
    x = ffn(x);
    x = shortcut + x;  // 残差
    
    return x;
}
```

**就这么简单**，12 层堆叠就是 GPT。

---

## 残差连接：为什么必要？

```cpp
// 没残差的版本
x = f(ln(x));
```

**问题**：深层网络梯度消失 → 训练不动。

```cpp
// 有残差的版本
x = x + f(ln(x));  // 梯度能直接沿 + 号回传
```

**效果**：梯度信号能绕过非线性，直接从输出传回输入。

可视化：
```
x → [LN → Attn] ─┐
  ↑              ├─ + → 输出
  └──────────────┘
       残差路径
```

---

## FeedForward：2 个线性层 + GELU

```cpp
struct FeedForward : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    
    FeedForward(int d_model) {
        int d_ff = 4 * d_model;  // GPT-2 用 4 倍
        fc1 = register_module("fc1",
            torch::nn::Linear(d_model, d_ff));
        fc2 = register_module("fc2",
            torch::nn::Linear(d_ff, d_model));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = fc1(x);          // [batch, seq, 768] → [batch, seq, 3072]
        x = torch::gelu(x);  // 激活
        x = fc2(x);          // [batch, seq, 3072] → [batch, seq, 768]
        return x;
    }
};
```

**关键比例**：d_ff = 4 × d_model = 3072。

---

## Pre-Norm vs Post-Norm

### Pre-Norm（本项目用）

```cpp
x = x + sublayer(layer_norm(x));
```

### Post-Norm（原 Transformer）

```cpp
x = layer_norm(x + sublayer(x));
```

| 方案 | 训练稳定性 | 性能 |
|---|---|---|
| **Pre-Norm** | ✅ 不用 warmup，梯度好 | 略低 |
| Post-Norm | ❌ 需要 warmup，深层不稳 | 略高 |

**GPT 用 Pre-Norm**：实际工程训练更友好。

---

## GPT-2 Block 完整配置

```cpp
struct GPTBlock : torch::nn::Module {
    int d_model = 768;
    LayerNorm ln1{768}, ln2{768};
    MultiHeadAttention attn{d_model, 12, 1024};  // d_in, n_head, context_length
    FeedForward ffn{d_model};
    
    torch::Tensor forward(torch::Tensor x) {
        x = x + attn(ln1(x));
        x = x + ffn(ln2(x));
        return x;
    }
};
```

**参数量**：
- LN: 2 × 1536 = 3,072
- Attn: 768×2304 + 2304 + 768² + 768 = **2.36M**
- FFN: 768×3072 + 3072 + 3072×768 + 768 = **4.72M**
- Block 总: **7.09M** × 12 层 = **85.1M**

---

## 12 层堆叠

```cpp
struct GPT : torch::nn::Module {
    TokenEmbedding tok_emb{50257, 768};
    PositionalEmbedding pos_emb{1024, 768};
    std::vector<GPTBlock> blocks;
    LayerNorm final_ln{768};
    Linear head{768, 50257};
    
    torch::Tensor forward(torch::Tensor ids) {
        int64_t seq_len = ids.size(1);
        auto x = tok_emb(ids) + pos_emb(arange(seq_len));
        
        for (auto& block : blocks) {
            x = block(x);
        }
        
        x = final_ln(x);
        return head(x);  // [batch, seq, 50257]
    }
};
```

**20 行**，整个 GPT 模型。

---

## 下一步

**4.4 加载 GPT-2 small 权重**：跨语言数值一致

---

## 互动

你读 GPT 代码时最绕的是哪里？
- 残差连接的顺序
- Pre/Post Norm 区别
- FFN 的维度变换
评论区聊聊。
