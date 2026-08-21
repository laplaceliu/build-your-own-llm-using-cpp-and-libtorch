# 2.5 词嵌入 + 位置编码：让模型理解词序

> 第 2 章第 5 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★★ 小红书 + 知乎

---

## 钩子

**同样的词，换个位置意思就变了，模型怎么知道？**

答案：加一个「位置向量」。

---

## 词嵌入：token → 向量

```cpp
torch::nn::Embedding tok_emb{50257, 768};
// 参数：50257 × 768 = 38.6M
// 每个 token 对应一行 768 维向量

auto token_ids = torch::tensor({15496, 11, 995, 0});  // 4 个 token
auto embeddings = tok_emb(token_ids);
// shape: [4, 768]
```

**本质**：一个 `50257 × 768` 的查找表。
- 词表越大（50257 个）→ 嵌入层参数越多
- 维度越高（768）→ 表达力越强，参数越多

---

## 位置编码：让模型知道第几位

```cpp
torch::nn::Embedding pos_emb{1024, 768};
// 参数：1024 × 768 = 0.79M
// 1024 = 最大序列长度

auto positions = torch::arange(0, 4);  // [0, 1, 2, 3]
auto pos_vectors = pos_emb(positions);
// shape: [4, 768]
```

**位置 0、1、2、3 各对应一个独特的向量**。

---

## 两者相加

```cpp
auto input_embeddings = tok_emb(token_ids) + pos_emb(positions);
// shape: [4, 768]
// 第 i 个位置的「输入」= 第 i 个 token 的词向量 + 第 i 个位置的位置向量
```

**关键**：
- 词嵌入告诉模型「这个词是什么意思」
- 位置嵌入告诉模型「这个词在哪儿」
- 两者相加 → 「这个词在第 i 位是什么意思」

---

## 形状变化流程

```
输入 token IDs
  shape: [4]
       ↓ Embedding(50257, 768)
  shape: [4, 768]
       ↓
  + Positional Embedding(1024, 768)
       ↓
  shape: [4, 768]   ← 喂给 Transformer Block
```

加上 batch 维度：
```
shape: [batch_size, seq_len, d_model]
     = [8, 4, 768]
```

---

## 为什么不用 sinusoidal？

GPT 选用**可学习的位置嵌入**，而不是 Transformer 原论文的 `sin/cos`。

| 方案 | 优点 | 缺点 |
|---|---|---|
| **可学习** (GPT) | 简单，模型自动学 | 长度固定（1024） |
| **sinusoidal** (原 Transformer) | 可外推到任意长度 | 表达力有限 |
| **RoPE** (LLaMA) | 长度外推好 | 实现复杂 |
| **ALiBi** (BLOOM) | 加 bias 即可 | 性能略差 |

本项目实现 GPT 风格：**可学习的位置嵌入**。

---

## 跨语言数值一致小实验

```cpp
torch::manual_seed(123);
auto emb = torch::nn::Embedding(10, 3);
auto v = emb(torch::tensor({0}));
// PyTorch: tensor([[ 0.3374, -0.1778, -0.3035]])
// LibTorch: 完全一致
```

**同一份 `manual_seed(123)`，两边输出 100% 一致**。
这是本系列所有实验可复现的基石。

---

## 下一步

**3.1 自注意力 Q/K/V**：三个角度读一句话

---

## 互动

你用 LLaMA 还是 GPT 风格的位置编码？
评论说说你的选择原因。
