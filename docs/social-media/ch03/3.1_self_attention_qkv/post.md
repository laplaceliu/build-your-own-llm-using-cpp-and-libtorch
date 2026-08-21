# 3.1 自注意力 Q/K/V：三个角度读一句话

> 第 3 章第 1 帖｜阅读 7 分钟｜口播 4 分钟
> 平台：★ 知乎（主）

---

## 钩子

**注意力机制 = 让模型「挑重点看」。**

Q（Query）、K（Key）、V（Value）是三个不同的「视角」。

---

## 一个直觉类比

你在图书馆找一本书：

| 概念 | 类比 | 含义 |
|---|---|---|
| **Query（Q）** | 你想找什么 | 我想看「深度学习」相关 |
| **Key（K）** | 每本书的标签 | 这本书：「Python 入门」 |
| **Value（V）** | 书的内容 | 真正要读的东西 |

**注意力分数** = Q 和 K 的匹配程度（点积）
**输出** = 按注意力分数加权求和 V

---

## 数学定义

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

逐项拆解：
1. `QK^T`：相似度矩阵（n×n）
2. `/ √d_k`：缩放，防止 softmax 饱和（d_k = head 维度）
3. `softmax`：归一化为概率
4. `× V`：加权求和

---

## 从零实现

```cpp
#include <torch/torch.h>

class SelfAttention : public torch::nn::Module {
public:
    SelfAttention(int d_in, int d_out) {
        W_query = register_parameter("W_q",
            torch::randn({d_in, d_out}) * 0.01);
        W_key = register_parameter("W_k",
            torch::randn({d_in, d_out}) * 0.01);
        W_value = register_parameter("W_v",
            torch::randn({d_in, d_out}) * 0.01);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x: [batch, seq_len, d_in]
        auto Q = x @ W_query;   // [batch, seq_len, d_out]
        auto K = x @ W_key;     // [batch, seq_len, d_out]
        auto V = x @ W_value;   // [batch, seq_len, d_out]
        
        // 注意力分数
        auto scores = Q @ K.transpose(-2, -1);              // [batch, seq, seq]
        scores = scores / std::sqrt((double)W_key.size(1)); // 缩放
        
        auto attn = torch::softmax(scores, /*dim=*/-1);    // 归一化
        
        return attn @ V;                                   // [batch, seq, d_out]
    }
    
private:
    torch::Tensor W_query, W_key, W_value;
};
```

---

## 一个具体例子

输入 6 个词，每个 3 维：
```
"The cat sat on the mat"
   ↓ Embedding(3)
[0.1, 0.2, 0.3]   # The
[0.5, 0.1, 0.7]   # cat
[0.2, 0.8, 0.4]   # sat
...
```

Q / K / V 都是 `[batch, 6, 2]`（设 d_out=2）：

| 词 | Q（想找什么） | K（有什么） | V（内容） |
|---|---|---|---|
| The | [0.3, 0.1] | [0.4, 0.2] | [0.5, 0.6] |
| cat | [0.7, 0.5] | [0.6, 0.3] | [0.8, 0.4] |

注意力分数 = Q · K^T：
```
       The  cat  sat  on  the mat
The   0.14 0.20 0.18 ... (Q[The] 和 K[cat] 相似度高)
cat   0.20 0.42 0.30 ...
sat   0.18 0.30 0.28 ...
```

`sat` 的注意力分数显示它最关注 `cat`（0.30）和自身（0.28）——合理！

---

## 为什么缩放因子是 √d_k？

不缩放会出现：
```
d_k = 64 时，Q·K^T 的方差 ≈ 64
softmax([1, 64, 100]) → [0, 0, 1]  ← 几乎 one-hot，梯度消失
```

缩放后：
```
softmax([0.01, 1, 1.5]) → [0.13, 0.36, 0.51]  ← 平滑分布
```

---

## 跨语言对比

```python
# PyTorch
import torch.nn.functional as F
scores = q @ k.T / (d_k ** 0.5)
attn = F.softmax(scores, dim=-1)
return attn @ v
```
```cpp
// LibTorch
auto scores = q @ k.transpose(-2, -1) / std::sqrt((double)d_k);
auto attn = torch::softmax(scores, /*dim=*/-1);
return attn @ v;
```

**逐行对应，0 误差**。

---

## 下一步

**3.2 因果注意力**：GPT 只能看前面的秘密

---

## 互动

你第一次看懂注意力是哪个瞬间？
- 知道 Q/K/V 的类比
- 看到 softmax 那一步
- 自己跑通代码
评论区聊聊。
