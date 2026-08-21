# 3.2 因果注意力：GPT 只能看前面的秘密

> 第 3 章第 2 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★★ 小红书（主）+ 知乎

---

## 钩子

**模型偷看答案？一行 mask 解决。**

GPT 是「因果」的：第 i 个位置的预测，只能依赖前 i 个位置，不能看未来。

---

## 为什么需要 mask？

训练时输入是一个完整句子：
```
["我", "爱", "吃", "苹", "果"]
```

如果不做 mask，模型在预测「果」时能看到「苹果」——**直接抄答案**。
训练和推理就对不上了。

---

## 怎么做 mask？上三角置 -inf

```cpp
auto mask = torch::triu(torch::ones({seq_len, seq_len}), /*diagonal=*/1);
// 上三角（不含对角线）= 1，下三角（含对角线）= 0

auto scores = q @ k.transpose(-2, -1) / std::sqrt(d_k);
scores = scores.masked_fill(mask == 1, -1e9);  // 上三角置 -inf
auto attn = torch::softmax(scores, -1);
// 上三角的 softmax = 0，真正实现「看不到未来」
```

**可视化**：
```
mask = [[0, 1, 1, 1],
        [0, 0, 1, 1],    ← 下三角（含对角线）= 0
        [0, 0, 0, 1],       「自己能看自己，能看过去」
        [0, 0, 0, 0]]       上三角 = 1
                          「看不到未来」
```

softmax 前：
```
scores = [[s00, -inf, -inf, -inf],    ← 第 1 个 token 只看自己
          [s10,  s11, -inf, -inf],    ← 第 2 个 token 看 [0,1]
          [s20,  s21,  s22, -inf],
          [s30,  s31,  s32,  s33]]
```

softmax 后上三角都是 0，attention 输出等于只看前 i 个 token。

---

## Dropout 在注意力里的作用

```cpp
auto dropout = torch::nn::Dropout(0.1);
auto attn = dropout(torch::softmax(scores, -1));
```

**作用**：训练时随机「关掉」10% 的注意力连接。
**为什么有效**：防止模型过度依赖某一个特定位置。

类比：学生答题时不能只盯着某一个字，要学会从多个线索推断。

---

## 完整 CausalAttention 实现

```cpp
class CausalAttention : public torch::nn::Module {
public:
    CausalAttention(int d_in, int d_out,
                    int context_length, float dropout)
        : dropout_(dropout) {
        W_q = register_parameter("W_q",
            torch::randn({d_in, d_out}) * 0.01);
        W_k = register_parameter("W_k",
            torch::randn({d_in, d_out}) * 0.01);
        W_v = register_parameter("W_v",
            torch::randn({d_in, d_out}) * 0.01);
        
        mask = register_buffer("mask",
            torch::triu(torch::ones({context_length, context_length}), 1));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto Q = x @ W_q;
        auto K = x @ W_k;
        auto V = x @ W_v;
        
        auto scores = Q @ K.transpose(-2, -1) / std::sqrt((double)W_k.size(1));
        scores = scores.masked_fill(mask == 1, -1e9);
        
        auto attn = torch::softmax(scores, -1);
        attn = dropout_(attn);
        
        return attn @ V;
    }
    
private:
    torch::Tensor W_q, W_k, W_v, mask;
    torch::nn::Dropout dropout_{nullptr};
};
```

---

## 三步记忆

1. **Mask**：上三角置 -inf
2. **Softmax**：自动让 -inf 变 0
3. **Dropout**：再随机砍 10%

---

## 下一步

**3.3 多头注意力**：为什么是「多头」而不是单头

---

## 互动

你训练时 `dropout` 设多少？
- 0.0（不下，太少数据除外）
- 0.1（标准）
- 0.2–0.3（数据少时）
评论区告诉我。
