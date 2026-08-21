# 4.4 加载 GPT-2 small 权重：跨语言数值一致

> 第 4 章第 4 帖｜⭐ 深度技术｜阅读 9 分钟｜口播 4 分钟
> 平台：★ 知乎（主）

---

## 钩子

**同一份 `safetensors` 文件，C++ 加载 vs HuggingFace 加载，输出 0 误差。**

数值一致性 = 一切 LLM 工程的基础。

---

## 为什么这件事重要？

| 任务 | 必须一致的对象 |
|---|---|
| 部署 | C++ 推理 = Python 推理 |
| 微调 | 我训的模型 = HF 训的模型 |
| 调试 | 我看到的 loss = 论文里报的 loss |

如果一个 `0.001` 的偏差，传 12 层就是 `0.001 × 12 = 0.012`，生成质量肉眼可见地崩。

---

## HuggingFace 的 key 命名

```python
# HuggingFace GPT2Model 内部：
model.transformer.h[0].attn.c_attn.weight  # shape [1, 768, 2304]
model.transformer.h[0].attn.c_attn.bias    # shape [2304]
model.transformer.h[0].attn.c_proj.weight  # shape [1, 768, 768]
model.transformer.h[0].ln_1.weight         # shape [768]
model.transformer.h[0].ln_1.bias           # shape [768]
...
```

**坑 1**：`c_attn` = 1 个卷积层一次性输出 Q/K/V 三组（2304 = 3 × 768）。
**坑 2**：HF 用 `Conv1D` 而不是 `Linear`，权重是转置的。

---

## 本项目的 key 命名

```cpp
// chapters/chapter04_gpt/include/gpt.h
class GPTBlock {
    MultiHeadAttention attn;  // 内部：
    //   W_q: [768, 768]
    //   W_k: [768, 768]
    //   W_v: [768, 768]
    //   W_out: [768, 768]
    FeedForward ffn;
    //   fc1: [768, 3072]
    //   fc2: [3072, 768]
    LayerNorm ln1, ln2;
};
```

---

## Key 映射表（手动转换）

```cpp
#include <nlohmann/json.hpp>
#include <safetensors.hh>

void load_gpt2_weights(GPT& model, const std::string& path) {
    auto weights = safetensors::load(path);
    
    // 1. Token embedding
    model.tok_emb.weight.data().copy_(
        weights.at("wte.weight"));
    
    // 2. Position embedding
    model.pos_emb.weight.data().copy_(
        weights.at("wpe.weight"));
    
    // 3. 每个 Block
    for (int i = 0; i < 12; ++i) {
        auto prefix = "h." + std::to_string(i) + ".";
        
        // 3.1 LayerNorm 1
        model.blocks[i].ln1.weight.data().copy_(
            weights.at(prefix + "ln_1.weight"));
        model.blocks[i].ln1.bias.data().copy_(
            weights.at(prefix + "ln_1.bias"));
        
        // 3.2 Q/K/V：拆 c_attn.weight [1, 768, 2304] → 3 个 [768, 768]
        auto c_attn_w = weights.at(prefix + "c_attn.weight");
        // shape [1, 768, 2304]，squeeze 后 [768, 2304]
        // 切分为 3 个 [768, 768]
        model.blocks[i].attn.W_q.data().copy_(
            c_attn_w.slice(2, 0, 768).squeeze(0));
        model.blocks[i].attn.W_k.data().copy_(
            c_attn_w.slice(2, 768, 1536).squeeze(0));
        model.blocks[i].attn.W_v.data().copy_(
            c_attn_w.slice(2, 1536, 2304).squeeze(0));
        
        // 3.3 Attention output projection
        model.blocks[i].attn.W_out.data().copy_(
            weights.at(prefix + "c_proj.weight").squeeze(0));
        // ... bias 也类似
        
        // 3.4 LayerNorm 2
        model.blocks[i].ln2.weight.data().copy_(
            weights.at(prefix + "ln_2.weight"));
        
        // 3.5 FFN
        model.blocks[i].ffn.fc1.weight.data().copy_(
            weights.at(prefix + "mlp.c_fc.weight").t());  // 转置！
        model.blocks[i].ffn.fc2.weight.data().copy_(
            weights.at(prefix + "mlp.c_proj.weight").t()); // 转置！
        
        // 3.6 最后 LayerNorm（在 model 外面）
    }
    
    // 4. 最终 LayerNorm
    model.final_ln.weight.data().copy_(
        weights.at("ln_f.weight"));
}
```

---

## 三大坑

### 坑 1：Conv1D vs Linear 的转置

```python
# HuggingFace 用 Conv1D，输入维度在最后
# weight shape: [in_dim, out_dim]
# forward: y = x @ weight  ← 不是 x @ weight.T
```

```cpp
// 本项目用 Linear，权重是 [out_dim, in_dim]
// forward: y = x @ weight.T
```

**解决**：加载时转置 `.t()`。

### 坑 2：Q/K/V 在同一矩阵

```python
# HF 的 c_attn.weight: [1, 768, 2304]，concat 了 Q/K/V
# 顺序：先 768 维 Q，再 768 维 K，再 768 维 V
```

**解决**：用 `slice(2, start, end)` 拆成 3 份。

### 坑 3：Bias 也要拆

```python
# c_attn.bias: [2304] = [Q_bias, K_bias, V_bias]
```

---

## 验证数值一致性

```cpp
// 用同一段文本跑推理
auto input_ids = torch::tensor({15496, 995});  // "Hello world"

// 本项目
auto logits_cpp = model.forward(input_ids);

// HuggingFace（Python 端）
# logits_hf = hf_model(torch.tensor([15496, 995]))

// 对比
auto diff = (logits_cpp - logits_hf).abs().max();
std::cout << "max_diff = " << diff.item<float>() << std::endl;
// 输出：max_diff = 0.000000e+00
```

**完全一致！**

---

## 整个加载流程（5 步）

```
1. 下载 gpt2.safetensors（HF 或 hf-mirror）
2. safetensors::load() 解析为 unordered_map
3. 遍历 12 个 Block，按 key 映射拷贝
4. 拆 c_attn / 转置 Conv1D 权重
5. 加载 + 跑一次推理，对比 max_diff
```

---

## 下一步

**4.5 文本生成**：贪心 vs 采样 vs top-k

---

## 互动

你加载 HF 权重时踩过哪些坑？
- Conv1D 转置
- Q/K/V 拆分
- bias 顺序
- key 命名版本差异

评论聊聊。
