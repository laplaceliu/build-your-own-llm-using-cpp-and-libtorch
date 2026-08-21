# E.2 LoRA 实现：冻结原参数 + 注入小矩阵

> 附录 E 第 2 帖｜阅读 5 分钟｜口播 4 分钟
> 平台：★ 知乎

---

## 钩子

**手写一个 LoRA 包装器，20 行代码搞定。**

---

## LoRALinear 实现

```cpp
class LoRALinear : public torch::nn::Module {
public:
    int r;           // 秩
    float alpha;     // 缩放系数
    
    torch::nn::Linear base;   // 原线性层（冻结）
    torch::Tensor A, B;       // LoRA 矩阵
    
    LoRALinear(int in_features, int out_features, int rank = 8)
        : r(rank), alpha(2.0f * rank) {
        
        // 原权重（冻结）
        base = register_module("base",
            torch::nn::Linear(in_features, out_features));
        for (auto& p : base->parameters()) {
            p.set_requires_grad(false);  // ← 关键
        }
        
        // LoRA 矩阵
        A = register_parameter("A",
            torch::randn({in_features, r}) * 0.01);
        B = register_parameter("B",
            torch::zeros({r, out_features}));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // 原路径
        auto base_out = base->forward(x);
        
        // LoRA 路径
        auto lora_out = (x @ A) @ B;
        
        return base_out + lora_out * (alpha / r);
    }
    
    // 推理时合并权重，节省内存
    torch::nn::Linear merge() {
        auto merged_weight = base->weight + (A @ B).T() * (alpha / r);
        auto merged_bias = base->bias;
        
        auto result = torch::nn::Linear(merged_weight.size(1),
                                         merged_weight.size(0));
        result->weight.data().copy_(merged_weight);
        result->bias.data().copy_(merged_bias);
        return result;
    }
};
```

---

## 在 GPT Block 中替换

```cpp
struct GPTBlockWithLoRA : torch::nn::Module {
    LoRALinear attn_W_q{768, 768, /*r=*/8};
    LoRALinear attn_W_v{768, 768, /*r=*/8};
    LoRALinear ffn_fc1{768, 3072, /*r=*/8};
    // ... 其他层保持不变 ...
    
    torch::Tensor forward(torch::Tensor x) {
        // ... 用 attn_W_q 替代原 W_q ...
    }
};
```

**只对 W_q、W_v 做 LoRA**——这是 LoRA 论文的标配。

---

## 训练参数量对比

```cpp
// 计算可训练参数
int64_t count_trainable(GPTBlockWithLoRA& block) {
    int64_t n = 0;
    for (auto& p : block.parameters()) {
        if (p.requires_grad()) {
            n += p.numel();
        }
    }
    return n;
}

// GPT-2 medium 全 Block（24 个）
// 全量微调：355M
// LoRA (r=8, W_q/W_v)：~3M
```

输出：
```
trainable params: 3.0M
all params: 355M
trainable%: 0.85%
```

**0.85% 的参数量达到 ≈ 全量微调 99% 的效果**。

---

## 训练循环（和全量微调一样）

```cpp
auto optimizer = AdamW(
    get_trainable_params(model),  // 只包含 A、B 参数
    1e-4                          // LoRA 用更大学习率
);
```

**关键**：优化器只收到 LoRA 的参数，所以 AdamW 状态也很小。

---

## 多任务切换示例

```cpp
// 训练任务 A
auto model_A = load_base_gpt2();
add_lora_to_model(model_A, rank=8);
train(model_A, task_A_data);
save_lora_weights(model_A, "task_A.bin");

// 训练任务 B
auto model_B = load_base_gpt2();
add_lora_to_model(model_B, rank=8);
train(model_B, task_B_data);
save_lora_weights(model_B, "task_B.bin");

// 推理时切换
auto model = load_base_gpt2();
if (task == "A") {
    load_lora_weights(model, "task_A.bin");
} else {
    load_lora_weights(model, "task_B.bin");
}
```

**优势**：base 模型只存一份（1.43 GB），不同 LoRA 各存几十 MB。

---

## 训练配置

| 参数 | 全量微调 | LoRA |
|---|---|---|
| `lr` | 5e-5 | 1e-4（更大） |
| `epochs` | 2-3 | 5-10（更多） |
| `rank` | — | 4 / 8 / 16 |
| `alpha` | — | r 或 2r |

---

## 下一步

**E.3 LoRA vs 全量微调**：显存与效果对比

---

## 互动

你用 LoRA 的目标层是？
- W_q, W_v（论文标配）
- 所有线性层（激进）
- 只 FFN（折中）

评论区告诉我。
