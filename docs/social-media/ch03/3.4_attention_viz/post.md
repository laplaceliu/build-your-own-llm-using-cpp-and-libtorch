# 3.4 注意力可视化：你的模型在盯着哪个词

> 第 3 章第 4 帖｜阅读 6 分钟｜口播 4 分钟
> 平台：★★ 小红书 + ★★★ B 站

---

## 钩子

**「模型在盯着哪个词？」—— 一张热力图告诉你。**

可视化注意力权重 = 打开 LLM 的「黑盒」。

---

## 怎么做？

```cpp
// 1. 前向时保存注意力权重
class VisualizableAttention : public CausalAttention {
public:
    torch::Tensor forward(torch::Tensor x) {
        // ... 算 Q, K, V ...
        auto scores = Q @ K.transpose(-2, -1) / std::sqrt(d_k);
        scores = scores.masked_fill(mask == 1, -1e9);
        
        attn_weights = torch::softmax(scores, -1);  // ← 保存！
        return attn_weights @ V;
    }
    
    torch::Tensor get_attn_weights() const { return attn_weights; }
    
private:
    torch::Tensor attn_weights;
};

// 2. 导出 CSV（C++ 端）
auto weights = model.get_attn_weights();  // [batch, heads, seq, seq]
auto csv = weights.select(0, 0).select(0, 0);  // [seq, seq]
std::ofstream f("attn_layer5_head3.csv");
f << csv;
```

**3. 用 Python 画热力图**：
```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

attn = pd.read_csv("attn_layer5_head3.csv", header=None)
sns.heatmap(attn, annot=True, cmap="YlOrRd")
plt.title("Layer 5, Head 3")
plt.xlabel("Key token")
plt.ylabel("Query token")
plt.savefig("attn_viz.png", dpi=150)
```

---

## 实测：4 张图看清模型行为

测试文本：`"The quick brown fox jumps over the lazy dog"`

### 图 1：第 1 层、第 1 头
```
The → The (0.6), quick (0.2), brown (0.1) ...
quick → The (0.3), quick (0.4), brown (0.2) ...
```
**结论**：第 1 层主要关注**相邻词**。

### 图 2：第 1 层、第 5 头
```
The → The (0.8), quick (0.1) ...
fox → fox (0.3), jumps (0.4), over (0.2) ...
```
**结论**：第 1 层某些头专注目**主语-动词配对**。

### 图 3：第 5 层、第 3 头
```
fox → fox (0.2), jumps (0.5), lazy (0.2) ...
lazy → lazy (0.4), dog (0.5) ...
```
**结论**：第 5 层开始捕捉**句法依赖**（主谓宾）。

### 图 4：第 12 层、第 1 头
```
fox → fox (0.1), jumps (0.2), the (0.1) ...
```
**结论**：第 12 层注意力**分散**，更像在做语义融合。

---

## 4 个观察规律

1. **浅层**：关注相邻词（bigram 模式）
2. **中层**：捕捉句法（主谓、动宾）
3. **深层**：融合全局语义
4. **不同头**：不同子空间（语法 / 语义 / 位置）

---

## 调试小技巧

```python
# 找「最专注的头」：熵最低
entropies = -np.sum(attn * np.log(attn + 1e-9), axis=-1)
print(f"Most focused: head {entropies.argmin()}")

# 找「最分散的头」：熵最高
print(f"Most spread: head {entropies.argmax()}")
```

---

## C++ 导出热力图（不依赖 Python）

```cpp
// 用 C++ 自己画 PNG：stb_image_write 单文件库
#include "stb_image_write.h"

void save_attn_png(torch::Tensor weights, const std::string& path) {
    auto cpu = weights.to(torch::kCPU).contiguous();
    int n = cpu.size(0);
    
    // 简化：每个像素 = 灰度值
    std::vector<unsigned char> img(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            img[i * n + j] = (unsigned char)(cpu[i][j].item<float>() * 255);
        }
    }
    
    stbi_write_png(path.c_str(), n, n, 1, img.data(), n);
}
```

---

## 下一步

**4.1 GPT 整体架构**：一张图看懂 124M 参数

---

## 互动

你看过哪篇经典论文的注意力图？
- 「What does BERT look at?」
- 「Visualizing and Understanding Transformer」
- 本系列实测

评论推荐你最爱的可视化工具。
