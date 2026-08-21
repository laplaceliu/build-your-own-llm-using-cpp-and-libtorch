# 1.1 张量的本质：多维数组 + 自动微分

> 第 1 章第 1 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎 + ★★ 小红书

---

## 钩子

你以为 `torch::Tensor` 是个张量？
**它其实是「数据块 + 元信息的组合体」**。

---

## 张量的 5 个组成

一个 Tensor 包含：

| 字段 | 含义 | 例子 |
|---|---|---|
| `data` | 一块连续内存（`float*`） | `[1.0, 2.0, 3.0, 4.0]` |
| `sizes` | 形状 | `[2, 2]` |
| `dtype` | 数据类型 | `kFloat32` |
| `device` | 设备 | `kCPU` / `kCUDA` |
| `requires_grad` | 是否求导 | `true` / `false` |

**关键**：数据在内存里是**一维连续**的，「多维」只是解释方式。

---

## 创建 6 种方式

```cpp
#include <torch/torch.h>

int main() {
    // 1. 直接给值
    auto a = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    // → [[1, 2], [3, 4]], shape=[2,2]
    
    // 2. 全零
    auto b = torch::zeros({3, 5});
    // → shape=[3,5], 全 0
    
    // 3. 全一
    auto c = torch::ones({2, 2});
    
    // 4. 序列
    auto d = torch::arange(0, 10, 2);
    // → [0, 2, 4, 6, 8]
    
    // 5. 随机（正态分布）
    auto e = torch::randn({2, 3});
    // → shape=[2,3], N(0,1)
    
    // 6. 从内存直接构造
    float data[] = {1, 2, 3, 4};
    auto f = torch::from_blob(data, {2, 2});
    // ⚠️ 注意：f 不拥有 data 的所有权
}
```

---

## 必备属性

```cpp
auto x = torch::randn({3, 4, 5});

x.sizes()      // → [3, 4, 5]
x.dtype()      // → kFloat32
x.device()     // → cpu
x.numel()      // → 60 (元素总数)
x.dim()        // → 3 (维度数)
```

---

## 一个常见误解

`torch::Tensor` 不是 `std::vector<std::vector<float>>`。
它的内存是**连续**的，多维形状只是 stride 解析。

```cpp
auto x = torch::randn({2, 3});
// 内存布局：[x[0,0], x[0,1], x[0,2], x[1,0], x[1,1], x[1,2]]
//        ↑ row-major (C 风格)
```

所以：
- `x[0][0]` 实际是「先跳到第 0 行，再跳到第 0 列」
- **不是**真的二维数组

---

## 下一步

**1.2 索引 + 切片**：避免 80% 的维度错误

---

## 互动

你之前踩过张量形状的什么坑？
`dim=3` 的张量你熟吗？评论区聊聊。
