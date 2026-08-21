# P2 — 第 1 章导览：30 行 LibTorch 入门

> 系列第 3 篇｜第 1 章导览｜阅读 4 分钟｜口播 3 分钟

---

## 钩子

**30 行 C++，跑通 PyTorch 同款张量运算。**

这是这个系列的第一站——
第 1 章 Hello LibTorch。

---

## 第 1 章做了什么？

整个第 1 章只有 **80 行代码**，干了两件事：

1. **6 个张量运算**：创建、初始化、随机化
2. **GPU 切换**：`torch::cuda::is_available()`

跑完你会看到终端打印 6 个张量，从 1×3 的小向量到 2×3 的随机矩阵。

---

## 第一个 hello

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // 1. 创建一个张量
    auto x = torch::tensor({{1.0, 2.0, 3.0},
                            {4.0, 5.0, 6.0}});
    
    // 2. 切到 GPU
    if (torch::cuda::is_available()) {
        x = x.to(torch::kCUDA);
        std::cout << "Running on GPU!" << std::endl;
    }
    
    // 3. 打印
    std::cout << x << std::endl;
    return 0;
}
```

就这么 14 行，你已经跑通了：
- ✅ 张量创建
- ✅ 设备检测
- ✅ CPU/GPU 切换
- ✅ 打印输出

---

## C++ vs Python 代码对比

| 操作 | Python (PyTorch) | C++ (LibTorch) |
|---|---|---|
| 创建张量 | `torch.tensor([[1, 2], [3, 4]])` | `torch::tensor({{1, 2}, {3, 4}})` |
| 全零张量 | `torch.zeros(2, 3)` | `torch::zeros({2, 3})` |
| 随机张量 | `torch.rand(2, 3)` | `torch::rand({2, 3})` |
| 切 GPU | `x.cuda()` | `x.to(torch::kCUDA)` |
| 打印 | `print(x)` | `std::cout << x << std::endl;` |
| 随机种子 | `torch.manual_seed(123)` | `torch::manual_seed(123);` |

**唯一区别**：C++ 用 `{}` 初始化列表 + `auto` 类型推断。

**最重要的是**：同一个 `manual_seed(123)`，两边输出**完全一致**。

---

## 接下来 3 帖

| 帖 | 内容 | 时长 |
|---|---|---|
| **1.1** | 张量的本质：多维数组 + 自动微分 | 3 分钟 |
| **1.2** | 索引 + 切片：避免 80% 的维度错误 | 3 分钟 |
| **1.3** | GPU 加速：一行代码快 30 倍 | 2 分钟 |

---

## 互动

你以前用过 PyTorch 吗？
有没有觉得 C++ 版比 Python 版「反直觉」的地方？
评论区聊聊。

---

**项目地址**：`github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch`
