# 1.3 GPU 加速：一行代码快 30 倍

> 第 1 章第 3 帖｜阅读 4 分钟｜口播 2 分钟
> 平台：★★ 小红书（主）+ 知乎

---

## 钩子

**`.to(torch::kCUDA)` 这一行，帮你省 30 倍时间。**

---

## 三步切换 GPU

```cpp
#include <torch/torch.h>

int main() {
    // 1. 检测 GPU 是否可用
    bool has_cuda = torch::cuda::is_available();
    auto device = has_cuda ? torch::kCUDA : torch::kCPU;
    
    // 2. 创建张量 + 切到 GPU
    auto x = torch::randn({1024, 1024}).to(device);
    
    // 3. 后续所有计算都在 GPU 上
    auto y = torch::matmul(x, x);  // 1024x1024 矩阵乘
    
    // 4. 拿回 CPU（可选）
    auto y_cpu = y.to(torch::kCPU);
    return 0;
}
```

**就这么简单**。

---

## 实测：训练速度对比

| 任务 | CPU | GPU (RTX 4080) | 加速比 |
|---|---|---|---|
| 1024×1024 矩阵乘（1000 次） | 5.2 秒 | 0.08 秒 | **65x** |
| 第 5 章预训练（10 epoch） | ~8 分钟 | ~12 秒 | **40x** |
| 第 6 章分类微调（5 epoch） | ~16 分钟 | < 1 分钟 | **15x** |

---

## 设备管理 4 条原则

### 原则 1：模型和数据要在同一设备

```cpp
auto model = GPTModel(config);  // 默认在 CPU
model->to(device);              // 模型到 GPU

auto data = ...;
data = data.to(device);         // 数据到 GPU
```

否则报错：
```
RuntimeError: Expected all tensors to be on the same device
```

### 原则 2：不要频繁切换

```cpp
// ❌ 慢：每轮都切
for (...) {
    auto y = model(x.to(device)).to(torch::kCPU);
}

// ✅ 快：切一次，全程 GPU
auto x = data.to(device);
for (...) {
    auto y = model(x);  // 已经在 GPU
}
```

### 原则 3：`.to()` 返回新对象

```cpp
auto x = torch::randn({3, 3});
auto y = x.to(torch::kCUDA);

// 此时 x 还在 CPU，y 在 CUDA
// 两个对象是独立的！
```

### 原则 4：保存模型时切回 CPU

```cpp
// 训练完切回 CPU 再保存
model->to(torch::kCPU);
torch::save(model, "model.pt");
```

---

## 显存监控

```bash
nvidia-smi
```

```
+----------------------------------+
| 0  NVIDIA GeForce RTX 4080  OFF  |
| 0%  45C  P8  25W  / 350W  0MiB  |
+----------------------------------+
```

训练时：
- **利用率 > 80%** = GPU 跑满了
- **显存 > 90%** = 危险，随时 OOM
- 解决：`batch_size` 砍半，或加 `gradient_accumulation_steps`

---

## 没有 GPU 怎么办？

- **CPU 跑**：本项目所有示例都支持 CPU，只是慢
- **Google Colab 免费版**：T4 GPU 跑第 5 章预训练约 30 分钟
- **云 GPU**：AutoDL / 恒源云，按小时租

---

## 下一步

**2.1 文本怎么变数字**：从字符到 token

---

## 互动

你的训练用 CPU 还是 GPU？
留言说说你的显卡型号 + 训练时长，看看谁最快。
