# 6.3 GPU 提速实录：16 分钟 → 1 分钟

> 第 6 章第 3 帖｜阅读 3 分钟｜口播 2 分钟
> 平台：★★ 小红书（主）

---

## 钩子

**同一个训练，CPU 和 GPU 差 15 倍。**

---

## 实测数据

测试任务：SMS Spam 微调，5 epoch

| 设备 | 时长 | 加速比 |
|---|---|---|
| **CPU (i7-12700)** | ~16 分钟 | 1x |
| **GPU (RTX 4080)** | < 1 分钟 | **15x** |
| GPU (RTX 3060) | ~2 分钟 | 8x |
| GPU (RTX 4090) | ~40 秒 | 24x |

---

## 切换只需要 3 行

```cpp
// 1. 检测设备
auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

// 2. 模型到 GPU
model->to(device);

// 3. 数据到 GPU（每个 batch）
auto ids = torch::tensor(...).to(device);
```

---

## 详细对比

```bash
# CPU 模式
$ ./build/chapter06_finetuning --device cpu --epochs 5
[06_finetuning] Starting on cpu
epoch 0 loss=0.42 acc=87.2%  [3m12s]
epoch 1 loss=0.18 acc=95.4%  [3m08s]
...
Total: 15m 50s

# GPU 模式
$ ./build/chapter06_finetuning --device cuda --epochs 5
[06_finetuning] Starting on cuda:0
epoch 0 loss=0.42 acc=87.2%  [11.2s]
epoch 1 loss=0.18 acc=95.4%  [11.0s]
...
Total: 56s
```

**loss 曲线完全一致**，只是快。

---

## nvidia-smi 监控

训练时观察：
```
+----------------------------------+
| 0  NVIDIA GeForce RTX 4080       |
| 97%  78C  P2  285W  / 350W       |
|        982MiB / 16384MiB         |
+----------------------------------+
```

- **GPU 利用率 97%** ← 跑满了
- **温度 78°C** ← 健康（>85°C 警惕）
- **显存 982 MB** ← 124M 模型足够

---

## 切换前后对比图

```
柱状图：
       CPU 16m     GPU 56s
        ████       ▌
        ████       ▌
        ████       ▌
        ████       ▌
```

---

## 没有 GPU 怎么办？

| 方案 | 适用 |
|---|---|
| **Google Colab 免费版** | T4 GPU，30 分钟够用 |
| **Kaggle Notebooks** | 免费 P100，每周 30 小时 |
| **AutoDL / 恒源云** | 1–2 元 / 小时 |
| **CPU 跑** | 本项目所有示例都支持 |

---

## 一个常见误区

```cpp
// ❌ 错误：data.to(device) 放循环外面
auto data = load_all_data();
data.to(device);  // 把所有数据放 GPU，OOM 风险

// ✅ 正确：每个 batch 单独切
for (auto& batch : dataloader) {
    auto x = batch.x.to(device);  // 只把当前 batch 放 GPU
    auto y = batch.y.to(device);
    // ...
}
```

---

## 下一步

**7.1 指令数据格式**：Alpaca / ChatML 怎么选

---

## 互动

你的训练用什么 GPU？
- RTX 3060（8x）
- RTX 4080（15x）
- RTX 4090（24x）
- A100（40x+）

评论区告诉我你的显卡 + 训练时长。
