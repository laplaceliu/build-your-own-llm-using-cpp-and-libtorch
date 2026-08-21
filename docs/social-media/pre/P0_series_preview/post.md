# P0 — 用 C++ 写 LLM 是一种怎样的体验

> 系列开篇｜阅读时长 5 分钟｜预计口播 2 分钟

---

## 钩子

你看到的 LLM 教程，几乎全是 Python。
但你有没有想过：**生产环境里的 LLM，是用什么语言跑的？**
答案：C++。

这次，我想用 C++ 复刻一本经典教材——
**Sebastian Raschka 的《Build a Large Language Model (From Scratch)》**。

---

## 这 10 周要做什么

整个系列一共 **43 帖**，分四个部分：

### 第一部分 · 入门 7 章
1. **Hello LibTorch**（3 帖）— 张量、GPU 切换
2. **处理文本数据**（5 帖）— 分词器、BPE、滑动窗口、词嵌入
3. **编码注意力机制**（4 帖）— QKV、因果 mask、多头、可视化
4. **从头实现 GPT 模型**（5 帖）— 12 层 Transformer Block、加载 GPT-2 权重
5. **预训练**（4 帖）— 损失函数、AdamW、训练循环、加载 OpenAI 权重
6. **分类微调**（3 帖）— 分类头、SMS Spam、GPU 提速实录
7. **指令微调**（4 帖）— 数据格式、GPT-2 medium (355M)、Ollama 评估

### 第二部分 · 附录 3 篇
- **D** 训练循环技巧（warmup + cosine、梯度裁剪、混合精度）
- **E** LoRA 参数高效微调（原理 + 实现 + 对比）
- **F** 推理增强（CoT、Self-Consistency、PRM）

### 第三部分 · 实战 3 个项目
- **gpt-bash**：自然语言 → Bash 命令（12k 样本）
- **gpt-toolcall**：让模型学会调工具（Hermes 16k 样本）
- **gpt-sft**：通用 SFT 框架（CLI + HTTP 服务化）

### 第四部分 · 收官 1 帖
- 全系列复盘 + 下一步路线（Llama / MoE / 量化）

---

## 你能从这个系列里得到什么

### ✅ 跨语言数值一致

同一个随机种子，我用代码验证过：
**C++ 跟 PyTorch 输出完全一致（`max_abs_diff = 0.0`）**。

这意味着你在网上看到的任何 Python 教程，都可以平移到 C++。

```python
# PyTorch
torch.manual_seed(123)
a = torch.rand(2, 3)
```
```cpp
// LibTorch
torch::manual_seed(123);
auto a = torch::rand({2, 3});
// 输出：与 PyTorch 完全一致
```

### ✅ 性能调优的真实数据

| 任务 | CPU | GPU (RTX 4080) | 加速比 |
|---|---|---|---|
| 第 6 章分类微调（5 epoch） | ~16 分钟 | < 1 分钟 | **15x** |
| 第 5 章预训练（10 epoch） | ~8 分钟 | ~12 秒 | **40x** |
| 第 7 章指令微调（355M, 1 epoch） | — | ~75 分钟 | — |

### ✅ 可部署形态

整个项目最终产物是：
- 一个 **C++ 二进制**（约 5 MB）
- 一个 **模型权重文件**（约 1.5 GB）

扔到服务器就能跑，**不需要 Python 运行时、不需要 PyTorch**。

---

## 难度自评

| 前置 | 要求 |
|---|---|
| **C++11** | 懂类、模板、智能指针即可 |
| **线性代数** | 会矩阵乘法、知道 softmax 是啥 |
| **PyTorch**（加分） | 用过更佳，能看懂 Python 教程 |
| **CUDA**（不需要） | LibTorch 已屏蔽 |

---

## 预期产出

| 目标 | 内容 |
|---|---|
| **小目标** | 跟着做跑通 GPT-2 small（124M）全流程 |
| **中目标** | 在指令微调数据集上微调出可用模型 |
| **大目标** | 理解「模型 → 训练 → 部署」全链路 |

---

## 下一条预告

**P1 · 环境搭建：LibTorch + CUDA + CMake 一键跑通**
> 5 分钟搞定环境，包括国内下载怎么突破 10MB/s、踩过的 3 个坑。

---

## 互动

你最想看哪一章？
评论区留言，点赞最高的下一篇优先讲。

---

**项目地址**：`github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch`
**系列目录**：见 `docs/social-media-plan.md`

> 本系列每周一 / 三 / 五 21:00 更新，欢迎追更。
