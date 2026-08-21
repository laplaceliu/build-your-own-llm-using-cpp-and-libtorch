# 社交媒体「图文教程」规划文档

> 项目：`Build Your Own LLM (C++ & LibTorch)`
> 目标：把 `chapters/` + `docs/modules/` + 三个扩展项目（gpt-bash / gpt-sft / gpt-toolcall）
> 的内容，拆成循序渐进、单条 1–5 分钟口播能讲完的图文教程。
> 本文档是**生产蓝图**，每条帖子都可以从这里直接拷贝生成。

---

## 0. 总体规划

### 0.1 定位

- **主题**：用 C++ + LibTorch 从零构建 LLM
- **角度**：和原书（Sebastian Raschka《Build a Large Language Model (From Scratch)》）
  对齐，但**用 C++ 复现**，强调「跨语言数值一致性」、「部署可行性」、「性能调优真实数据」
- **差异化**：市面上几乎全是 Python 教程，C++ 全栈实现是稀缺内容

### 0.2 平台选择（按帖配比）

| 平台 | 适合的帖 | 占比 | 备注 |
|---|---|---|---|
| **知乎专栏** | 长图文（架构篇、原理篇） | ~40% | 适合代码段 + 公式 + 文字 |
| **小红书** | 短图文（对比、可视化、踩坑） | ~25% | 封面图决定点击率 |
| **公众号** | 系列合集 + 长图文 | ~20% | 二次分发到知乎 |
| **B 站 / 视频号** | 「图文动态」或短视频 | ~10% | 同一份素材剪短版 |
| **Twitter / X** | 一图流 + 关键结论 | ~5% | 英文圈引流量 |

> 实际发布时一稿多投：先做 1 套主图 + 文字，按平台裁剪。

### 0.3 单帖形式规范

| 项 | 规范 |
|---|---|
| 口播时长 | **1–5 分钟**（≈ 250–800 字口播脚本） |
| 正文长度 | 800–1500 字（中文） |
| 图片数量 | **4–8 张**（封面 1 + 步骤图 3–6 + 收尾图 1） |
| 封面尺寸 | 1080×1080（小红书/朋友圈）+ 1920×1080（知乎/B 站）二版 |
| 代码截图 | 深色主题、`Fira Code` 字体、行号显示、关键行高亮 |
| 公式 | LaTeX 转 PNG（透明背景，宽度 1200px） |
| 配色 | 项目主色 = `#5B8DEF`（LibTorch 蓝）+ `#FF6B6B`（错误红）+ `#2C3E50`（文字） |

### 0.4 发布节奏建议

- **更新频率**：3–4 帖 / 周（兼顾质量 + 算法推荐）
- **总期数**：约 **43 帖**（详见第 2 节清单）
- **整体周期**：10–12 周（≈ 2.5 个月）跑完一轮
- **节奏类型**：
  - 每周一 / 周三 / 周五 21:00 发布
  - 每章结束做一篇「章节复盘」合集
  - 每两周一次「QA / 踩坑」特辑（评论区问题汇总）

### 0.5 受众画像

| 维度 | 描述 |
|---|---|
| 主受众 | 后端 / 嵌入式 / 性能优化工程师，懂 C++，想了解 LLM 原理 |
| 次受众 | 算法工程师，懂 Python 想拓展 C++ 视角；学生，想做毕业设计 |
| 排除 | 完全不会 C++ 的纯算法选手（前置条件劝退） |
| 痛点 | 看了 Python 教程但不知道生产部署长啥样；想跑通但环境卡三天 |

---

## 1. 系列全景图（章节 → 帖数）

```
Pre 序章（3 帖）
├─ P0 系列预告：用 C++ 写 LLM 是一种怎样的体验
├─ P1 环境搭建：LibTorch + CUDA + CMake 一键跑通
└─ P2 第 1 章导览：30 行 LibTorch 入门

Ch1 Hello LibTorch（3 帖）
Ch2 处理文本数据（5 帖）
Ch3 编码注意力机制（4 帖）
Ch4 从头实现 GPT 模型（5 帖）
Ch5 预训练（4 帖）
Ch6 分类微调（3 帖）
Ch7 指令微调（4 帖）

App D 训练循环技巧（2 帖）
App E LoRA 参数高效微调（3 帖）
App F 推理增强 CoT/SC/PRM（3 帖）

X 扩展项目实战（4 帖）
├─ X1 gpt-bash：自然语言 → Bash
├─ X2 gpt-toolcall：让模型学会调工具
├─ X3 gpt-sft 框架：CLI + HTTP 服务化
└─ X4 总结：从 124M 到 355M，下一步往哪走

总计：3 + 3 + 5 + 4 + 5 + 4 + 3 + 4 + 2 + 3 + 3 + 4 = 43 帖
```

---

## 2. 逐帖详细规划

> **字段说明**
> - **钩子**：开头 1 句话吸引点击（30 字以内）
> - **时长**：建议口播分钟数
> - **平台**：★ 知乎  ★★ 小红书  ★★★ B 站
> - **要点**：3–5 条核心信息点
> - **视觉**：需要的图（代码截图 / 示意图 / 终端输出 / 公式）
> - **素材**：可以直接复用的项目内文件路径

---

### 序章（3 帖）

#### P0 系列预告：用 C++ 写 LLM 是一种怎样的体验
- **时长**：2 分钟
- **平台**：★★ 小红书 + 知乎
- **钩子**：「Python 写 LLM 你见得多了，C++ 写呢？」
- **要点**：
  1. 这条系列要做什么（7 章 + 3 附录 + 3 实战项目）
  2. 为什么选 C++：性能、可部署、跟 PyTorch 同一套 C++ 后端
  3. 难度自评：需 C++11 基础 + 线性代数常识
  4. 预期产出：能跑通 GPT-2 small (124M) 全流程 + 指令微调出可用模型
- **视觉**：
  - 封面：项目结构图（README 里的目录树，配色重制）
  - 截图：`scripts/run.sh chapter07_instruction_tuning` 终端输出
  - 对比图：Python 版教程截图 vs 本项目章节截图
- **素材**：`README.md`、`docs/modules/ROOT/pages/index.adoc`

#### P1 环境搭建：LibTorch + CUDA + CMake 一键跑通
- **时长**：4 分钟
- **平台**：★ 知乎（带完整命令清单）
- **钩子**：「装环境比写代码难？看完这条 5 分钟搞定」
- **要点**：
  1. 三件套：LibTorch 2.13+cu130、CUDA Toolkit 13.0、GCC ≥ 8
  2. 一行命令：`./scripts/setup.sh --all`（下载 + 编译 + 启动 Ollama）
  3. 常见坑：`libre2-dev` 缺失 torchtext BPE 跑不了
  4. 国内下载：用 hf-mirror 镜像，10 MB/s+
- **视觉**：
  - 步骤流程图（4 步带绿色对勾）
  - 终端截图：`./scripts/setup.sh --all` 实时输出
  - 错误对照：缺 `libre2-dev` 的报错截图 vs 修复后
- **素材**：`scripts/setup.sh`、`scripts/download-weights.sh`、`docs/modules/ROOT/pages/prerequisites.adoc`

#### P2 第 1 章导览：30 行 LibTorch 入门
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「30 行 C++，跑通 PyTorch 同款张量运算」
- **要点**：
  1. 第 1 章做了什么：6 个张量运算 + GPU 切换
  2. 第 1 个 hello：`torch::tensor({{1,2},{3,4}})`
  3. `manual_seed`：保证 C++ 和 Python 数值一致
- **视觉**：
  - 代码截图：`chapters/chapter01_hello_torch/main.cpp` 关键 30 行
  - 输出截图：终端打印的 6 个张量
  - 对比：C++ 写法 vs Python 写法（双栏）
- **素材**：`chapters/chapter01_hello_torch/main.cpp`、`docs/modules/chapter01_hello_torch/pages/index.adoc`

---

### 第 1 章：Hello LibTorch（3 帖）

#### 1.1 张量的本质：多维数组 + 自动微分
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. `torch::Tensor` = 数据块 + shape + dtype + device
  2. 创建 6 种方式：`tensor` / `zeros` / `ones` / `arange` / `randn` / `from_blob`
  3. 必备属性：`.sizes()`、`.dtype()`、`.device()`
- **视觉**：
  - 张量内存布局图（3D 张量 = 一维连续内存 + shape）
  - 6 种创建代码截图
- **素材**：`chapters/chapter01_hello_torch/main.cpp` 第 1–40 行

#### 1.2 索引 + 切片：避免 80% 的维度错误
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. `tensor[i]`、`tensor[i][j]`、`tensor.index({...})`
  2. 切片 `[a:b:c]` 与 Python 等价
  3. 坑：C++ 默认 `int` = i32，张量索引用 `int64_t`，类型不匹配直接报错
- **视觉**：
  - 二维张量索引示意（彩色方块）
  - 错误对比截图（左：错误类型，右：修正后）

#### 1.3 GPU 加速：一行代码快 30 倍
- **时长**：2 分钟
- **平台**：★★ 小红书
- **钩子**：「`.to(torch::kCUDA)` 这一行，帮你省 30 倍时间」
- **要点**：
  1. `torch::cuda::is_available()` 自动检测
  2. CPU/GPU 切换只需要 `.to(device)`
  3. 本项目实测：第 5 章预训练 8 分钟 → 12 秒
- **视觉**：
  - 终端截图：`cuda:0` 设备输出
  - 性能对比柱状图（CPU vs GPU 时长）
- **素材**：`README.md`「GPU 支持」段落

---

### 第 2 章：处理文本数据（5 帖）

#### 2.1 文本怎么变数字：从字符到 token
- **时长**：2 分钟
- **平台**：★★ 小红书
- **钩子**：「LLM 不认识汉字，它只认识 0/1」
- **要点**：
  1. 三层映射：字符 → 词元（token）→ ID → 嵌入向量
  2. 为什么要分词：减少词汇表 + 复用常见组合
  3. 本章目标：实现一个 GPT-2 同款 BPE
- **视觉**：四步流程图 + 「Hello World」分词前后对比

#### 2.2 简单分词器：正则 + 词表双向映射
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. `std::regex` 复刻 Python `re.split`
  2. 词汇表 = `std::set`（去重 + 排序）
  3. `encode` / `decode` 双向映射
  4. 特殊词元 `<|endoftext|>` 和 `<|unk|>` 的作用
- **视觉**：
  - 代码截图：`include/simple_tokenizer.h` 关键方法
  - 终端输出：`encode(text) = [1, 56, 2, ...]`
- **素材**：`chapters/chapter02_text_data/include/simple_tokenizer.h`、`docs/modules/chapter02_text_data/pages/index.adoc` 2.2–2.4 节

#### 2.3 BPE：GPT-2 用的算法 5 分钟讲透
- **时长**：5 分钟（重点帖）
- **平台**：★ 知乎
- **要点**：
  1. BPE = 字节级 + 贪心合并 + 频率统计
  2. `bytes_to_unicode`：把 0–255 映射到可打印字符
  3. 单词切分正则：`'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+|...`
  4. 合并循环：找 rank 最小对 → 替换 → 直到无可合并
  5. 练习 2.1：`Akwirw ier` 不需要 `<|unk|>` 也能拆
- **视觉**：
  - 流程图：4 步 BPE 流程
  - 终端输出：练习 2.1 的 6 个 token 拆分
  - 对比：手写版 vs torchtext 官方版（结果一致）
- **素材**：`chapters/chapter02_text_data/src/bpe_tokenizer.cpp`、`include/bpe_tokenizer.h`

#### 2.4 滑动窗口采样：把长文切成训练样本
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 输入块 `[i : i+max_length]` + 目标块右移一位
  2. `stride` 控制窗口步进（重叠 vs 不重叠）
  3. `batch_size=8, max_length=4, stride=4` 第一个 batch 长啥样
- **视觉**：
  - 滑动窗口示意（4 步动画式分解）
  - 代码截图：`GPTDataLoader` 实现
- **素材**：`docs/modules/chapter02_text_data/pages/index.adoc` 2.6 节

#### 2.5 词嵌入 + 位置编码：让模型理解词序
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「同样的词，换个位置意思就变了，模型怎么知道？」
- **要点**：
  1. `nn::Embedding(50257, 256)` 是查找表
  2. 位置嵌入和词嵌入相加 → 输入 shape `[8, 4, 256]`
  3. `manual_seed(123)` 保证 C++ 和 Python 权重数值完全一致
- **视觉**：
  - 形状变化流程图：`[8,4]` → `[8,4,256]`
  - 代码截图：嵌入相加那 3 行
- **素材**：`docs/modules/chapter02_text_data/pages/index.adoc` 2.7–2.8 节

---

### 第 3 章：编码注意力机制（4 帖）

#### 3.1 自注意力 Q/K/V：三个角度读一句话
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. Q = 我要找什么，K = 我有什么，V = 我的内容
  2. `attention(Q, K, V) = softmax(QK^T / √d_k) V`
  3. 从零实现：`nn::Linear` 三次得到 QKV
- **视觉**：
  - 公式图（居中、带 √d_k 缩放解释）
  - 示意：「The cat sat」三个词的 QKV 流向
- **素材**：`chapters/chapter03_attention/include/attention.h` 中的 `SelfAttention`

#### 3.2 因果注意力：GPT 只能看前面的秘密
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「模型偷看答案？一行 mask 解决」
- **要点**：
  1. 因果 mask：上三角置 `-inf`
  2. 为什么要 mask：训练和推理对齐，防止信息泄露
  3. Dropout 在注意力里的作用：防止过拟合当前位置
- **视觉**：
  - 掩码矩阵对比图（左：未掩码，右：掩码后）
  - 代码截图：`CausalAttention` 的 `masked_fill`

#### 3.3 多头注意力：为什么是「多头」而不是单头
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 多头 = 把 QKV 拆成 `num_heads` 份并行算
  2. 每个头关注不同子空间（语法 / 语义 / 长距离）
  3. 拼接 + 线性投影回原维度
- **视觉**：
  - 多头分流的结构图
  - GPT-2 small：12 头 × 64 维 = 768 维（数字拆解）

#### 3.4 注意力可视化：你的模型在盯着哪个词
- **时长**：4 分钟
- **平台**：★★ 小红书 + B 站
- **要点**：
  1. 把 `attention_weights` 画成热力图
  2. 实测：第 1 层关注相邻词；深层关注长距离依赖
  3. C++ 怎么可视化：`Tensor → CSV → Python seaborn`
- **视觉**：
  - 4 张注意力热力图（不同层 / 不同头）
  - 代码截图：导出 CSV 的几行

---

### 第 4 章：从头实现 GPT 模型（5 帖）

#### 4.1 GPT 整体架构：一张图看懂 124M 参数
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. GPT-2 small 配置：`n_layers=12, d_model=768, n_heads=12`
  2. 12 层 Transformer Block 堆叠
  3. 每一层：LayerNorm → 注意力 → 残差 → LayerNorm → FFN → 残差
- **视觉**：
  - 完整架构图（自上而下，分块配色）
  - 参数统计饼图（嵌入 / 注意力 / FFN）

#### 4.2 LayerNorm + GELU：两个「看似简单」的零件
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. LayerNorm vs BatchNorm：按特征归一化，不依赖 batch
  2. GELU vs ReLU：平滑版，平滑处有非零梯度
  3. 为什么 GPT 选这两个：训练稳定性 + 收敛速度
- **视觉**：
  - GELU vs ReLU 激活函数曲线对比图
  - 代码截图：`LayerNorm` 和 `GELU` 实现（各 5–10 行）

#### 4.3 Transformer Block：注意力 + FFN 怎么拼
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. FFN = 2 个线性层 + GELU，维度 768 → 3072 → 768
  2. 残差连接：输入直接加到输出（解决梯度消失）
  3. Pre-Norm vs Post-Norm：GPT 用 Pre-Norm
- **视觉**：
  - Block 内部结构图（清晰标注维度）
  - 代码截图：`TransformerBlock::forward`

#### 4.4 加载 GPT-2 small 权重：跨语言数值一致
- **时长**：4 分钟
- **平台**：★ 知乎（深度技术）
- **要点**：
  1. HuggingFace `safetensors` 文件格式解析
  2. 权重 key 重命名：HF 用 `attn.c_attn.weight` 一卷积，我们要拆 Q/K/V 三矩阵
  3. `tensor.equal()` 验证 C++ 加载结果与官方完全一致
- **视觉**：
  - Key 映射对照表（HF → 本项目）
  - 终端输出：`max_diff = 0.000000e+00` 验证截图

#### 4.5 文本生成：贪心 vs 采样 vs top-k
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「同一个 prompt，三种生成方式差别有多大」
- **要点**：
  1. 贪心解码：每步选概率最大的 token（无聊、循环）
  2. 温度采样：`logits / T`，T 大 = 随机，T 小 = 贪心
  3. Top-k：只在前 k 个 token 里采样（避免低概率长尾）
- **视觉**：
  - 同一 prompt 三种生成结果对比（猫和老鼠梗）
  - 概率分布柱状图（标注 top-k 截断位置）
- **素材**：`docs/modules/chapter05_pretraining/pages/index.adoc`「解码策略」节

---

### 第 5 章：预训练（4 帖）

#### 5.1 损失函数：交叉熵 + 困惑度
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 交叉熵 = `-log(p[target])`
  2. 困惑度 = `e^loss`，越低越好
  3. 文本生成任务的 label = 输入右移一位
- **视觉**：
  - 公式图（交叉熵）
  - 训练曲线截图：loss 从 11 → 7 的下降

#### 5.2 AdamW 优化器：为什么不用 SGD
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. Adam = Momentum + RMSProp
  2. AdamW = Adam + 解耦的权重衰减（L2）
  3. GPT-2 论文配置：`lr=2.5e-4, betas=(0.9, 0.95), eps=1e-8`
- **视觉**：
  - 公式对比表（4 种优化器）
  - 不同学习率训练曲线对比

#### 5.3 训练循环：从 batch 到 GPU 10 行代码
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. 循环 5 步：forward → loss → backward → step → zero_grad
  2. `torch::Tensor::backward()` 自动求导
  3. 实测：10 轮预训练 CPU 8 分钟 vs GPU 12 秒
- **视觉**：
  - 训练循环伪代码
  - 终端输出：`epoch 1/10 loss=11.2 ...`

#### 5.4 加载 OpenAI 权重：跨框架实战
- **时长**：5 分钟（重点帖）
- **平台**：★ 知乎
- **要点**：
  1. 本章 main.cpp 实现了「加载 OpenAI GPT-2 权重 → 推理」全流程
  2. 数值一致性是硬指标：`max_abs_diff < 1e-5`
  3. 部署形态：纯 C++ 推理，不依赖 Python
- **视觉**：
  - 加载流程图（5 步）
  - `max_abs_diff` 验证截图
  - 生成结果对比（同一 prompt，C++ vs HuggingFace Python）

---

### 第 6 章：分类微调（3 帖）

#### 6.1 分类头：把生成模型改成分类器
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 冻结 Transformer 主体，只训练分类头
  2. 取最后一个 token 的隐藏状态作为句子表示
  3. `nn::Linear(d_model, num_classes)`
- **视觉**：
  - 改造前后对比图（生成头 → 分类头）
  - 代码截图：`ClassificationHead`

#### 6.2 SMS Spam 微调：完整流程跑一遍
- **时长**：5 分钟
- **平台**：★ 知乎（实战帖）
- **要点**：
  1. 数据集：5,572 条 SMS（ham/spam 二分类）
  2. 训练配置：5 epoch, lr=5e-5, batch=8
  3. 评估指标：accuracy / precision / recall / F1
  4. 结果：测试集准确率 ~99%
- **视觉**：
  - 数据分布饼图（ham vs spam）
  - 训练 loss 曲线
  - 混淆矩阵

#### 6.3 GPU 提速实录：16 分钟 → 1 分钟
- **时长**：2 分钟
- **平台**：★★ 小红书
- **钩子**：「同一个训练，CPU 和 GPU 差 15 倍」
- **要点**：
  1. CPU 5 轮训练 ~16 分钟
  2. RTX 4080 GPU 5 轮训练 < 1 分钟
  3. 切换只需要 3 行：`device` + `.to()` + tensor 初始化
- **视觉**：
  - 终端 `nvidia-smi` 截图（GPU 利用率 ~95%）
  - 对比柱状图

---

### 第 7 章：指令微调（4 帖）

#### 7.1 指令数据格式：Alpaca / ChatML 怎么选
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. Alpaca 格式：`{instruction, input, output}` + 模板
  2. ChatML：`<|im_start|>system\n...<|im_end|>`
  3. 本项目用 Alpaca 风格（简单 + 通用）
- **视觉**：
  - 两种格式样例对比
  - 模板代码截图：`format_input_alpaca`

#### 7.2 GPT-2 medium 355M：能否在 C++ 跑得动
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「3.55 亿参数的模型，一台游戏本能跑吗？」
- **要点**：
  1. medium 配置：24 层 / 16 头 / 1024 维
  2. 显存：FP32 ~1.4 GB，加载够用
  3. 实测：RTX 4080 微调 1 epoch ≈ 75 分钟
- **视觉**：
  - small vs medium 参数对比表
  - 训练时长 + 显存监控图

#### 7.3 评估方案：Ollama 打分怎么用
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. 用本地 Ollama + llama3 做「裁判模型」
  2. Prompt：让 llama3 给 1–5 分 + 改进建议
  3. 评估 100 条测试集，对比微调前后
- **视觉**：
  - 评估 Prompt 截图
  - 分数分布对比图

#### 7.4 指令微调前后对比：同一个 prompt 两种回答
- **时长**：3 分钟
- **平台**：★★ 小红书 + B 站
- **钩子**：「同一个问题，微调前 vs 微调后，差距有多大」
- **要点**：
  1. 5 组对比：写作文 / 改语病 / 翻译 / 摘要 / 知识问答
  2. 同一 prompt，base model 输出跑题，SFT 后输出结构化
  3. 失败案例：模型仍然会胡编（提示局限性）
- **视觉**：
  - 5 组对话截图（左：base，右：SFT）
  - 失败案例标注

---

### 附录 D：训练循环技巧（2 帖）

#### D.1 学习率调度：Warmup + Cosine
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. Warmup：前 N 步 lr 从 0 线性升到峰值（防早期震荡）
  2. Cosine 衰减：余弦曲线平滑下降
  3. GPT-2 论文：warmup=2000 steps + min_lr=10% peak
- **视觉**：
  - 学习率曲线图（warmup + cosine 完整周期）
  - 不同策略对比曲线

#### D.2 梯度裁剪 + 混合精度
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 梯度裁剪：`grad.clamp_(-1, 1)` 防梯度爆炸
  2. 混合精度：FP32 主权重 + FP16 计算 + 损失缩放
  3. LibTorch 调用：`torch::autocast` + `GradScaler`
- **视觉**：
  - 公式图：梯度裁剪 + 损失缩放
  - 终端输出：AMP 启用前后显存对比

---

### 附录 E：LoRA 参数高效微调（3 帖）

#### E.1 LoRA 原理：低秩分解为什么有效
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. 核心假设：权重更新矩阵 ΔW 是低秩的
  2. ΔW = A × B，A: d×r，B: r×d，r << d
  3. 原权重冻结，只训练 A、B（参数量降到 ~1%）
- **视觉**：
  - 低秩分解示意图（彩色矩阵块）
  - 参数量对比表（small / medium 全量 vs LoRA）

#### E.2 LoRA 实现：冻结原参数 + 注入小矩阵
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. 自定义 `LoRALinear` 包装 `nn::Linear`
  2. forward: `base_out + x @ A @ B * (alpha / r)`
  3. `requires_grad_(false)` 冻结 base
- **视觉**：
  - 代码截图：`LoRALinear` 关键 20 行
  - 训练参数量统计：`trainable params: 0.5M / all params: 355M (0.14%)`

#### E.3 LoRA vs 全量微调：显存与效果对比
- **时长**：3 分钟
- **平台**：★★ 小红书
- **要点**：
  1. 显存：全量微调 1.4 GB → LoRA 0.2 GB（优化器状态也省）
  2. 效果：loss 曲线高度接近，差距 < 2%
  3. 适用场景：显存紧张 + 多任务切换 + 个人开发者
- **视觉**：
  - 双柱对比图（显存 + 效果）
  - 终端：`nvidia-smi` 显存占用对比

---

### 附录 F：推理增强（3 帖）

#### F.1 Chain-of-Thought：让模型「自言自语」
- **时长**：3 分钟
- **平台**：★★ 小红书
- **钩子**：「让模型一步步想，数学题正确率翻倍」
- **要点**：
  1. CoT 提示词：`Let's think step by step`
  2. 思维链长度 vs 准确率的 trade-off
  3. 适用：算术 / 推理 / 多步规划；不适用：简单事实问答
- **视觉**：
  - 同一数学题，无 CoT（左，答错）vs 有 CoT（右，答对）
  - 准确率柱状图（GSM8K 上 18% → 57%）

#### F.2 Self-Consistency：多次采样 + 投票
- **时长**：3 分钟
- **平台**：★ 知乎
- **要点**：
  1. 思路：采样 N 条推理路径，少数服从多数
  2. N = 5–10 性价比最高
  3. 实测：GSM8K 准确率再涨 5–10%
- **视觉**：
  - 5 条路径 + 投票结果示意图
  - N vs 准确率曲线

#### F.3 过程奖励模型 PRM：把推理拆开打分
- **时长**：5 分钟
- **平台**：★ 知乎（高阶）
- **要点**：
  1. ORM（结果奖励）vs PRM（过程奖励）：PRM 给每一步打分
  2. PRM 训练数据：人标注每一步对错
  3. 推理时：beam search 选 PRM 总分最高的路径
- **视觉**：
  - PRM 评分流程图
  - ORM vs PRM 准确率对比

---

### 扩展项目实战（4 帖）

#### X1 gpt-bash：自然语言 → Bash 命令
- **时长**：4 分钟
- **平台**：★ 知乎 + ★★ 小红书
- **要点**：
  1. 数据：nl2bash 12k 样本（自然语言 ↔ Bash）
  2. 训练：复用 gpt-sft 训练框架，5 epoch small
  3. 沙箱 `--exec`：白名单命令白名单目录，防止危险
  4. 效果示例：「列出最大的 5 个文件」→ `du -ah . | sort -rh | head -5`
- **视觉**：
  - 对话框截图（自然语言 → Bash 输出）
  - 沙箱配置文件截图
- **素材**：`gpt-bash/README.md`、`gpt-bash/chat/`

#### X2 gpt-toolcall：让模型学会调工具
- **时长**：5 分钟
- **平台**：★ 知乎 + ★★★ B 站
- **要点**：
  1. 数据：Hermes-function-calling-v1（16k+ 样本）
  2. 特殊词元：`<|tool_call|>{...}</|tool_call|>` 包裹 JSON
  3. Python dispatcher 多轮对话循环（C++ 只做单轮推理）
  4. 5 个工具实测：calculator / weather / shell / search / datetime
- **视觉**：
  - 多轮对话截图（user → tool_call → tool result → assistant）
  - 工具注册代码截图
- **素材**：`gpt-toolcall/README.md`、`gpt-toolcall/chat/dispatcher.py`

#### X3 gpt-sft 框架：CLI + HTTP 服务化
- **时长**：4 分钟
- **平台**：★ 知乎
- **要点**：
  1. core 静态库：模型工厂 / 训练 / 推理统一 API
  2. `sft_train` CLI：`--data --epochs --lr --out`
  3. `sft_serve` HTTP：`POST /v1/chat`，OpenAI 兼容协议
  4. 部署形态：单二进制 + 模型文件 + 端口开放
- **视觉**：
  - 架构图：core / train / serve 三模块
  - `curl` 调用截图：`/health` 和 `/v1/chat`
- **素材**：`gpt-sft/README.md`、`gpt-sft/serve/src/sft_serve.cpp`

#### X4 总结：从 124M 到 355M，下一步往哪走
- **时长**：3 分钟
- **平台**：★ 知乎（系列收官）
- **要点**：
  1. 全系列回顾：7 章 + 3 附录 + 3 实战 = 43 帖
  2. 关键技术：跨语言数值一致 + GPU 加速 + LoRA 节省显存
  3. 后续路线：Llama 架构 / MoE / 量化（INT8/INT4）/ vLLM 风格推理
  4. 互动：评论区征集下一系列主题
- **视觉**：
  - 全系列脑图（一图汇总）
  - 「下一步」路线图

---

## 3. 生产规范（每帖通用）

### 3.1 视觉规范

| 元素 | 规范 |
|---|---|
| 主色 | `#5B8DEF`（LibTorch 蓝） |
| 辅色 | `#FF6B6B`（错误红）/ `#52C41A`（成功绿）/ `#FAAD14`（提示黄） |
| 文字色 | `#2C3E50`（深灰）/ `#7F8C8D`（浅灰） |
| 字体 | 中文：思源黑体 / 英文：JetBrains Mono / 代码：Fira Code |
| 封面构图 | 左：标题（大字） + 右：核心图（代码 / 示意） |
| 视觉一致性 | 所有架构图用同一种配色 + 字体 + 圆角（4px） |

### 3.2 代码截图规范

```cpp
// 截图模板示意
torch::Tensor x = torch::randn({2, 3});
std::cout << x << std::endl;
//                  ^^^ 关键行用红色高亮
```

- 深色主题（One Dark / Dracula）
- 行号显示，关键行加红色边框或黄色高亮
- 配套文字解释：每段代码下 1–2 行说明

### 3.3 文字脚本结构

```
[钩子]（30 字以内，吸引点击）
[背景]（50–80 字，为何重要）
[正文]（600–1200 字，分 3–5 个小节，每节配图）
[小结]（50–80 字，本帖要点回顾）
[预告]（30 字以内，下帖内容）
```

### 3.4 标题模板

- **教程类**：「[章号]. [主题]：[价值主张]」
  - 例：`2.3 BPE 分词：GPT-2 用的算法 5 分钟讲透`
- **对比类**：「[A] vs [B]：[场景]怎么选」
  - 例：`LoRA vs 全量微调：显存紧张时该用哪个`
- **踩坑类**：「[问题]？[解决方案]」
  - 例：`torchtext BPE 跑不起来？装一个包就行`

---

## 4. 发布节奏详细表

| 周次 | 发布帖 | 平台 |
|---|---|---|
| W01 | P0 / P1 / P2 | 知乎 + 小红书 |
| W02 | 1.1 / 1.2 / 1.3 | 知乎 × 2 + 小红书 |
| W03 | 2.1 / 2.2 / 2.3 / 2.4 | 知乎 × 3 + 小红书 |
| W04 | 2.5 / 3.1 / 3.2 / 3.3 | 知乎 × 3 + 小红书 |
| W05 | 3.4 / 4.1 / 4.2 / 4.3 | 知乎 × 3 + B 站 |
| W06 | 4.4 / 4.5 / 5.1 / 5.2 | 知乎 × 4 |
| W07 | 5.3 / 5.4 / 6.1 / 6.2 | 知乎 × 4 |
| W08 | 6.3 / 7.1 / 7.2 / 7.3 | 知乎 × 3 + 小红书 |
| W09 | 7.4 / D.1 / D.2 / E.1 | 知乎 × 3 + 小红书 |
| W10 | E.2 / E.3 / F.1 / F.2 | 知乎 × 3 + 小红书 |
| W11 | F.3 / X1 / X2 / X3 | 知乎 × 3 + B 站 |
| W12 | X4（收官） | 知乎 + 全平台 |

---

## 5. 复用素材清单（按帖索引）

> 直接打开这些文件就能取材：

```
chapters/chapter01_hello_torch/main.cpp                # 1.1, 1.2, 1.3, P2
chapters/chapter02_text_data/
  ├── include/simple_tokenizer.h                       # 2.2
  ├── include/bpe_tokenizer.h                          # 2.3
  ├── src/bpe_tokenizer.cpp                            # 2.3
  └── third_party/                                     # 2.3
chapters/chapter03_attention/
  ├── include/attention.h                              # 3.1, 3.2, 3.3
  └── src/main.cpp                                     # 3.4
chapters/chapter04_gpt/
  ├── include/gpt.h                                    # 4.1, 4.2, 4.3
  └── src/main.cpp                                     # 4.4, 4.5
chapters/chapter05_pretraining/
  ├── include/training.h                               # 5.1, 5.2, 5.3
  ├── include/safetensors.h                            # 5.4
  └── src/main.cpp                                     # 5.4
chapters/chapter06_finetuning/
  ├── include/finetuning.h                             # 6.1, 6.2
  └── src/main.cpp                                     # 6.3
chapters/chapter07_instruction_tuning/
  ├── include/instruction.h                            # 7.1
  └── src/main.cpp                                     # 7.2, 7.3, 7.4
chapters/chapterD_training_loop/                       # D.1, D.2
chapters/chapterE_lora/                                # E.1, E.2, E.3
chapters/chapterF_reasoning/                           # F.1, F.2, F.3
gpt-bash/                                              # X1
gpt-toolcall/                                          # X2
gpt-sft/                                               # X3
README.md                                              # P0, GPU 加速
scripts/setup.sh                                       # P1
```

---

## 6. 跟踪与优化

### 6.1 数据指标（每周追踪）

| 指标 | 目标 |
|---|---|
| 平均阅读量 | 单帖 5k+（小红书）/ 2k+（知乎） |
| 收藏率 | > 8%（教程类核心指标） |
| 评论质量 | 每帖 ≥ 5 条技术提问，月底汇总做 QA 特辑 |
| 涨粉 | 全系列跑完 +1k 知乎粉丝 / +500 小红书粉丝 |

### 6.2 内容复盘节点

- **W03 末**：第 1–2 章复盘，调整后续节奏
- **W06 末**：第 3–5 章复盘，决定是否压缩 LoRA 篇
- **W09 末**：附录篇复盘，确认扩展项目是否单列
- **W12 末**：全系列收官，写复盘文章 + 准备下一系列

### 6.3 风险与应对

| 风险 | 应对 |
|---|---|
| 部分章节内容单薄（如 D / F） | 合并帖，或引入外部资料（论文 / 官方教程）做对照 |
| GPU 实测数据变化 | 用历史截图 + 当前实测双图对比，标注时间 |
| 评论区技术问题答不上 | 沉淀到项目 `docs/qa.md`，做成 FAQ 特辑 |
| 算法推荐不给量 | 启动二创：小红书做「金句卡图」、B 站做「3 分钟速览」 |

---

## 7. 下一步行动清单

> 用户确认本规划后，可以并行做的事：

- [ ] **建文件夹**：`docs/social-media/{pre,ch01..ch07,D,E,F,X}/`，每帖一目录
- [ ] **建素材库**：`docs/social-media/assets/{code,diagram,formula,screenshot}/`
- [ ] **画首图**：P0 封面 + 项目结构图（建议 Figma / Excalidraw）
- [ ] **拍代码**：用 `carbon.now.sh` 把第 1 章 30 行做成首张代码图
- [ ] **录屏工具**：准备 `asciinema` 录终端，`OBS` 录 VS Code
- [ ] **建 README**：每帖目录下放 `script.md`（口播脚本）+ `outline.md`（大纲）
- [ ] **建跟踪表**：用飞书 / Notion 做发布看板

---

> 本文档持续维护：每发完一帖，回填实际表现（阅读量 / 收藏率 / 评论高频问题）。
> 下次更新：W04（W3 末复盘后）。
