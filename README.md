# Build Your Own LLM (C++ & LibTorch)

手把手从零构建大语言模型（LLM）的教程仓库，代码使用 **C++** 与 **LibTorch**（PyTorch C++ 前端），文档使用 **Antora** 按章节独立编排。

## 仓库结构

```
.
├── CMakeLists.txt            # 顶层 CMake：一键编译全部章节
├── cmake/
│   └── libtorch.cmake        # 定位 LibTorch + add_libtorch_executable 帮助函数
├── scripts/                  # 统一准备/构建/运行脚本（见下）
│   ├── setup.sh              # 一键准备：下载数据/权重 + Ollama + 编译
│   ├── download-data.sh      # 下载各章数据集（词表/文本/Spam/指令数据）
│   ├── download-weights.sh   # 下载 GPT-2 small/medium 权重
│   ├── start-ollama.sh       # 启动 Ollama 服务 + 拉取 llama3
│   ├── build.sh              # 配置并编译全部章节
│   ├── run.sh                # 设置环境并运行指定章节
│   └── common.sh             # 公共环境变量与下载函数（source 用）
├── chapters/                 # C++ 工程（每章一个独立子工程）
│   ├── CMakeLists.txt
│   ├── chapter01_hello_torch/
│   │   ├── CMakeLists.txt    # 每章可脱离顶层独立构建
│   │   └── main.cpp
│   ├── chapter02_text_data/
│   │   ├── CMakeLists.txt
│   │   ├── include/          # simple_tokenizer.h / bpe_tokenizer.h / torchtext_gpt2_bpe.h
│   │   ├── src/              # main.cpp / bpe_tokenizer.cpp / torchtext_gpt2_bpe.cpp
│   │   ├── third_party/      # torchtext 官方 GPT2BPEEncoder 源码（pytorch/text）
│   │   └── data/             # the-verdict.txt + GPT-2 词表
│   ├── chapter03_attention/
│   │   ├── CMakeLists.txt
│   │   ├── include/          # attention.h（SelfAttention/CausalAttention/MultiHeadAttention）
│   │   └── src/              # main.cpp
│   ├── chapter04_gpt/
│   │   ├── CMakeLists.txt
│   │   ├── include/          # gpt.h（LayerNorm/GELU/FeedForward/TransformerBlock/GPTModel）
│   │   └── src/              # main.cpp（依赖第 2/3 章 tokenizer 与注意力）
│   ├── chapter05_pretraining/
│   │   ├── CMakeLists.txt
│   │   ├── include/          # training.h / dataloader.h / safetensors.h
│   │   └── src/              # main.cpp（预训练 + 解码策略 + OpenAI 权重加载）
│   ├── chapter06_finetuning/
│   │   ├── CMakeLists.txt
│   │   ├── include/          # finetuning.h（分类头/评估/训练）
│   │   ├── src/              # main.cpp（垃圾消息分类微调，GPU 加速）
│   │   └── data/             # SMS Spam 数据集（首次运行需下载）
│   └── chapter07_instruction_tuning/
│       ├── CMakeLists.txt
│       ├── include/          # instruction.h（format_input/collate/加载器）
│       ├── src/              # main.cpp（GPT-2 medium 指令微调 + Ollama 评估）
│       └── data/             # instruction-data.json + gpt2-medium.safetensors
└── docs/                     # Antora 文档项目（playbook + 组件 + 模块）
    ├── playbook.yml          # 站点 playbook
    ├── antora.yml            # 组件描述
    ├── ui/                   # 本地缓存的默认 UI bundle
    └── modules/
        ├── ROOT/             # 首页、环境准备、总导航
        ├── chapter01_hello_torch/   # 按章命名 module
        ├── chapter02_text_data/
        ├── chapter03_attention/
        ├── chapter04_gpt/
        ├── chapter05_pretraining/
        ├── chapter06_finetuning/
        └── chapter07_instruction_tuning/
```

## 环境要求

| 组件 | 版本 |
|------|------|
| LibTorch（CUDA 版） | 2.13.0+cu130（解压到 `/opt/libtorch`） |
| CUDA Toolkit | 13.0 |
| 编译器 | GCC ≥ 8（需支持 C++17，建议 11） |
| CMake | ≥ 3.16 |
| Antora | 3.x（node ≥ 18） |
| RE2（`libre2-dev`） | torchtext 官方 BPE 分词器的正则依赖 |

LibTorch 查找顺序：`-DTORCH_ROOT=<dir>` > `$TORCH_ROOT` > 默认 `/opt/libtorch`。

> 第 2 章的 torchtext 官方 `GPT2BPEEncoder` 依赖 RE2：
> `sudo apt-get install libre2-dev`

**GPU 支持**：所有章节自动检测 CUDA（`torch::cuda::is_available()`），可用时在 GPU
上运行（如 RTX 4080）。权重始终在 CPU 上 `manual_seed` 初始化（保证与书数值一致）
后再迁移 GPU，并禁用 TF32 保持 matmul 精度。GPU 加速效果：第 5 章 10 轮预训练
CPU 约 8 分钟 → GPU 约 10-15 秒（~30 倍）；第 6 章 5 轮微调 CPU 约 16 分钟 →
GPU 不到 1 分钟（~15-20 倍）。

## 快速上手（统一脚本）

所有准备/构建/运行操作统一由 `scripts/` 下的脚本完成：

```bash
# 一键准备：下载数据集 + 下载权重 + 启动 Ollama + 编译
./scripts/setup.sh --all

# 或按需分步
./scripts/download-data.sh          # 各章数据集（幂等，已存在则跳过）
./scripts/download-weights.sh       # GPT-2 small/medium 权重（约 2 GB）
./scripts/start-ollama.sh --pull    # 启动 Ollama + 拉取 llama3（7.8 用）
./scripts/build.sh                  # 配置并编译全部章节

# 运行指定章节（自动设置 LD_LIBRARY_PATH / CUDA 环境）
./scripts/run.sh chapter01_hello_torch
./scripts/run.sh chapter02_text_data
./scripts/run.sh chapter03_attention
./scripts/run.sh chapter04_gpt
./scripts/run.sh chapter05_pretraining      # 可传 epochs：run.sh chapter05_pretraining <data_dir> 10
./scripts/run.sh chapter06_finetuning
./scripts/run.sh chapter07_instruction_tuning
```

> 提示：`scripts/common.sh` 集中管理 `LIBTORCH_ROOT`、`CUDA_VERSION` 等路径
> （默认 `/opt/libtorch` + CUDA 13.0，可按机器修改）。下载优先直连
> （hf-mirror / github raw），失败自动回退代理。

## 构建文档站点

```bash
cd docs
antora playbook.yml          # 产物输出到 docs/public/
python3 -m http.server 8080 -d public   # 本地预览
```

UI bundle 已随仓库缓存于 `docs/ui/ui-bundle.zip`；如需更新，可重新下载：

```bash
curl -fsSL -o docs/ui/ui-bundle.zip \
  "https://gitlab.com/antora/antora-ui-default/-/jobs/artifacts/master/raw/build/ui-bundle.zip?job=bundle-stable"
```

## 如何新增一章

1. 代码：`chapters/chapter02_xxx/`（CMakeLists + main.cpp），并在 `chapters/CMakeLists.txt` 追加 `add_subdirectory(chapter02_xxx)`。
2. 文档：`docs/modules/chapter02_xxx/pages/index.adoc` + `nav.adoc`，并在 `docs/modules/ROOT/nav.adoc` 与首页目录中加入链接。
3. 提交后运行 `antora playbook.yml` 验证站点。

## 章节计划

- [x] 第 1 章 Hello LibTorch：张量基础
- [x] 第 2 章 处理文本数据（分词 / BPE / 滑动窗口 / 嵌入）
- [x] 第 3 章 编码注意力机制（自注意力 / 因果注意力 / 多头注意力）
- [x] 第 4 章 从头实现 GPT 模型（1.24 亿参数 GPT-2 small + 文本生成）
- [x] 第 5 章 预训练（损失评估 / AdamW 训练 / 温度与 Top-k 解码 / 加载 OpenAI 权重）
- [x] 第 6 章 分类微调（SMS Spam 二分类，支持 GPU 加速）
- [x] 第 7 章 指令微调（GPT-2 medium 355M + Alpaca 风格指令微调 + Ollama 评估）
- [ ] 更多敬请期待……
