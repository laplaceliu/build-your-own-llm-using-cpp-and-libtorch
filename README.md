# Build Your Own LLM (C++ & LibTorch)

手把手从零构建大语言模型（LLM）的教程仓库，代码使用 **C++** 与 **LibTorch**（PyTorch C++ 前端），文档使用 **Antora** 按章节独立编排。

## 仓库结构

```
.
├── CMakeLists.txt            # 顶层 CMake：一键编译全部章节
├── cmake/
│   └── libtorch.cmake        # 定位 LibTorch + add_libtorch_executable 帮助函数
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
│   └── chapter04_gpt/
│       ├── CMakeLists.txt
│       ├── include/          # gpt.h（LayerNorm/GELU/FeedForward/TransformerBlock/GPTModel）
│       └── src/              # main.cpp（依赖第 2/3 章 tokenizer 与注意力）
└── docs/                     # Antora 文档项目（playbook + 组件 + 模块）
    ├── playbook.yml          # 站点 playbook
    ├── antora.yml            # 组件描述
    ├── ui/                   # 本地缓存的默认 UI bundle
    └── modules/
        ├── ROOT/             # 首页、环境准备、总导航
        ├── chapter01_hello_torch/   # 按章命名 module
        ├── chapter02_text_data/
        ├── chapter03_attention/
        └── chapter04_gpt/
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

## 编译并运行 C++ 章节

```bash
# 配置（g++ 默认版本过旧时显式指定）
cmake -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

# 编译全部章节
cmake --build build -j$(nproc)

# 运行第 1 章（需把 libtorch / CUDA 运行库加入 LD_LIBRARY_PATH）
LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/cuda-13.0/lib64 \
  ./build/chapters/chapter01_hello_torch/chapter01_hello_torch
```

> 提示：配置阶段需要 CUDA 工具链（因为链接了 CUDA 版 libtorch），
> 可先 `export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0`。
> 另外，若 shell 中残留了本项目以外的 `LD_LIBRARY_PATH`，建议先 `unset LD_LIBRARY_PATH` 再编译。

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
- [ ] 第 5 章 预训练
- [ ] 第 6 章 分类微调
- [ ] 第 7 章 指令微调
- [ ] 更多敬请期待……
