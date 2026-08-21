# P1 — 环境搭建：LibTorch + CUDA + CMake 一键跑通

> **本帖定位**：5 分钟跑通完整环境，附国内下载方案和踩坑清单
> **预期时长**：口播 4 分钟（≈ 600 字），正文 1500 字
> **平台配比**：知乎（主，带完整命令清单）+ B 站（视频版）

---

## 钩子（30 字以内）

1. 「装环境比写代码难？看完这条 5 分钟搞定」
2. 「LibTorch + CUDA，国内下载也能 10 MB/s+」
3. 「跑了 3 天装环境？这条帮你省 2 天半」

**封面建议**：用 4 步流程图作背景（绿对勾）+ 标语「5 分钟搞定 LibTorch 环境」。

---

## 正文大纲

### 1. 为什么 LibTorch 比 PyTorch 难装（30 秒）
- PyTorch `pip install` 一行，LibTorch 要手动配置 5 件事
- 三件套：LibTorch 库、CMake 模块查找、CUDA 环境变量
- 国内下载：huggingface 直连慢，huggingface 走镜像

### 2. 三件套清单（45 秒）
- **LibTorch**：本项目用 `2.13+cu130`
- **CUDA Toolkit**：12.x 或 13.0
- **GCC**：≥ 8（推荐 11）
- 一个一键脚本：`./scripts/setup.sh --all`

### 3. 国内下载：hf-mirror 镜像方案（45 秒）
- HuggingFace 直连平均 100 KB/s
- hf-mirror 镜像稳定 10 MB/s+
- 用法：`HF_MIRROR=hf-mirror` 环境变量切换
- 下载脚本：`scripts/download-weights.sh` 已内置断点续传

### 4. 3 个常见坑（60 秒）
- **坑 1**：`libre2-dev` 缺失 → BPE 跑不了 → `apt install libre2-dev`
- **坑 2**：`libtorch` 链接报 `undefined symbol` → 检查 `LD_LIBRARY_PATH`
- **坑 3**：ARM 交叉编译串了宿主 stdlib → 用 `unset CPLUS_INCLUDE_PATH`

### 5. 验证 + 下一步（30 秒）
- 跑 `./scripts/run.sh chapter01_hello_torch` 验证
- 看到 6 个张量输出 = 环境 OK
- 下一条：第 1 章 Hello LibTorch 入门

---

## 视觉清单

| 编号 | 内容 | 类型 | 制作建议 |
|---|---|---|---|
| 1 | 封面 | 设计图 | 4 步流程 + 标语 |
| 2 | 三件套清单 | 信息图 | 三个软件包 + 版本号 |
| 3 | 终端输出 | 截图 | `setup.sh --all` 完整过程 |
| 4 | hf-mirror 速度对比 | 数据图 | 100 KB/s vs 10 MB/s 柱状图 |
| 5 | 3 个坑的截图 | 截图 | 报错信息 vs 修复后 |
| 6 | 验证输出 | 截图 | chapter01 终端输出 |

---

## 复用的项目素材

- `scripts/setup.sh`
- `scripts/download-weights.sh`
- `docs/modules/ROOT/pages/prerequisites.adoc`
- `chapters/chapter01_hello_torch/main.cpp`

---

## 标题候选

| 平台 | 标题 |
|---|---|
| 知乎 | 《LibTorch + CUDA 环境搭建：5 分钟一键跑通（附国内下载方案）》 |
| B 站 | 《5 分钟搞定 C++ LLM 开发环境》 |
| 小红书 | 「装了 3 天 LibTorch？这条帮你省 2 天半」 |
