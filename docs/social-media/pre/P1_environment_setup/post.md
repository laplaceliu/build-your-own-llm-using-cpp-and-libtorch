# P1 — LibTorch + CUDA + CMake 一键跑通

> 系列第 2 篇｜环境搭建｜阅读时长 6 分钟｜口播 4 分钟

---

## 钩子

装环境比写代码难？
看完这条，**5 分钟搞定 LibTorch + CUDA**。

---

## 为什么 LibTorch 比 PyTorch 难装？

PyTorch 一行 `pip install torch` 就完事。
但 LibTorch 是 C++ 库，要手动配 3 件事：

1. **LibTorch 库文件**（约 2 GB）
2. **CMake 找包配置**（`TorchConfig.cmake`）
3. **CUDA 环境变量**（`LD_LIBRARY_PATH`）

再加国内下载 HuggingFace 权重慢，整个流程能卡你 2–3 天。
这条帖子，一次性帮你解决。

---

## 三件套清单

| 组件 | 版本 | 来源 |
|---|---|---|
| **LibTorch** | 2.13+cu130 | pytorch.org（推荐用 hf-mirror） |
| **CUDA Toolkit** | 12.x / 13.0 | nvidia.com |
| **GCC** | ≥ 8（推荐 11） | apt install g++ |

**一键脚本**：
```bash
git clone https://github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch
cd build-your-own-llm-using-cpp-and-libtorch
./scripts/setup.sh --all
```

---

## 国内下载：hf-mirror 镜像方案

| 镜像 | 平均速度 |
|---|---|
| HuggingFace 直连 | ~100 KB/s |
| **hf-mirror.com** | **10 MB/s+** |

```bash
# 默认走 hf-mirror
HF_MIRROR=hf-mirror ./scripts/download-weights.sh gpt2-medium

# 切回官方（海外用户）
HF_MIRROR=huggingface ./scripts/download-weights.sh gpt2-medium
```

下载脚本内置：
- `-C -` 断点续传
- `--retry 5 --retry-all-errors`
- `wc -c` 校验文件大小
- `.part` 后缀保留便于重跑

---

## 3 个常见坑

### 坑 1：`libre2-dev` 缺失 → BPE 跑不了

```
error: tokenizers/cpu_tokenizer.o: undefined reference to `re2::...`
```

**修复**：
```bash
sudo apt install libre2-dev libre2-utils
```

### 坑 2：链接报 `undefined symbol`

```bash
./chapter01: symbol lookup error: ... undefined symbol: _ZN5torch...
```

**原因**：`LD_LIBRARY_PATH` 残留了旧版本 LibTorch
**修复**：
```bash
unset LD_LIBRARY_PATH
./scripts/run.sh chapter01_hello_torch
```

### 坑 3：ARM 交叉编译串了宿主 stdlib

```bash
fatal error: x86intrin.h: No such file
```

**原因**：编译期 `CPLUS_INCLUDE_PATH` 没清理
**修复**：
```bash
unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH CPATH LIBRARY_PATH CC CXX
```

---

## 验证

跑通后应该看到：

```bash
$ ./scripts/run.sh chapter01_hello_torch
[Hello LibTorch] Created 2x3 tensor:
 0.1596  0.0464  0.0702
 0.0206  0.6146  0.6040
[CPU backend]
cuda available? true / false
...
```

6 个张量都打印出来 = 环境 OK。

---

## 下一条

**P2 · 第 1 章导览：30 行 LibTorch 入门**
> 6 个张量运算一行行拆开讲。

---

## 互动

你装 LibTorch 卡在哪一步？
评论区贴报错信息，下一条帮你汇总答疑。

---

**项目地址**：`github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch`
