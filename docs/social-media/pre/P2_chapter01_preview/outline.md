# P2 — 第 1 章导览：30 行 LibTorch 入门

> **本帖定位**：导览帖，3 分钟带你看懂第 1 章做了什么
> **预期时长**：口播 3 分钟（≈ 400 字），正文 1000 字
> **平台配比**：小红书（主）+ 知乎

---

## 钩子

「30 行 C++，跑通 PyTorch 同款张量运算」

---

## 正文大纲

### 1. 第 1 章做了什么（30 秒）
- 6 个张量运算：tensor/zeros/ones/arange/randn/手动 seed
- GPU 切换：`torch::cuda::is_available()`
- 一共约 80 行代码

### 2. 第一个 hello（45 秒）
- `torch::tensor({{1,2},{3,4}})` 创建 2×3 张量
- `.to(torch::kCUDA)` 切 GPU
- `torch::manual_seed(123)` 保证跨语言数值一致

### 3. C++ vs Python 代码对比（45 秒）
- 4 组核心 API 对照
- 唯一区别：C++ 用 `{` `}` 初始化列表 + 类型推断 `auto`

### 4. 下一步预告（30 秒）
- 第 1 章接下来会拆成 3 帖：1.1 张量、1.2 索引、1.3 GPU

---

## 视觉清单

| 编号 | 内容 |
|---|---|
| 1 | 封面：30 行代码截屏 + 终端输出 |
| 2 | 代码截图：`main.cpp` 关键 30 行 |
| 3 | 终端输出：6 个张量 |
| 4 | 对比表：C++ vs Python 4 组 |
| 5 | 路线图：本帖 → 1.1 → 1.2 → 1.3 |

---

## 复用的项目素材

- `chapters/chapter01_hello_torch/main.cpp`
- `docs/modules/chapter01_hello_torch/pages/index.adoc`
