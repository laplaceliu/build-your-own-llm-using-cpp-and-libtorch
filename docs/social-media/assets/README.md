# 素材库（assets）

> 所有图文的图片 / 代码截图 / 信息图都集中在这里管理。
> 每张图的命名规范：`{post_id}_{seq}_{type}.{ext}`
> 例：`P0_02_cover.png` = P0 第 2 张图，封面，PNG 格式。

---

## 目录说明

| 子目录 | 内容 | 文件类型 |
|---|---|---|
| `cover/` | 封面图（每帖 2 版：1080×1080 + 1920×1080） | PNG / JPG |
| `code/` | 代码截图（深色主题，关键行高亮） | PNG |
| `diagram/` | 信息图、架构图、流程图 | PNG / SVG |
| `formula/` | 公式（LaTeX 转 PNG，透明背景） | PNG |
| `screenshot/` | 终端输出、IDE 截图、`nvidia-smi` 截图 | PNG |

---

## 制作工具约定

| 工具 | 用途 |
|---|---|
| **carbon.now.sh** | 代码截图（首选，支持高亮行） |
| **Excalidraw** | 架构图、流程图（手绘风格，易读） |
| **Figma** | 封面、信息图（精细控制） |
| **LaTeX + MathJax** | 公式截图 |
| **asciinema** | 终端输出录制（可重放） |
| **OBS Studio** | IDE / 桌面录制 |

---

## 配色规范

| 用途 | 色值 |
|---|---|
| 主色（LibTorch 蓝） | `#5B8DEF` |
| 错误红 | `#FF6B6B` |
| 成功绿 | `#52C41A` |
| 提示黄 | `#FAAD14` |
| 深灰文字 | `#2C3E50` |
| 浅灰文字 | `#7F8C8D` |
| 背景白 | `#FFFFFF` |
| 背景灰 | `#F5F7FA` |

---

## 字体规范

| 场景 | 字体 |
|---|---|
| 中文正文 | 思源黑体（Source Han Sans） |
| 英文正文 | Inter / Helvetica |
| 代码 | Fira Code / JetBrains Mono |

---

## 当前素材清单

| 帖 ID | 素材名 | 状态 | 路径 |
|---|---|---|---|
| P0 | 项目结构图 | ⚪ 待制作 | `diagram/P0_01_project_structure.png` |
| P0 | 43 帖路线图 | ⚪ 待制作 | `diagram/P0_02_series_roadmap.png` |
| P0 | PyTorch vs LibTorch × 5 | ⚪ 待制作 | `code/P0_03_*_compare.png` |
| P0 | 终端输出截图 | ⚪ 待制作 | `screenshot/P0_04_terminal.png` |
| P0 | CPU/GPU 柱状图 | ⚪ 待制作 | `diagram/P0_05_cpu_vs_gpu.png` |
| P0 | 封面图 1080 | ⚪ 待制作 | `cover/P0_cover_1080.png` |
| P0 | 封面图 1920 | ⚪ 待制作 | `cover/P0_cover_1920.png` |
