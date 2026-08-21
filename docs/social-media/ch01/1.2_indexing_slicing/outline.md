# 1.2 索引 + 切片

> 时长：3 分钟｜平台：★ 知乎

## 要点
1. 三种索引：`x[i]` / `x[i][j]` / `x.index({...})`
2. 切片：slice(start, end, step)
3. 坑：C++ int (32位) ≠ torch 索引用 int64_t (64位)

## 视觉
- 二维张量索引示意（彩色方块）
- 编译错误截图 vs 修复后
- 切片动画分解

## 素材
- `chapters/chapter01_hello_torch/main.cpp` 第 40-80 行
