# 4.1 GPT 整体架构

> 时长：4 分钟｜平台：★ 知乎

## 钩子
GPT-2 small 124M 参数到底是怎么分布的？

## 要点
1. 整体架构图：输入 → 12 层 Block → 输出
2. GPT-2 small 配置：12 层 / 768 维 / 12 头
3. 参数分布：Embedding 31% + Block 69%
4. 数据流形状追踪

## 视觉
- 完整架构图（自上而下，分块配色）
- 参数统计饼图

## 素材
- chapters/chapter04_gpt/include/gpt.h
