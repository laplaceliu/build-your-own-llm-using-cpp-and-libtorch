# 3.3 多头注意力

> 时长：3 分钟｜平台：★ 知乎

## 钩子
为什么是「多头」而不是「一个更强的头」？

## 要点
1. 直觉：多个专科医生会诊
2. 数学：concat + 投影
3. 实现：循环 vs 批量并行
4. 3 个原因：不同子空间 / 并行效率 / 防过拟合
5. GPT-2 配置：small 12×64

## 视觉
- 多头分流结构图
- GPT-2 4 个尺寸对比表

## 素材
- chapters/chapter03_attention/include/attention.h 中 MultiHeadAttention
