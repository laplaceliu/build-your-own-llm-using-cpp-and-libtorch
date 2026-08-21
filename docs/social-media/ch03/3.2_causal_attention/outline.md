# 3.2 因果注意力

> 时长：3 分钟｜平台：★★ 小红书

## 钩子
模型偷看答案？一行 mask 解决

## 要点
1. 因果 mask：上三角置 -inf
2. 训练/推理对齐：防止信息泄露
3. Dropout：防止过拟合特定位置
4. 完整 CausalAttention 实现

## 视觉
- 掩码矩阵对比图（左：未掩码，右：掩码后）
- 代码截图：masked_fill 那一行

## 素材
- chapters/chapter03_attention/include/attention.h 中 CausalAttention
