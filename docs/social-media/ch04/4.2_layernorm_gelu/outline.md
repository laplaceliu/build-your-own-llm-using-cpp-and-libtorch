# 4.2 LayerNorm + GELU

> 时长：3 分钟｜平台：★ 知乎

## 钩子
LayerNorm 和 GELU 看似简单，GPT 选它们是有原因的

## 要点
1. LayerNorm vs BatchNorm：单样本 vs 跨样本
2. GELU vs ReLU：平滑版
3. GPT 选 LayerNorm + GELU 的 4 个原因
4. 在 Transformer Block 中的位置

## 视觉
- GELU vs ReLU 激活曲线对比图
- 代码截图：LayerNorm + GELU 各 5-10 行

## 素材
- chapters/chapter04_gpt/include/gpt.h 中的 LayerNorm 和 GELU
