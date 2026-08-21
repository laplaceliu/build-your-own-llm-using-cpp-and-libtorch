# 5.2 AdamW 优化器

> 时长：3 分钟｜平台：★ 知乎

## 钩子
训练 LLM 不用 SGD，用 AdamW。差别在哪？

## 要点
1. Adam = Momentum + RMSProp
2. AdamW = Adam + 解耦的权重衰减
3. GPT-2 配置：lr=2.5e-4, betas=(0.9, 0.95), wd=0.1
4. 调参 4 经验：先小后大 / 看曲线 / warmup / 微调小 lr

## 视觉
- 公式对比表（4 种优化器）
- 不同学习率训练曲线

## 素材
- chapters/chapter05_pretraining/include/training.h
