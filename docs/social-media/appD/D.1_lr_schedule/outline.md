# D.1 学习率调度

> 时长：3 分钟｜平台：★ 知乎

## 钩子
lr 设好就行？训练中后期要不要变？

## 要点
1. 两阶段：warmup + cosine decay
2. warmup：前 N 步线性升
3. cosine：从 peak 平滑降到 min
4. GPT-2 配置：peak=2.5e-4, min=2.5e-5, warmup=2000

## 视觉
- 学习率曲线图（完整周期）
- 不同策略对比曲线

## 素材
- chapters/chapterD_training_loop
