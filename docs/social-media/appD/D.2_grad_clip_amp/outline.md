# D.2 梯度裁剪 + 混合精度

> 时长：3 分钟｜平台：★ 知乎

## 钩子
训练崩了 90% 是这两个原因：梯度爆炸 + 显存不够

## 要点
1. 梯度裁剪：clip_grad_norm_ 防爆炸
2. AMP：FP16 + 损失缩放
3. 实测：显存省 39%，速度快 2.1 倍
4. LibTorch GradScaler 3 步法

## 视觉
- 公式图：梯度裁剪 + 损失缩放
- 终端输出：AMP 前后显存对比

## 素材
- chapters/chapterD_training_loop
