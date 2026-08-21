# 6.2 SMS Spam 微调

> ⭐ 实战帖｜时长：5 分钟｜平台：★ 知乎

## 钩子
5572 条 SMS，能训出 99% 准确率的垃圾邮件分类器

## 要点
1. 数据集：5572 条（ham 86.6% / spam 13.4%）
2. 配置：lr=5e-5, epochs=5, batch=8
3. 训练：5 epoch 到 99.2% 准确率
4. 评估：accuracy 99.1% / F1 96.8%

## 视觉
- 数据分布饼图
- 训练 loss 曲线
- 混淆矩阵

## 素材
- chapters/chapter06_finetuning/src/main.cpp
