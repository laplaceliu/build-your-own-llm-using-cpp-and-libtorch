# 6.1 分类头

> 时长：3 分钟｜平台：★ 知乎

## 钩子
「会写文章」的模型怎么改成「判断垃圾邮件」？

## 要点
1. 改造：生成头 → 分类头
2. 用最后一个 token 的隐藏状态
3. 3 种微调方案：冻结 / 全量 / 部分
4. 训练配置：lr=5e-5, epochs=5

## 视觉
- 改造前后对比图
- 代码截图：ClassificationHead

## 素材
- chapters/chapter06_finetuning/include/finetuning.h
