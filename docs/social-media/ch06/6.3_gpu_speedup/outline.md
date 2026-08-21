# 6.3 GPU 提速实录

> 时长：2 分钟｜平台：★★ 小红书

## 钩子
同一个训练，CPU 和 GPU 差 15 倍

## 要点
1. 实测：CPU 16min vs GPU 56s（15x）
2. 切换只需要 3 行：device / model.to / data.to
3. nvidia-smi 监控：利用率 97%
4. 没 GPU 怎么办：Colab / Kaggle / 云

## 视觉
- 终端 nvidia-smi 截图
- 对比柱状图

## 素材
- chapters/chapter06_finetuning/src/main.cpp
