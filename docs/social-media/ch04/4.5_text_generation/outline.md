# 4.5 文本生成

> 时长：3 分钟｜平台：★★ 小红书

## 钩子
同一个 prompt，三种生成方式差别有多大？

## 要点
1. 贪心：argmax，无聊但稳定
2. 温度采样：logits / T，控制随机性
3. Top-k：前 k 里采样
4. Top-p：累计概率到 p 为止（最常用）

## 视觉
- 同一 prompt 三种结果对比
- 概率分布柱状图（top-k 截断位置）
- 4 种任务配置表

## 素材
- chapters/chapter04_gpt/src/main.cpp 中 generate 函数
