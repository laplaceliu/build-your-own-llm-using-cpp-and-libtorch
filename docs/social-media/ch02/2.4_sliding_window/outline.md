# 2.4 滑动窗口采样

> 时长：3 分钟｜平台：★ 知乎

## 钩子
模型输入长度固定，但文本无限——怎么切？

## 要点
1. 滑动窗口：max_length + stride
2. 输入 vs 目标：右移一位
3. GPTDataLoader 实现
4. 参数选择：max_length / stride / batch_size

## 视觉
- 滑动窗口动画（4 步分解）
- 代码截图：GPTDataLoader
- 参数对照表

## 素材
- docs/modules/chapter02_text_data/pages/index.adoc §2.6
