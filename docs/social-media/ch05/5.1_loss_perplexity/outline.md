# 5.1 损失函数

> 时长：3 分钟｜平台：★ 知乎

## 钩子
loss 11 是什么水平？困惑度能告诉你

## 要点
1. 交叉熵：-mean(log(p[target]))
2. 文本生成：输入右移一位作 label
3. 困惑度 PPL = e^loss
4. GPT-2 训练曲线参考

## 视觉
- 公式图（交叉熵）
- 训练曲线截图：loss 11 → 7

## 素材
- chapters/chapter05_pretraining/include/training.h
