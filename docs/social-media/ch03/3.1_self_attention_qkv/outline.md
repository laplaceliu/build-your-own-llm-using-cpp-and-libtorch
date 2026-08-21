# 3.1 自注意力 Q/K/V

> 时长：4 分钟｜平台：★ 知乎

## 钩子
注意力机制 = 让模型「挑重点看」

## 要点
1. Q/K/V 直觉：找书类比
2. 公式：softmax(QK^T / √d_k) V
3. 从零实现 SelfAttention
4. 实例：The cat sat 注意力分数
5. 缩放因子 √d_k 必要性

## 视觉
- 公式图（居中）
- Q/K/V 流程示意图
- 注意力分数矩阵热力图

## 素材
- chapters/chapter03_attention/include/attention.h
- docs/modules/chapter03_attention/pages/index.adoc §3.1-3.2
