# 4.3 Transformer Block

> 时长：3 分钟｜平台：★ 知乎

## 钩子
Transformer Block = 注意力子层 + FFN 子层 + 2 个残差

## 要点
1. 4 步：LN → Attn → 残差 → LN → FFN → 残差
2. 残差连接：解决梯度消失
3. FFN：768 → 3072 → 768
4. Pre-Norm vs Post-Norm

## 视觉
- Block 内部结构图（标注维度）
- 代码截图：TransformerBlock::forward
- 残差路径示意图

## 素材
- chapters/chapter04_gpt/include/gpt.h 中的 TransformerBlock
