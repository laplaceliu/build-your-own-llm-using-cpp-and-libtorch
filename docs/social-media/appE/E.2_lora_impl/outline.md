# E.2 LoRA 实现

> 时长：4 分钟｜平台：★ 知乎

## 钩子
手写一个 LoRA 包装器，20 行代码

## 要点
1. LoRALinear 20 行代码实现
2. 冻结 base + 注入 A/B
3. 训练参数：3M / 355M (0.85%)
4. 多任务切换：共享 base + 不同 LoRA

## 视觉
- 代码截图：LoRALinear 关键 20 行
- 训练参数量统计截图

## 素材
- chapters/chapterE_lora
