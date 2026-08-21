# 4.4 加载 GPT-2 small 权重

> ⭐ 深度技术｜时长：4 分钟｜平台：★ 知乎

## 钩子
同一份 safetensors，C++ vs HF 加载，输出 0 误差

## 要点
1. HF key 命名：c_attn (Q/K/V 合一) + Conv1D (转置)
2. 3 大坑：转置 / QKV 拆 / bias 拆
3. 5 步加载流程
4. 验证：max_diff = 0

## 视觉
- Key 映射对照表（HF → 本项目）
- 终端输出：max_diff = 0.000000e+00

## 素材
- chapters/chapter04_gpt/src/main.cpp
- chapters/chapter04_gpt/include/safetensors.h
