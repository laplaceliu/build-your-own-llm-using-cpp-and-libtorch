# 5.4 加载 OpenAI 权重

> ⭐ 重点帖｜时长：5 分钟｜平台：★ 知乎 + ★★★ B 站

## 钩子
同一个 prompt，C++ vs HuggingFace Python，输出 0 误差

## 要点
1. 部署形态：纯 C++ 单文件 4.2 MB + 权重 498 MB
2. 5 步加载：tokenizer → 模型 → 权重 → GPU → 推理
3. 验证 max_abs_diff = 0
4. 启动 < 200ms，对比 Python 3-5 秒

## 视觉
- 加载流程图（5 步）
- max_abs_diff 验证截图
- 生成结果对比
- 部署文件清单

## 素材
- chapters/chapter05_pretraining/src/main.cpp
- chapters/chapter05_pretraining/include/safetensors.h
