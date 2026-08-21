# 7.3 评估方案：Ollama 打分

> 时长：4 分钟｜平台：★ 知乎

## 钩子
没有标准答案的任务，怎么评估？

## 要点
1. LLM-as-a-Judge：用强模型当裁判
2. 4 步流程：测试集 → 生成 → 打分 → 分析
3. 实测：Base 2.34 → SFT 3.87
4. 本地 Ollama 免费，GPT-4 API 贵

## 视觉
- 评估 Prompt 截图
- 分数分布对比图（4-5 分比例 20%→56%）

## 素材
- scripts/eval_llm_judge.py
