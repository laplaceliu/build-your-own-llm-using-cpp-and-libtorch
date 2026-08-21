# 2.3 BPE 算法

> ⭐ 重点帖｜时长：5 分钟｜平台：★ 知乎

## 钩子
GPT-2 怎么分词？BPE = 字节级 + 贪心合并 + 频率统计

## 要点
1. BPE 三步：256 字节 → 贪心合并 → 目标大小
2. GPT-2 5 个细节：bytes_to_unicode / 正则 / 统计 / ranks / 练习
3. 流程图：训练 vs 推理
4. 与官方 tiktoken 数值一致

## 视觉
- 流程图：4 步 BPE
- 终端输出：练习 2.1 的 6 个 token 拆分
- 对比：手写版 vs torchtext 官方版

## 素材
- chapters/chapter02_text_data/src/bpe_tokenizer.cpp
- chapters/chapter02_text_data/include/bpe_tokenizer.h
