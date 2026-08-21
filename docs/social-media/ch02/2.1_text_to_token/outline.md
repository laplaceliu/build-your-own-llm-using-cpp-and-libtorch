# 2.1 文本怎么变数字

> 时长：2 分钟｜平台：★★ 小红书

## 钩子
LLM 不认识汉字，它只认识 0/1

## 要点
1. 三层映射：字符 → token → ID → 向量
2. 为什么要分词：序列长度 + 词汇表大小 + OOV
3. 4 种主流算法：BPE / WordPiece / Unigram / SentencePiece
4. 本项目实现 GPT-2 同款 BPE

## 视觉
- 三层映射流程图
- "你好世界" 字符级 vs 词级对比表
- 4 种算法对比表

## 素材
- docs/modules/chapter02_text_data/pages/index.adoc §2.1
