# 2.2 简单分词器

> 时长：4 分钟｜平台：★ 知乎

## 钩子
80 行 C++，复刻 Python re.split + 词表查询

## 要点
1. 正则切分：`std::regex` + `sregex_iterator`
2. 词表 = `unordered_map<string, int>` + 反向 map
3. 特殊词元：<|unk|> / <|endoftext|>
4. encode/decode 双向映射

## 视觉
- 代码截图：80 行 simple_tokenizer.h
- 终端输出：encode/decode 测试
- 数据结构图

## 素材
- chapters/chapter02_text_data/include/simple_tokenizer.h
- docs/modules/chapter02_text_data/pages/index.adoc §2.2-2.4
