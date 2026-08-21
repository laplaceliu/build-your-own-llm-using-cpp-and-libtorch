# 2.3 BPE：GPT-2 用的算法 5 分钟讲透

> 第 2 章第 3 帖｜⭐ 重点帖｜阅读 9 分钟｜口播 5 分钟
> 平台：★ 知乎（主）+ ★★ 小红书（精简版）

---

## 钩子

**GPT-2 怎么分词？BPE = 字节级 + 贪心合并 + 频率统计。**

玩具版的硬伤：碰到 OOV 直接扔 `<|unk|>`。
BPE 通过「高频对合并」解决 OOV，同时控制词表大小。

---

## BPE 三步走

```
训练阶段：
  1. 初始词表 = 256 个字节（0x00–0xFF）
  2. 反复找最高频的相邻 token 对，合并成新 token
  3. 直到词表大小达到目标（比如 50,257）

推理阶段：
  给定文本 → 字节 → 按训练时学到的合并规则 → token 序列
```

---

## 一个最小例子

语料（假设只有这两句重复 100 次）：
```
"aaabdaaabac"
```

**训练过程**：

| 步 | 最高频对 | 合并后 | 词表大小 |
|---|---|---|---|
| 0 | (a, a) | aa | 257 |
| 1 | (aa, a) | aaa | 258 |
| 2 | (aaa, a) | aaaa | 259 |
| 3 | (aa, b) | aab | 260 |
| 4 | (aab, a) | aaba | 261 |
| 5 | (aaba, a) | aabaa | 262 |

最终编码 "aaabdaaabac"：
```
[aab, d, aab, a, c]
5 个 token！比 11 个字符少了 55%。
```

这就是 BPE 的精髓：**高频子串变成单 token，罕见子串保持原子**。

---

## GPT-2 实际算法（5 个细节）

### 细节 1：bytes_to_unicode

为什么要把字节先映射到 unicode？
- 0–255 里很多是控制字符（`\x00`, `\x01`）不能直接当字符串
- 解决：把可打印字节映到 `ĀāĂă...`，剩下留原样

```cpp
std::map<uint8_t, std::string> bytes_to_unicode() {
    // 可打印字符：!"#$%&'()*+,-./:;<=>?@...
    // 不可打印：映到 Ā=256, ā=257, Ă=258 ...
    
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = '¡'; b <= '¬'; ++b) bs.push_back(b);
    for (int b = '®'; b <= 'ÿ'; ++b) bs.push_back(b);
    
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    
    std::map<uint8_t, std::string> mapping;
    for (size_t i = 0; i < bs.size(); ++i) {
        mapping[bs[i]] = char32_to_utf8(cs[i]);
    }
    return mapping;
}
```

### 细节 2：单词切分正则

```python
# Python 原版正则
pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+"""
     + r"""| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
```

C++ 等价（用 `std::wregex` 支持 Unicode）：
```cpp
std::wregex re(LR"(('s|'t|'re|'ve|'m|'ll|'d)"
                LR"(| ?[A-Za-z]+| ?[0-9]+)"
                LR"(| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+))");
```

### 细节 3：统计 + 合并

```cpp
// 伪代码
std::map<std::pair<std::string, std::string>, int> get_pairs(
    const std::vector<std::string>& word) {
    std::map<std::pair<std::string, std::string>, int> pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs[{word[i], word[i+1]}]++;
    }
    return pairs;
}

// 主循环
while (vocab_size < target_size) {
    auto pairs = get_pairs(corpus);
    auto best = max_frequency(pairs);  // 找最高频对
    vocab.insert(best.first + best.second);  // 合并入词表
    corpus = merge_all(corpus, best);  // 全局替换
}
```

### 细节 4：缓存最优合并（GPT-2 训练后输出）

训练完成后，输出一个 `bpe_ranks.json`：
```json
{"Ġ t": 0, "Ġ a": 1, "h e": 2, ...}
```
推理时按 rank 升序合并。

### 细节 5：练习 2.1 — `Akwirw ier` 怎么拆

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Akwirw ier")
# → [33901, 86, 343, 86, 220, 959]
# decode 每个：
# 33901 → "Ak"
# 86    → "w"
# 343   → "ir"
# 86    → "w"
# 220   → " "
# 959   → "ier"
# 所以是 ["Ak", "w", "ir", "w", " ", "ier"]，不需要 <|unk|>
```

---

## 完整流程图

```
训练：
   字节语料
     ↓ bytes_to_unicode
   unicode 字符
     ↓ 正则切分
   单词列表（每个单词 = token 序列）
     ↓ 统计相邻对频率
   pairs[('a', 'b')] = 1234
     ↓ 找最大
   best = ('a', 'b')
     ↓ 全局合并
   所有 "ab" → "ab"（作为新 token）
     ↓ 重复 N 次
   bpe_ranks.json

推理：
   输入文本
     ↓ bytes_to_unicode
     ↓ 正则切分
     ↓ 按 bpe_ranks 顺序合并
   token IDs
```

---

## 测试对比：手写 vs 官方

```cpp
#include <tokenizers_cpp.h>
auto official = tokenizers::Tokenizer::FromBlob(...);

// 用本项目 BPE 编码
auto my_ids = my_bpe.encode("Hello, world!");
// → [15496, 11, 995, 0]

// 用官方 tiktoken 编码
auto official_ids = official.encode("Hello, world!");
// → [15496, 11, 995, 0]

assert(my_ids == official_ids);  // ✅ 完全一致
```

**跨语言数值一致 = 跨框架可对比**。

---

## 下一步

**2.4 滑动窗口采样**：把长文切成训练样本

---

## 互动

你用过哪些分词库？
- HuggingFace Tokenizers
- tiktoken
- SentencePiece
- 本项目手写版

评论区说说体验。
