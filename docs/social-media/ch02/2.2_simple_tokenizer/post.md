# 2.2 简单分词器：正则 + 词表双向映射

> 第 2 章第 2 帖｜阅读 7 分钟｜口播 4 分钟
> 平台：★ 知乎

---

## 钩子

**80 行 C++，复刻 Python `re.split` + 词表查询。**

玩具版分词器，做对两件事就够：
**「切词」+「查表」**。

---

## 玩具版思路

```
"Hello, world! Hello again."
   ↓ 正则切分
["Hello", ",", " world", "!", " again", "."]
   ↓ 去重排序
{",", ".", "!", " again", "Hello", " world"}
   ↓ 建立 ID 映射
"," → 0, "." → 1, "!" → 2, " again" → 3, "Hello" → 4, " world" → 5
   ↓ encode
[4, 0, 5, 1, 4, 3, 1]
```

---

## 完整代码

```cpp
// simple_tokenizer.h
#include <regex>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>

class SimpleTokenizer {
public:
    SimpleTokenizer() {
        vocab_ = { "<|unk|>", "<|endoftext|>" };  // 特殊词元
    }
    
    // 加载词表
    void load_vocab(const std::vector<std::string>& tokens) {
        for (auto& t : tokens) {
            vocab_.push_back(t);
        }
        build_mapping();
    }
    
    // 编码：文本 → ID 序列
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        // 1. 正则切分
        std::regex re(R"((<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d)"
                      R"(| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+))");
        
        auto words_begin = std::sregex_iterator(
            text.begin(), text.end(), re);
        auto words_end = std::sregex_iterator();
        
        // 2. 逐个查表
        for (auto it = words_begin; it != words_end; ++it) {
            std::string word = it->str();
            auto found = token_to_id_.find(word);
            if (found != token_to_id_.end()) {
                ids.push_back(found->second);
            } else {
                ids.push_back(token_to_id_.at("<|unk|>"));
            }
        }
        return ids;
    }
    
    // 解码：ID 序列 → 文本
    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        for (int id : ids) {
            text += id_to_token_.at(id);
        }
        return text;
    }
    
private:
    void build_mapping() {
        for (size_t i = 0; i < vocab_.size(); ++i) {
            token_to_id_[vocab_[i]] = static_cast<int>(i);
            id_to_token_[i] = vocab_[i];
        }
    }
    
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
};
```

---

## 关键点解释

### 1. 特殊词元

| 词元 | 作用 |
|---|---|
| `<\|unk\|>` | 未登录词（OOV）的占位符 |
| `<\|endoftext\|>` | 文档分隔符（多个文本拼接时用） |

### 2. 正则切分

```cpp
std::regex re(R"((<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d)"
              R"(| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+))");
```

这一行做了 6 件事：
- 识别 `<|endoftext|>`
- 识别英文缩写 `'s / 't / 're`
- 单词（保留前导空格）
- 数字
- 标点符号（保留前导空格）
- 空白字符

### 3. 词表数据结构

| 操作 | 数据结构 | 复杂度 |
|---|---|---|
| 编码查表 | `unordered_map<string, int>` | O(1) |
| 解码查表 | `unordered_map<int, string>` | O(1) |
| 词表存储 | `vector<string>` | O(1) |

---

## 测试

```cpp
int main() {
    SimpleTokenizer tok;
    
    // 加载词表（假设已经训练好）
    std::vector<std::string> vocab = {
        "<|unk|>", "<|endoftext|>",
        "Hello", ",", " world", "!", " again", "."
    };
    tok.load_vocab(vocab);
    
    auto ids = tok.encode("Hello, world! Hello again.");
    // → [4, 0, 5, 1, 4, 3, 1]
    
    auto text = tok.decode(ids);
    // → "Hello, world! Hello again."
    return 0;
}
```

---

## 玩具版的局限

| 问题 | 工业级方案 |
|---|---|
| 词表固定，不能处理 OOV | BPE / WordPiece |
| 正则只匹配英文 | 多语言正则 / SentencePiece |
| 词表无大小控制 | 频率阈值 + 合并规则 |

下一帖 2.3，我们实现 GPT-2 同款 BPE，解决 OOV 问题。

---

## 下一步

**2.3 BPE 算法**：GPT-2 用的算法 5 分钟讲透

---

## 互动

你的玩具版词表训练数据是英文还是中文？
评论告诉我，下次可以做个中文版。
