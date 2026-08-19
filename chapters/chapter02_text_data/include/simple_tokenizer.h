// simple_tokenizer.h
// 第 2 章 2.2~2.4 节：简易分词器 SimpleTokenizerV1 / SimpleTokenizerV2 的 C++ 实现
//
// 对应书中 Python 代码清单 2-3 / 2-4：
//   - SimpleTokenizerV1：按正则 ([,.?_!"()']|--|\s) 分割，去掉空白/空串，查词汇表转 ID
//   - SimpleTokenizerV2：在 V1 基础上，未知词用 <|unk|> 替换
#pragma once

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ch2 {

// 去掉首尾空白（对应 Python 的 str.strip()）
inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\n\r\f\v");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(b, e - b + 1);
}

// Python re.split(pattern, text) 在"整个模式被一个捕获组包裹"时的 C++ 等价实现。
// 返回的列表中既包含分隔符之间的文本，也包含分隔符本身（与 Python 行为一致）。
inline std::vector<std::string> re_split_capture(const std::string& text,
                                                 const std::string& pattern) {
    std::vector<std::string> parts;
    std::regex re(pattern);
    std::sregex_iterator it(text.begin(), text.end(), re);
    std::sregex_iterator end;
    size_t pos = 0;
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        parts.push_back(text.substr(pos, m.position() - pos));  // 分隔符前的文本
        parts.push_back(m.str(0));                              // 分隔符（捕获组）
        pos = m.position() + m.length();
    }
    parts.push_back(text.substr(pos));  // 尾部
    return parts;
}

// 分词：分割后过滤空串与纯空白
// 对应 Python: [item.strip() for item in result if item.strip()]
inline std::vector<std::string> split_tokens(const std::string& text,
                                             const std::string& pattern) {
    std::vector<std::string> tokens;
    for (const auto& p : re_split_capture(text, pattern)) {
        std::string s = trim(p);
        if (!s.empty()) tokens.push_back(s);
    }
    return tokens;
}

// ---- SimpleTokenizerV1（代码清单 2-3）----
class SimpleTokenizerV1 {
public:
    explicit SimpleTokenizerV1(const std::unordered_map<std::string, int>& vocab)
        : str_to_int_(vocab) {
        for (const auto& kv : vocab) int_to_str_[kv.second] = kv.first;
    }

    // 文本 -> 词元 ID
    virtual std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        for (const auto& t : split_tokens(text, split_pattern())) {
            ids.push_back(str_to_int_.at(t));  // 词汇表中不存在的词会抛异常（KeyError）
        }
        return ids;
    }

    // 词元 ID -> 文本
    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) text += " ";
            text += int_to_str_.at(ids[i]);
        }
        // 移除标点符号前的空格：re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        static const std::regex re("\\s+([,.:;?_!\"()'])");
        return std::regex_replace(text, re, "$1");
    }

protected:
    // V1 的分词正则（代码清单 2-3）
    virtual std::string split_pattern() const {
        return "([,._?!\"()']|--|\\s)";
    }

    std::unordered_map<std::string, int> str_to_int_;
    std::unordered_map<int, std::string> int_to_str_;
};

// ---- SimpleTokenizerV2（代码清单 2-4）----
class SimpleTokenizerV2 : public SimpleTokenizerV1 {
public:
    explicit SimpleTokenizerV2(const std::unordered_map<std::string, int>& vocab)
        : SimpleTokenizerV1(vocab) {}

    // 未知词用 <|unk|> 替换
    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> ids;
        for (const auto& t : split_tokens(text, split_pattern())) {
            auto it = str_to_int_.find(t);
            if (it != str_to_int_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(str_to_int_.at("<|unk|>"));  // 未知词 -> <|unk|>
            }
        }
        return ids;
    }

protected:
    // V2 的分词正则（代码清单 2-4，比 V1 多了 : 和 ;）
    std::string split_pattern() const override {
        return "([,.:;?_!\"()']|--|\\s)";
    }
};

}  // namespace ch2
