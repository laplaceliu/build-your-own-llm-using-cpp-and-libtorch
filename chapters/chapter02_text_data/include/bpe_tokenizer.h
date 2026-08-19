// bpe_tokenizer.h
// 第 2 章 2.5 节：GPT-2 字节级 BPE 分词器（tiktoken.get_encoding("gpt2") 的 C++ 等价实现）
//
// 数据文件（与 tiktoken gpt2 编码等价）：
//   encoder.json —— token -> id（等价于 HF gpt2 的 vocab.json）
//   vocab.bpe    —— merge 规则，每行 "a b"，行号即 rank（等价于 HF gpt2 的 merges.txt）
//
// 流程：文本 UTF-8 -> bytes -> bytes_to_unicode 映射为可打印字符
//       -> 正则切分成"单词"（含前导空格） -> 单词内贪心 merge -> 查表得 token id
#pragma once

#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ch2 {

class BpeTokenizer {
public:
    BpeTokenizer(const std::string& encoder_json_path, const std::string& vocab_bpe_path);

    // 文本 -> token ID。allowed_special 中列出的特殊词元（如 <|endoftext|>）会被整体保留。
    std::vector<int> encode(const std::string& text,
                            const std::vector<std::string>& allowed_special = {}) const;

    // token ID -> 文本
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return encoder_.size(); }

    // 查询某 token 对应的 ID，不存在返回 -1
    int token_id(const std::string& token) const {
        auto it = encoder_.find(token);
        return it == encoder_.end() ? -1 : it->second;
    }

private:
    void build_byte_to_unicode();
    std::vector<std::string> split_words(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& b2u_word) const;
    std::vector<int> encode_plain(const std::string& text) const;

    std::unordered_map<std::string, int> encoder_;                 // token -> id
    std::unordered_map<int, std::string> decoder_;                 // id -> token
    std::map<std::pair<std::string, std::string>, int> merges_;    // (a,b) -> rank
    std::array<std::string, 256> byte_to_char_;                    // byte -> b2u 字符
    std::unordered_map<std::string, int> char_to_byte_;            // b2u 字符 -> byte
};

}  // namespace ch2
