// bpe_tokenizer.cpp
// GPT-2 字节级 BPE 分词器实现
#include "bpe_tokenizer.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <fstream>
#include <regex>

namespace ch2 {

namespace {

// 解码 UTF-8 序列中的下一个 code point（i 会前进）
uint32_t utf8_decode(const std::string& s, size_t& i) {
    uint8_t c = static_cast<uint8_t>(s[i]);
    if (c < 0x80) {
        i += 1;
        return c;
    } else if ((c & 0xE0) == 0xC0) {
        uint32_t cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
                      (static_cast<uint8_t>(s[i + 1]) & 0x3F);
        i += 2;
        return cp;
    } else if ((c & 0xF0) == 0xE0) {
        uint32_t cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
                      ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                      (static_cast<uint8_t>(s[i + 2]) & 0x3F);
        i += 3;
        return cp;
    } else {
        uint32_t cp = (static_cast<uint32_t>(c & 0x07) << 18) |
                      ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                      ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                      (static_cast<uint8_t>(s[i + 3]) & 0x3F);
        i += 4;
        return cp;
    }
}

// 编码一个 code point 为 UTF-8
std::string utf8_encode(uint32_t cp) {
    if (cp < 0x80) {
        return std::string(1, static_cast<char>(cp));
    } else if (cp < 0x800) {
        return {static_cast<char>(0xC0 | (cp >> 6)),
                static_cast<char>(0x80 | (cp & 0x3F))};
    } else if (cp < 0x10000) {
        return {static_cast<char>(0xE0 | (cp >> 12)),
                static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                static_cast<char>(0x80 | (cp & 0x3F))};
    } else {
        return {static_cast<char>(0xF0 | (cp >> 18)),
                static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
                static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                static_cast<char>(0x80 | (cp & 0x3F))};
    }
}

// 按 code point 切分字符串为字符数组
std::vector<std::string> split_code_points(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = utf8_decode(s, i);
        chars.push_back(utf8_encode(cp));
    }
    return chars;
}

}  // namespace

// GPT-2 的 bytes_to_unicode 映射（openai 原始算法）：
// 可打印 ASCII 与部分 latin-1 字符映射到自身，其余字节映射到 U+0100 起的空位
void BpeTokenizer::build_byte_to_unicode() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);        // 33..126
    for (int i = 161; i <= 172; ++i) bs.push_back(i);        // 161..172
    for (int i = 174; i <= 255; ++i) bs.push_back(i);        // 174..255
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < 256; ++i) {
        std::string ch = utf8_encode(static_cast<uint32_t>(cs[i]));
        byte_to_char_[static_cast<uint8_t>(bs[i])] = ch;
        char_to_byte_[ch] = bs[i];
    }
}

BpeTokenizer::BpeTokenizer(const std::string& encoder_json_path,
                           const std::string& vocab_bpe_path) {
    build_byte_to_unicode();

    // ---- 加载 encoder.json：token -> id ----
    std::ifstream jfs(encoder_json_path);
    if (!jfs.is_open()) {
        throw std::runtime_error("无法打开 encoder.json: " + encoder_json_path);
    }
    nlohmann::json j;
    jfs >> j;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = it.value().get<int>();
        encoder_[it.key()] = id;
        decoder_[id] = it.key();
    }

    // ---- 加载 vocab.bpe：每行 "a b"，行号即 merge 的 rank ----
    std::ifstream mfs(vocab_bpe_path);
    if (!mfs.is_open()) {
        throw std::runtime_error("无法打开 vocab.bpe: " + vocab_bpe_path);
    }
    std::string line;
    bool first_line = true;
    int rank = 0;
    while (std::getline(mfs, line)) {
        if (first_line) {  // 跳过 "#version: 0.2"
            first_line = false;
            continue;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto pos = line.find(' ');
        if (pos == std::string::npos) continue;
        std::string a = line.substr(0, pos);
        std::string b = line.substr(pos + 1);
        merges_.insert({{a, b}, rank++});
    }
}

// GPT-2 的"单词"切分正则（tiktoken gpt2 regex 的 ASCII 近似版）。
// 注意：\p{L}/\p{N} 在 std::regex 中不可用，用 [A-Za-z]/[0-9] 代替，
// 对英文文本（本书所有示例）结果与 tiktoken 完全一致。
std::vector<std::string> BpeTokenizer::split_words(const std::string& text) const {
    // 's|'t|'re|'ve|'m|'ll|'d | 可选空格+字母串 | 可选空格+数字串 |
    // 可选空格+其他标点串 | 行尾空白 | 其余空白
    static const std::regex word_re(
        "(?:'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+)");
    std::vector<std::string> words;
    std::sregex_iterator it(text.begin(), text.end(), word_re);
    std::sregex_iterator end;
    for (; it != end; ++it) words.push_back(it->str(0));
    return words;
}

// 在单个"单词"内做贪心 BPE merge（token 为 bytes_to_unicode 后的字符序列）
std::vector<std::string> BpeTokenizer::bpe(const std::string& b2u_word) const {
    std::vector<std::string> word = split_code_points(b2u_word);
    if (word.size() == 1) return word;

    while (true) {
        // 找 rank 最小的相邻对（等价于 tiktoken 的 min(pairs, key=merges.get)）
        std::pair<std::string, std::string> best;
        int best_rank = INT_MAX;
        for (size_t k = 0; k + 1 < word.size(); ++k) {
            auto it = merges_.find({word[k], word[k + 1]});
            int r = (it == merges_.end()) ? INT_MAX : it->second;
            if (r < best_rank) {
                best_rank = r;
                best = {word[k], word[k + 1]};
            }
        }
        if (best_rank == INT_MAX) break;  // 没有可合并的对

        // 合并所有出现的 best 对
        std::vector<std::string> merged;
        for (size_t k = 0; k < word.size();) {
            if (k + 1 < word.size() && word[k] == best.first && word[k + 1] == best.second) {
                merged.push_back(word[k] + word[k + 1]);
                k += 2;
            } else {
                merged.push_back(word[k]);
                k += 1;
            }
        }
        word.swap(merged);
        if (word.size() == 1) break;
    }
    return word;
}

// 普通文本编码（不含特殊词元）
std::vector<int> BpeTokenizer::encode_plain(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& word : split_words(text)) {
        // word 的每个字节 -> bytes_to_unicode 字符
        std::string b2u_word;
        for (char c : word) {
            b2u_word += byte_to_char_[static_cast<uint8_t>(c)];
        }
        for (const auto& token : bpe(b2u_word)) {
            auto it = encoder_.find(token);
            if (it != encoder_.end()) ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int> BpeTokenizer::encode(const std::string& text,
                                      const std::vector<std::string>& allowed_special) const {
    std::vector<int> ids;
    if (allowed_special.empty()) {
        return encode_plain(text);
    }

    // 按特殊词元切分文本，普通段做 BPE 编码，特殊词元整体映射为其 ID
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        std::string best_special;
        for (const auto& s : allowed_special) {
            size_t p = text.find(s, pos);
            if (p != std::string::npos && (best == std::string::npos || p < best)) {
                best = p;
                best_special = s;
            }
        }
        if (best == std::string::npos) {
            auto part = encode_plain(text.substr(pos));
            ids.insert(ids.end(), part.begin(), part.end());
            break;
        }
        auto part = encode_plain(text.substr(pos, best - pos));
        ids.insert(ids.end(), part.begin(), part.end());
        ids.push_back(encoder_.at(best_special));  // 特殊词元 ID（如 <|endoftext|> = 50256）
        pos = best + best_special.size();
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    // id -> b2u 字符序列
    std::string chars;
    for (int id : ids) {
        auto it = decoder_.find(id);
        if (it != decoder_.end()) chars += it->second;
    }
    // b2u 字符 -> 字节
    std::string bytes;
    size_t i = 0;
    while (i < chars.size()) {
        uint32_t cp = utf8_decode(chars, i);
        auto it = char_to_byte_.find(utf8_encode(cp));
        if (it != char_to_byte_.end()) bytes.push_back(static_cast<char>(it->second));
    }
    return bytes;  // 字节串即 UTF-8 文本
}

}  // namespace ch2
