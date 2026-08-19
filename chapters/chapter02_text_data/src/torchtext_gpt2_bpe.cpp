// torchtext_gpt2_bpe.cpp
// torchtext::GPT2BPEEncoder 的工厂实现
#include "torchtext_gpt2_bpe.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ch2 {
namespace {

// 将 code point 编码为 UTF-8 字符串（对应 Python 的 chr()）
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

// openai gpt-2 的 bytes_to_unicode 算法：
// 可打印的 ASCII 与部分 latin-1 字符映射到自身，其余字节映射到 U+0100 起空位。
// 返回 byte -> (code point 的 UTF-8 编码字符串)
std::unordered_map<int64_t, std::string> build_byte_encoder() {
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
    std::unordered_map<int64_t, std::string> m;
    for (size_t i = 0; i < 256; ++i) {
        m[static_cast<int64_t>(bs[i])] = utf8_encode(static_cast<uint32_t>(cs[i]));
    }
    return m;
}

// 解析 merge 行 "a b"（按空白切分），返回两个 token
std::pair<std::string, std::string> split_merge_line(const std::string& line) {
    std::istringstream iss(line);
    std::string a, b;
    iss >> a >> b;
    return {a, b};
}

}  // namespace

c10::intrusive_ptr<torchtext::GPT2BPEEncoder> load_gpt2_bpe(
    const std::string& encoder_json_path,
    const std::string& vocab_bpe_path,
    bool caching_enabled) {
    // ---- bpe_encoder: token -> id（来自 encoder.json）----
    std::ifstream jfs(encoder_json_path);
    if (!jfs.is_open()) {
        throw std::runtime_error("无法打开 encoder.json: " + encoder_json_path);
    }
    nlohmann::json j;
    jfs >> j;
    std::unordered_map<std::string, int64_t> bpe_encoder;
    for (auto it = j.begin(); it != j.end(); ++it) {
        bpe_encoder[it.key()] = it.value().get<int64_t>();
    }

    // ---- bpe_merge_ranks: "a<sep>b" -> rank（来自 vocab.bpe）----
    // 与 torchtext Python 实现一致：跳过首行（#version），
    // 每行 "a b" 用 separator 连接作为 key，行号作为 rank。
    // 注意：HF 的 merges.txt 末尾没有空行，因此只跳过首行、忽略空行。
    std::ifstream mfs(vocab_bpe_path);
    if (!mfs.is_open()) {
        throw std::runtime_error("无法打开 vocab.bpe: " + vocab_bpe_path);
    }
    const std::string separator = "\x01";  // torchtext 使用的 \u0001
    std::unordered_map<std::string, int64_t> bpe_merge_ranks;
    std::string line;
    bool first_line = true;
    int64_t rank = 0;
    while (std::getline(mfs, line)) {
        if (first_line) {  // "#version: 0.2"
            first_line = false;
            continue;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto pair = split_merge_line(line);
        bpe_merge_ranks[pair.first + separator + pair.second] = rank++;
    }

    // ---- byte_encoder: byte -> unicode 字符 ----
    auto byte_encoder = build_byte_encoder();

    // ---- 构造官方 GPT2BPEEncoder ----
    return c10::make_intrusive<torchtext::GPT2BPEEncoder>(
        bpe_encoder, bpe_merge_ranks, separator, byte_encoder, caching_enabled);
}

}  // namespace ch2
