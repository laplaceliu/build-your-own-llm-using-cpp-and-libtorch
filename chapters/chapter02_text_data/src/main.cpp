// main.cpp
// 第 2 章：处理文本数据（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印，方便对照验证：
//   2.2 文本分词            —— 正则分割 + the-verdict.txt
//   2.3 词元 -> 词元 ID     —— 词汇表 + SimpleTokenizerV1
//   2.4 特殊上下文词元      —— <|unk|> / <|endoftext|> + SimpleTokenizerV2
//   2.5 BPE                 —— GPT-2 字节级 BPE 分词器（tiktoken 等价）
//   2.6 滑动窗口数据采样    —— GPTDatasetV1 / DataLoader 等价实现
//   2.7 词元嵌入            —— torch::nn::Embedding
//   2.8 位置嵌入            —— token embedding + positional embedding
#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"
#include "simple_tokenizer.h"
#include "torchtext_gpt2_bpe.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

namespace {

// ---- 工具 ----
std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("无法打开文件: " + path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

// 打印 token 及其文本（便于对照书中"上下文 ----> 目标"演示）
void print_predict_pairs(const ch2::BpeTokenizer& tk,
                         const std::vector<int>& sample, int context_size) {
    for (int i = 1; i <= context_size; ++i) {
        std::vector<int> ctx(sample.begin(), sample.begin() + i);
        std::vector<int> desired{sample[i]};
        std::cout << tk.decode(ctx) << " ----> " << tk.decode(desired) << "\n";
    }
}

void print_ids(const std::string& label, const std::vector<int>& ids) {
    std::cout << label;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "\n";
}

// ---- 2.6 滑动窗口数据加载器（GPTDatasetV1 + DataLoader 的 C++ 等价）----
struct GPTBatch {
    torch::Tensor inputs;   // (batch, max_length)
    torch::Tensor targets;  // (batch, max_length)
};

class GPTDataLoader {
public:
    GPTDataLoader(const std::vector<int>& token_ids, int batch_size, int max_length,
                  int stride, bool shuffle, bool drop_last, uint64_t seed = 0) {
        // 对应 GPTDatasetV1: for i in range(0, len - max_length, stride)
        for (int i = 0; i + max_length < static_cast<int>(token_ids.size()); i += stride) {
            input_rows_.push_back(
                std::vector<int64_t>(token_ids.begin() + i, token_ids.begin() + i + max_length));
            target_rows_.push_back(
                std::vector<int64_t>(token_ids.begin() + i + 1,
                                     token_ids.begin() + i + max_length + 1));
        }

        size_t n = input_rows_.size();
        if (drop_last) n = n / batch_size * batch_size;

        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(idx.begin(), idx.end(), rng);
        }

        for (size_t b = 0; b < n; b += batch_size) {
            std::vector<torch::Tensor> in_ts, tgt_ts;
            for (size_t k = b; k < std::min(b + static_cast<size_t>(batch_size), n); ++k) {
                in_ts.push_back(torch::tensor(input_rows_[idx[k]], torch::kLong));
                tgt_ts.push_back(torch::tensor(target_rows_[idx[k]], torch::kLong));
            }
            batches_.push_back({torch::stack(in_ts), torch::stack(tgt_ts)});
        }
    }

    size_t num_batches() const { return batches_.size(); }
    size_t num_samples() const { return input_rows_.size(); }
    const std::vector<GPTBatch>& batches() const { return batches_; }

private:
    std::vector<std::vector<int64_t>> input_rows_;
    std::vector<std::vector<int64_t>> target_rows_;
    std::vector<GPTBatch> batches_;
};

// ---- 2.2 文本分词 ----
void demo_tokenization(const std::string& raw_text) {
    section("2.2 文本分词（Text Tokenization）");
    std::cout << "Total number of character: " << raw_text.size() << "\n";
    std::cout << "raw_text[:99]: " << raw_text.substr(0, 99) << "\n";

    // 简单的示例文本
    const std::string text = "Hello, world. This, is a test.";
    std::cout << "\n-- 按空白分割 (re.split(r'(\\s)', text)) --\n";
    for (const auto& t : ch2::re_split_capture(text, "(\\s)")) {
        std::cout << "'" << t << "' ";
    }
    std::cout << "\n";

    std::cout << "\n-- 在 [,.] 或空白处分割，再过滤空白 --\n";
    std::cout << "['";
    const auto& tokens1 = ch2::split_tokens(text, "([,.]|\\s)");
    for (size_t i = 0; i < tokens1.size(); ++i) {
        if (i) std::cout << "', '";
        std::cout << tokens1[i];
    }
    std::cout << "']\n";

    std::cout << "\n-- 更多标点: \"Hello, world. Is this-- a test?\" --\n";
    const auto& tokens2 =
        ch2::split_tokens("Hello, world. Is this-- a test?", "([,.:;?_!\"()']|--|\\s)");
    std::cout << "['";
    for (size_t i = 0; i < tokens2.size(); ++i) {
        if (i) std::cout << "', '";
        std::cout << tokens2[i];
    }
    std::cout << "']\n";

    // 应用到 The Verdict 全文
    std::cout << "\n-- 对 the-verdict.txt 全文分词 --\n";
    const auto& preprocessed = ch2::split_tokens(raw_text, "([,.:;?_!\"()']|--|\\s)");
    std::cout << "词元数量 len(preprocessed) = " << preprocessed.size() << "\n";
    std::cout << "preprocessed[:30]: [";
    for (size_t i = 0; i < 30 && i < preprocessed.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << "'" << preprocessed[i] << "'";
    }
    std::cout << "]\n";
}

// ---- 2.3 词元 -> 词元 ID（SimpleTokenizerV1）----
std::unordered_map<std::string, int> demo_vocab_v1(const std::string& raw_text) {
    section("2.3 将词元转换为词元 ID（SimpleTokenizerV1）");

    const auto& preprocessed = ch2::split_tokens(raw_text, "([,.:;?_!\"()']|--|\\s)");
    std::set<std::string> unique(preprocessed.begin(), preprocessed.end());
    std::cout << "词汇表大小 vocab_size = " << unique.size() << "\n";

    std::unordered_map<std::string, int> vocab;
    int i = 0;
    for (const auto& t : unique) vocab[t] = i++;
    std::cout << "vocab 前 10 个条目:\n";
    for (const auto& t : unique) {
        std::cout << "  ('" << t << "', " << vocab[t] << ")\n";
        if (vocab[t] >= 9) break;
    }

    ch2::SimpleTokenizerV1 tokenizer(vocab);
    const std::string text =
        "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride.";
    auto ids = tokenizer.encode(text);
    std::cout << "\nencode(text) = [";
    for (size_t k = 0; k < ids.size(); ++k) {
        if (k) std::cout << ", ";
        std::cout << ids[k];
    }
    std::cout << "]\n";
    std::cout << "decode(ids) = \"" << tokenizer.decode(ids) << "\"\n";
    return vocab;
}

// ---- 2.4 特殊上下文词元（SimpleTokenizerV2）----
void demo_vocab_v2(const std::string& raw_text) {
    section("2.4 引入特殊上下文词元（SimpleTokenizerV2）");

    const auto& preprocessed = ch2::split_tokens(raw_text, "([,.:;?_!\"()']|--|\\s)");
    std::set<std::string> unique(preprocessed.begin(), preprocessed.end());
    std::vector<std::string> all_tokens(unique.begin(), unique.end());
    all_tokens.push_back("<|endoftext|>");
    all_tokens.push_back("<|unk|>");
    std::cout << "更新后的词汇表大小 = " << all_tokens.size() << "\n";

    std::unordered_map<std::string, int> vocab;
    for (size_t i = 0; i < all_tokens.size(); ++i) vocab[all_tokens[i]] = i;
    std::cout << "词汇表最后 5 个条目:\n";
    for (size_t i = all_tokens.size() - 5; i < all_tokens.size(); ++i) {
        std::cout << "  ('" << all_tokens[i] << "', " << vocab[all_tokens[i]] << ")\n";
    }

    ch2::SimpleTokenizerV2 tokenizer(vocab);
    const std::string text1 = "Hello, do you like tea?";
    const std::string text2 = "In the sunlit terraces of the palace.";
    const std::string text = text1 + " <|endoftext|> " + text2;
    std::cout << "\ntext = \"" << text << "\"\n";

    auto ids = tokenizer.encode(text);
    std::cout << "encode(text) = [";
    for (size_t k = 0; k < ids.size(); ++k) {
        if (k) std::cout << ", ";
        std::cout << ids[k];
    }
    std::cout << "]\n";
    std::cout << "decode(encode(text)) = \"" << tokenizer.decode(ids) << "\"\n";
}

// ---- 2.5 BPE ----
void demo_bpe(const ch2::BpeTokenizer& tokenizer) {
    section("2.5 BPE（字节对编码，tiktoken gpt2 等价）");
    std::cout << "词汇表大小 vocab_size = " << tokenizer.vocab_size() << "\n";

    const std::string text =
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
    auto integers = tokenizer.encode(text, {"<|endoftext|>"});
    std::cout << "\nencode(text) = [";
    for (size_t i = 0; i < integers.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << integers[i];
    }
    std::cout << "]\n";

    std::string decoded = tokenizer.decode(integers);
    std::cout << "decode(integers) = \"" << decoded << "\"\n";

    // 练习 2.1：对未知单词 "Akwirw ier" 进行分词
    const std::string unknown = "Akwirw ier";
    auto unknown_ids = tokenizer.encode(unknown);
    std::cout << "\n[练习 2.1] 未知单词 \"" << unknown << "\" 的 BPE 分词:\n";
    std::cout << "  encode = [";
    for (size_t i = 0; i < unknown_ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << unknown_ids[i];
    }
    std::cout << "]\n";
    for (int id : unknown_ids) {
        std::cout << "  decode([" << id << "]) = \"" << tokenizer.decode({id}) << "\"\n";
    }
    std::cout << "  decode(全部) = \"" << tokenizer.decode(unknown_ids) << "\"\n";
}

// ---- 2.5b 使用 torchtext 官方 GPT2BPEEncoder 的 BPE ----
// torchtext（PyTorch 文本库）提供了 GPT-2 BPE 分词器的官方 C++ 实现
// torch::text::GPT2BPEEncoder（源码见 third_party/torchtext/），与手写版对照。
void demo_bpe_torchtext(const ch2::BpeTokenizer& mine,
                        const std::string& data_dir, const std::string& raw_text) {
    section("2.5b BPE（torchtext 官方 GPT2BPEEncoder）");
    auto tt = ch2::load_gpt2_bpe(data_dir + "/encoder.json", data_dir + "/vocab.bpe");
    std::cout << "词汇表大小 vocab_size = " << tt->GetBPEEncoder().size() << "\n";

    // 注册特殊词元 <|endoftext|>（等价于 tiktoken 的 allowed_special={"<|endoftext|>"}）
    c10::Dict<std::string, std::string> empty_standard_tokens;
    int added = tt->AddSpecialTokens(empty_standard_tokens, {"<|endoftext|>"});
    std::cout << "AddSpecialTokens 新增词元数 = " << added << "（<|endoftext|> 已在词表，仅注册为整体 token）\n";

    const std::string text =
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
    auto integers = tt->Encode(text);
    std::cout << "\nencode(text) = [";
    for (size_t i = 0; i < integers.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << integers[i];
    }
    std::cout << "]\n";
    std::cout << "decode(integers) = \"" << tt->Decode(integers) << "\"\n";

    // 练习 2.1：未知单词
    const std::string unknown = "Akwirw ier";
    auto unknown_ids = tt->Encode(unknown);
    std::cout << "\n[练习 2.1] 未知单词 \"" << unknown << "\":\n  encode = [";
    for (size_t i = 0; i < unknown_ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << unknown_ids[i];
    }
    std::cout << "]\n";
    for (int id : unknown_ids) {
        std::cout << "  decode([" << id << "]) = \"" << tt->Decode({id}) << "\"\n";
    }
    std::cout << "  decode(全部) = \"" << tt->Decode(unknown_ids) << "\"\n";

    // 与手写版交叉验证。
    // 注意：上面含 <|endoftext|> 的文本中，两版输出不同（torchtext 会把特殊词元
    // 两侧的空格吞掉：本输出为 ..., 30, 50256, 818, ...，而 tiktoken/手写版为
    // ..., 30, 220, 50256, 554, ...），这是特殊词元周围空格处理约定不同所致，
    // 与 BPE 核心算法无关。下面用不含特殊词元的普通文本验证算法本身的一致性：
    const std::string plain = "The sunlit terraces of the palace. Akwirw ier";
    auto tt_plain = tt->Encode(plain);
    auto mine_plain = mine.encode(plain);
    bool same = (tt_plain.size() == mine_plain.size());
    if (same) {
        for (size_t i = 0; i < mine_plain.size(); ++i) {
            if (mine_plain[i] != static_cast<int>(tt_plain[i])) { same = false; break; }
        }
    }
    std::cout << "\n-- 与手写版交叉验证（不含特殊词元的普通文本）--\n";
    std::cout << "  text = \"" << plain << "\"\n";
    std::cout << "  torchtext encode: [";
    for (size_t i = 0; i < tt_plain.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << tt_plain[i];
    }
    std::cout << "]\n";
    std::cout << "  手写版 encode:   [";
    for (size_t i = 0; i < mine_plain.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << mine_plain[i];
    }
    std::cout << "]\n";
    std::cout << "  结果一致: " << (same ? "yes" : "no") << "（长度 " << tt_plain.size()
              << "）\n";

    // 全文 token 数（书中 2.6 节应得 5145）
    auto tt_full = tt->Encode(raw_text);
    auto mine_full = mine.encode(raw_text);
    std::cout << "\n-- the-verdict.txt 全文 --\n";
    std::cout << "  torchtext token 数 = " << tt_full.size() << "\n";
    std::cout << "  手写版 token 数   = " << mine_full.size() << "\n";
}

// ---- 2.6 滑动窗口数据采样 ----
void demo_dataloader(const ch2::BpeTokenizer& tokenizer, const std::string& raw_text) {
    section("2.6 使用滑动窗口进行数据采样");
    auto enc_text = tokenizer.encode(raw_text);
    std::cout << "BPE 分词后词元总数 len(enc_text) = " << enc_text.size() << "\n";

    // 移除前 50 个词元（与书中一致）
    std::vector<int> enc_sample(enc_text.begin() + 50, enc_text.end());

    const int context_size = 4;
    std::vector<int> x(enc_sample.begin(), enc_sample.begin() + context_size);
    std::vector<int> y(enc_sample.begin() + 1, enc_sample.begin() + context_size + 1);
    std::cout << "\nx = [";
    for (size_t i = 0; i < x.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << x[i];
    }
    std::cout << "]\ny = [";
    for (size_t i = 0; i < y.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << y[i];
    }
    std::cout << "]\n";

    std::cout << "\n-- 下一单词预测任务（ID 形式）--\n";
    for (int i = 1; i <= context_size; ++i) {
        std::cout << "  [";
        for (int k = 0; k < i; ++k) {
            if (k) std::cout << ", ";
            std::cout << enc_sample[k];
        }
        std::cout << "] ----> " << enc_sample[i] << "\n";
    }

    std::cout << "\n-- 下一单词预测任务（文本形式）--\n";
    print_predict_pairs(tokenizer, enc_sample, context_size);

    // ---- 数据加载器：batch_size=1, max_length=4, stride=1 ----
    std::cout << "\n-- DataLoader(batch_size=1, max_length=4, stride=1) --\n";
    GPTDataLoader dl1(enc_text, /*batch_size=*/1, /*max_length=*/4, /*stride=*/1,
                      /*shuffle=*/false, /*drop_last=*/true);
    std::cout << "数据集样本数 = " << dl1.num_samples() << "\n";
    const auto& b1 = dl1.batches()[0];
    std::cout << "first_batch inputs  = " << b1.inputs << "\n";
    std::cout << "first_batch targets = " << b1.targets << "\n";
    const auto& b2 = dl1.batches()[1];
    std::cout << "second_batch inputs = " << b2.inputs << "\n";

    // ---- batch_size=8, max_length=4, stride=4 ----
    std::cout << "\n-- DataLoader(batch_size=8, max_length=4, stride=4) --\n";
    GPTDataLoader dl2(enc_text, /*batch_size=*/8, /*max_length=*/4, /*stride=*/4,
                      /*shuffle=*/false, /*drop_last=*/true);
    const auto& b8 = dl2.batches()[0];
    std::cout << "Inputs shape = " << b8.inputs.sizes() << "\n";
    std::cout << "Inputs:\n" << b8.inputs << "\n";
    std::cout << "Targets:\n" << b8.targets << "\n";
}

// ---- 2.7 词元嵌入 ----
void demo_token_embedding(const torch::Device& device) {
    section("2.7 创建词元嵌入");
    std::cout << "设备: " << device << "\n";
    const int vocab_size = 6;
    const int output_dim = 3;
    torch::manual_seed(123);
    auto embedding_layer = torch::nn::Embedding(vocab_size, output_dim);
    // 权重在 CPU 上 seed 123 初始化（与书数值一致），然后迁移到计算设备
    embedding_layer->to(device);
    std::cout << "embedding_layer.weight（6x3）:\n"
              << embedding_layer->weight.to(torch::kCPU) << "\n";

    auto e3 = embedding_layer->forward(torch::tensor({3}, torch::kLong).to(device));
    std::cout << "embedding(tensor([3])):\n" << e3.to(torch::kCPU) << "\n";

    auto input_ids = torch::tensor({2, 3, 5, 1}, torch::kLong).to(device);
    std::cout << "embedding(tensor([2, 3, 5, 1])):\n"
              << embedding_layer->forward(input_ids).to(torch::kCPU) << "\n";
}

// ---- 2.8 位置嵌入 ----
void demo_positional_embedding(const ch2::BpeTokenizer& tokenizer,
                               const std::string& raw_text,
                               const torch::Device& device) {
    section("2.8 编码单词位置信息");
    const int vocab_size = 50257;
    const int output_dim = 256;
    const int max_length = 4;
    auto token_embedding_layer = torch::nn::Embedding(vocab_size, output_dim);
    token_embedding_layer->to(device);

    auto enc_text = tokenizer.encode(raw_text);
    GPTDataLoader dl(enc_text, /*batch_size=*/8, /*max_length=*/max_length,
                     /*stride=*/max_length, /*shuffle=*/false, /*drop_last=*/true);
    const auto& first = dl.batches()[0];
    torch::Tensor inputs = first.inputs.to(device);
    std::cout << "Token IDs (8x4):\n" << inputs.to(torch::kCPU) << "\n";
    std::cout << "Inputs shape = " << inputs.sizes() << "\n";

    auto token_embeddings = token_embedding_layer->forward(inputs);
    std::cout << "token_embeddings.shape = " << token_embeddings.sizes() << "\n";

    auto pos_embedding_layer = torch::nn::Embedding(max_length, output_dim);
    pos_embedding_layer->to(device);
    auto pos_embeddings =
        pos_embedding_layer->forward(torch::arange(max_length, torch::kLong).to(device));
    std::cout << "pos_embeddings.shape = " << pos_embeddings.sizes() << "\n";

    auto input_embeddings = token_embeddings + pos_embeddings;
    std::cout << "input_embeddings.shape = " << input_embeddings.sizes() << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = DATA_DIR;
    if (argc > 1) data_dir = argv[1];

    try {
        const std::string raw_text = read_file(data_dir + "/the-verdict.txt");

        torch::Device device =
            torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                        : torch::Device(torch::kCPU);
        std::cout << "=== 第 2 章：处理文本数据（C++ + LibTorch）===\n"
                  << "数据目录: " << data_dir << "\n"
                  << "设备: " << device << "\n";

        demo_tokenization(raw_text);
        demo_vocab_v1(raw_text);
        demo_vocab_v2(raw_text);

        ch2::BpeTokenizer tokenizer(data_dir + "/encoder.json", data_dir + "/vocab.bpe");
        demo_bpe(tokenizer);
        demo_bpe_torchtext(tokenizer, data_dir, raw_text);
        demo_dataloader(tokenizer, raw_text);
        demo_token_embedding(device);
        demo_positional_embedding(tokenizer, raw_text, device);

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 2 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
