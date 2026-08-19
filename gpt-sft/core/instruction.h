// instruction.h
// 第 7 章：通过微调遵循人类指令 —— 数据集/批处理/训练/回复抽取
//
// 对应书中代码：
//   format_input               代码清单 7-2（Alpaca 提示词风格）
//   InstructionDataset         代码清单 7-4（预词元化）
//   custom_collate_fn          代码清单 7-5（填充 + 目标左移 + -100 掩码）
//   InstructionLoader          DataLoader(collate_fn=...) 的等价实现
//   extract_response           从生成文本抽取回复（7.5/7.7 节）
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"

namespace ch7 {

// ==========================================================================
// 7.2 指令样本结构与 format_input（代码清单 7-2）
// ==========================================================================
struct InstructionEntry {
    std::string instruction;
    std::string input;   // 可能为空
    std::string output;
    std::string model_response;  // 7.7 节由模型生成后填充
};

// Alpaca 提示词风格（代码清单 7-2）
inline std::string format_input(const InstructionEntry& entry) {
    std::string instruction_text =
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n### Instruction:\n" + entry.instruction;
    std::string input_text =
        entry.input.empty() ? "" : ("\n\n### Input:\n" + entry.input);
    return instruction_text + input_text;
}

// 完整训练文本：指令 + 输入 + 期望回复
inline std::string format_full_text(const InstructionEntry& entry) {
    return format_input(entry) + "\n\n### Response:\n" + entry.output;
}

// ==========================================================================
// 7.3 批处理：InstructionDataset（代码清单 7-4）+ custom_collate_fn（清单 7-5）
// ==========================================================================
class InstructionDataset {
public:
    InstructionDataset(const std::vector<InstructionEntry>& data,
                       ch2::BpeTokenizer& tokenizer) {
        for (const auto& entry : data) {
            encoded_texts_.push_back(
                tokenizer.encode(format_full_text(entry), {"<|endoftext|>"}));
        }
    }

    size_t size() const { return encoded_texts_.size(); }
    const std::vector<int>& encoded(int64_t i) const { return encoded_texts_[i]; }

private:
    std::vector<std::vector<int>> encoded_texts_;
};

struct InstructionBatch {
    torch::Tensor inputs;   // [batch, seq]
    torch::Tensor targets;  // [batch, seq]（含 -100 掩码）
};

// custom_collate_fn（代码清单 7-5）：填充到批次最长 + 目标左移 + 掩码多余 pad
// 返回 (inputs, targets)；targets 中除第一个 pad 外全部替换为 ignore_index=-100。
inline std::pair<torch::Tensor, torch::Tensor> custom_collate_fn(
    const std::vector<std::vector<int>>& batch, int64_t pad_token_id = 50256,
    int64_t ignore_index = -100, c10::optional<int64_t> allowed_max_length = c10::nullopt,
    const torch::Device& device = torch::Device(torch::kCPU)) {
    // batch_max_length = max(len(item) + 1)
    int64_t batch_max_length = 0;
    for (const auto& item : batch)
        batch_max_length = std::max(batch_max_length,
                                    static_cast<int64_t>(item.size()) + 1);

    std::vector<torch::Tensor> inputs_lst, targets_lst;
    for (const auto& item : batch) {
        std::vector<int64_t> new_item(item.begin(), item.end());
        new_item.push_back(pad_token_id);  // 附加结束符
        // 填充到 batch_max_length
        while (static_cast<int64_t>(new_item.size()) < batch_max_length)
            new_item.push_back(pad_token_id);

        // inputs = padded[:-1], targets = padded[1:]
        std::vector<int64_t> inputs(new_item.begin(), new_item.end() - 1);
        std::vector<int64_t> targets(new_item.begin() + 1, new_item.end());

        // 掩码：targets 中除第一个 pad 外的所有 pad 替换为 ignore_index
        bool first_pad_seen = false;
        for (auto& t : targets) {
            if (t == pad_token_id) {
                if (first_pad_seen) t = ignore_index;
                first_pad_seen = true;
            }
        }

        // 可选截断到 allowed_max_length
        if (allowed_max_length.has_value()) {
            int64_t lim = allowed_max_length.value();
            if (static_cast<int64_t>(inputs.size()) > lim) inputs.resize(lim);
            if (static_cast<int64_t>(targets.size()) > lim) targets.resize(lim);
        }

        inputs_lst.push_back(torch::tensor(inputs, torch::kLong).to(device));
        targets_lst.push_back(torch::tensor(targets, torch::kLong).to(device));
    }
    return {torch::stack(inputs_lst), torch::stack(targets_lst)};
}

// InstructionLoader：等价 DataLoader(dataset, batch_size, collate_fn=custom_collate_fn,
//                                   shuffle=..., drop_last=..., num_workers=0)
class InstructionLoader {
public:
    InstructionLoader(InstructionDataset& ds, int64_t batch_size, bool shuffle,
                      bool drop_last, int64_t pad_token_id = 50256,
                      c10::optional<int64_t> allowed_max_length = c10::nullopt,
                      const torch::Device& device = torch::Device(torch::kCPU),
                      uint64_t seed = 123) {
        size_t n = ds.size();
        if (drop_last) n = n / batch_size * batch_size;
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(idx.begin(), idx.end(), rng);
        }
        for (size_t b = 0; b < n; b += batch_size) {
            std::vector<std::vector<int>> batch_vec;
            for (size_t k = b; k < std::min(b + static_cast<size_t>(batch_size), n); ++k)
                batch_vec.push_back(ds.encoded(idx[k]));
            auto [inputs, targets] =
                custom_collate_fn(batch_vec, pad_token_id, /*ignore_index=*/-100,
                                  allowed_max_length, device);
            batches_.push_back({inputs, targets});
        }
    }

    size_t num_batches() const { return batches_.size(); }
    const std::vector<InstructionBatch>& batches() const { return batches_; }

private:
    std::vector<InstructionBatch> batches_;
};

// ==========================================================================
// 7.5/7.7 从生成文本中抽取模型回复
// ==========================================================================
inline std::string extract_response(const std::string& generated_text,
                                    const std::string& input_text) {
    std::string response = generated_text.substr(input_text.size());
    // 移除 "### Response:" 并去除首尾空白
    const std::string marker = "### Response:";
    auto pos = response.find(marker);
    if (pos != std::string::npos) response = response.substr(pos + marker.size());
    auto trim = [](std::string s) {
        size_t b = s.find_first_not_of(" \t\n\r\f\v");
        if (b == std::string::npos) return std::string();
        size_t e = s.find_last_not_of(" \t\n\r\f\v");
        return s.substr(b, e - b + 1);
    };
    return trim(response);
}

}  // namespace ch7
