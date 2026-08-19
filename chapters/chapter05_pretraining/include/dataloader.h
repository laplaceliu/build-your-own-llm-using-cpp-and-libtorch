// dataloader.h
// 第 2 章 2.6 节的滑动窗口数据加载器（GPTDatasetV1 + DataLoader 的 C++ 等价），
// 第 5 章预训练复用。支持 shuffle（等价 create_dataloader_v1 的 shuffle 选项）。
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace ch2 {

struct GPTBatch {
    torch::Tensor inputs;   // (batch, max_length)
    torch::Tensor targets;  // (batch, max_length)
};

class GPTDataLoader {
public:
    GPTDataLoader(const std::vector<int>& token_ids, int batch_size, int max_length,
                  int stride, bool shuffle, bool drop_last, uint64_t seed = 0) {
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

}  // namespace ch2
