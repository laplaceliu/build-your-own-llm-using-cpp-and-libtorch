// training.h
// 第 5 章：在无标签数据上进行预训练 —— 训练/评估/解码函数
//
// 对应书中代码：
//   text_to_token_ids / token_ids_to_text   代码清单 5-1
//   calc_loss_batch / calc_loss_loader      代码清单 5-2
//   train_model_simple / evaluate_model / generate_and_print_sample  代码清单 5-3
//   generate（temperature + top_k + eos）   代码清单 5-4
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "gpt.h"

namespace ch5 {

// ---- 代码清单 5-1：文本 <-> 词元 ID ----
inline torch::Tensor text_to_token_ids(const std::string& text,
                                       ch2::BpeTokenizer& tokenizer) {
    auto encoded = tokenizer.encode(text, {"<|endoftext|>"});
    return torch::tensor(encoded, torch::kLong).unsqueeze(0);
}

inline std::string token_ids_to_text(const torch::Tensor& token_ids,
                                     ch2::BpeTokenizer& tokenizer) {
    auto flat = token_ids.squeeze(0);
    std::vector<int> ids;
    ids.reserve(flat.size(0));
    for (int64_t i = 0; i < flat.size(0); ++i) {
        ids.push_back(flat.index({i}).item<int>());
    }
    return tokenizer.decode(ids);
}

// ---- calc_loss_batch：单个批次的交叉熵损失 ----
inline torch::Tensor calc_loss_batch(const torch::Tensor& input_batch,
                                     const torch::Tensor& target_batch,
                                     ch4::GPTModel& model) {
    auto logits = model->forward(input_batch);
    auto loss = torch::nn::functional::cross_entropy(
        logits.flatten(0, 1), target_batch.flatten());
    return loss;
}

// ---- calc_loss_loader：遍历批次求平均损失（代码清单 5-2）----
inline double calc_loss_loader(ch4::GPTModel& model,
                               const std::vector<ch2::GPTBatch>& loader,
                               c10::optional<int64_t> num_batches = c10::nullopt) {
    double total_loss = 0.0;
    if (loader.empty()) return std::numeric_limits<double>::quiet_NaN();
    int64_t n = num_batches.has_value()
                    ? std::min(num_batches.value(), static_cast<int64_t>(loader.size()))
                    : static_cast<int64_t>(loader.size());
    torch::NoGradGuard no_grad;
    model->eval();
    for (int64_t i = 0; i < n; ++i) {
        auto loss = calc_loss_batch(loader[i].inputs, loader[i].targets, model);
        total_loss += loss.item<double>();
    }
    model->train();
    return total_loss / n;
}

// ---- evaluate_model：训练/验证损失（对应代码清单 5-3 中的 evaluate_model）----
inline std::pair<double, double> evaluate_model(
    ch4::GPTModel& model, const std::vector<ch2::GPTBatch>& train_loader,
    const std::vector<ch2::GPTBatch>& val_loader, int64_t eval_iter) {
    model->eval();
    double train_loss = 0.0;
    {
        torch::NoGradGuard no_grad;
        int64_t n = std::min(eval_iter, static_cast<int64_t>(train_loader.size()));
        for (int64_t i = 0; i < n; ++i) {
            train_loss += calc_loss_batch(train_loader[i].inputs, train_loader[i].targets, model)
                              .item<double>();
        }
        train_loss /= n;

        double val_loss = 0.0;
        n = std::min(eval_iter, static_cast<int64_t>(val_loader.size()));
        for (int64_t i = 0; i < n; ++i) {
            val_loss += calc_loss_batch(val_loader[i].inputs, val_loader[i].targets, model)
                            .item<double>();
        }
        val_loss /= n;
        model->train();
        return {train_loss, val_loss};
    }
}

// ---- generate_and_print_sample：生成 50 词元文本样本 ----
inline void generate_and_print_sample(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                                      const std::string& start_context) {
    model->eval();
    int64_t context_size = model->pos_emb->weight.size(0);
    auto encoded = text_to_token_ids(start_context, tokenizer);
    torch::Tensor token_ids;
    {
        torch::NoGradGuard no_grad;
        token_ids = ch4::generate_text_simple(model, encoded, /*max_new_tokens=*/50,
                                              context_size);
    }
    auto decoded = token_ids_to_text(token_ids, tokenizer);
    std::string one_line;
    for (char c : decoded) one_line += (c == '\n') ? ' ' : c;
    std::cout << one_line << "\n";
    model->train();
}

// ---- train_model_simple（代码清单 5-3）----
inline void train_model_simple(ch4::GPTModel& model,
                               const std::vector<ch2::GPTBatch>& train_loader,
                               const std::vector<ch2::GPTBatch>& val_loader,
                               torch::optim::AdamW& optimizer, int64_t num_epochs,
                               int64_t eval_freq, int64_t eval_iter,
                               const std::string& start_context,
                               ch2::BpeTokenizer& tokenizer) {
    int64_t tokens_seen = 0;
    int64_t global_step = -1;

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        for (const auto& batch : train_loader) {
            optimizer.zero_grad();
            auto loss = calc_loss_batch(batch.inputs, batch.targets, model);
            loss.backward();
            optimizer.step();
            tokens_seen += batch.inputs.numel();
            global_step += 1;

            if (global_step % eval_freq == 0) {
                auto [train_loss, val_loss] =
                    evaluate_model(model, train_loader, val_loader, eval_iter);
                std::cout << "Ep " << (epoch + 1) << " (Step " << global_step
                          << "): Train loss " << train_loss << ", Val loss " << val_loss
                          << "\n";
            }
        }
        generate_and_print_sample(model, tokenizer, start_context);
    }
}

// ==========================================================================
// 5.3.3 代码清单 5-4：带 temperature / top_k / eos_id 的 generate 函数
// ==========================================================================
inline torch::Tensor generate(ch4::GPTModel& model, const torch::Tensor& idx,
                              int64_t max_new_tokens, int64_t context_size,
                              double temperature = 0.0,
                              c10::optional<int64_t> top_k = c10::nullopt,
                              c10::optional<int64_t> eos_id = c10::nullopt) {
    torch::Tensor cur = idx;
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        auto idx_cond =
            cur.index({torch::indexing::Slice(),
                       torch::indexing::Slice(-context_size, torch::indexing::None)});
        torch::Tensor logits;
        {
            torch::NoGradGuard no_grad;
            logits = model->forward(idx_cond);
        }
        logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

        if (top_k.has_value()) {
            auto k = top_k.value();
            auto topk = torch::topk(logits, k);
            auto min_val = std::get<1>(topk).narrow(1, k - 1, 1);  // topk 末位是 min
            // torch::topk 返回 (values, indices)，values[:, -1] 即 min_val
            auto top_values = std::get<0>(topk);
            auto threshold = top_values.narrow(1, k - 1, 1);
            logits = torch::where(logits < threshold,
                                  torch::full_like(logits, -std::numeric_limits<double>::infinity()),
                                  logits);
            (void)min_val;
        }

        torch::Tensor idx_next;
        if (temperature > 0.0) {
            auto logits_scaled = logits / temperature;
            auto probs = torch::softmax(logits_scaled, -1);
            idx_next = torch::multinomial(probs, /*num_samples=*/1);
        } else {
            idx_next = torch::argmax(logits, -1, /*keepdim=*/true);
        }

        if (eos_id.has_value() &&
            idx_next.item<int64_t>() == eos_id.value()) {
            break;
        }
        cur = torch::cat({cur, idx_next}, 1);
    }
    return cur;
}

}  // namespace ch5
