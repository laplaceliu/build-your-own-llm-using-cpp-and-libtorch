// finetuning.h
// 第 6 章：针对分类的微调 —— 数据集/分类头/评估/训练/推理函数
//
// 对应书中代码：
//   6.2  create_balanced_dataset / random_split（代码清单 6-2/6-3）
//   6.3  SpamDataset（代码清单 6-4）+ DataLoader（代码清单 6-5）
//   6.5  GPTClassifier：替换 out_head + 冻结参数（代码清单 6-7）
//   6.6  calc_accuracy_loader（代码清单 6-8）/ calc_loss_loader（代码清单 6-9）
//   6.7  train_classifier_simple / evaluate_model（代码清单 6-10）
//   6.8  classify_review（代码清单 6-12）
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"
#include "gpt.h"

namespace ch6 {

// ==========================================================================
// 6.2 数据集准备
// ==========================================================================
struct SpamItem {
    int label;        // 0 = ham, 1 = spam
    std::string text;
};

// 读取 SMSSpamCollection.tsv（制表符分隔：label<TAB>text）
inline std::vector<SpamItem> read_spam_tsv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("无法打开数据集: " + path);
    std::vector<SpamItem> items;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string label = line.substr(0, tab);
        std::string text = line.substr(tab + 1);
        items.push_back({label == "spam" ? 1 : 0, text});
    }
    return items;
}

// 平衡数据集：取全部 spam，从 ham 中随机采样等量（seed 123）
// （等价代码清单 6-2 的 df[ham].sample(num_spam, random_state=123)）
inline std::vector<SpamItem> create_balanced_dataset(const std::vector<SpamItem>& data,
                                                    uint64_t seed = 123) {
    std::vector<SpamItem> ham, spam;
    for (const auto& it : data) (it.label == 1 ? spam : ham).push_back(it);

    std::mt19937 rng(seed);
    std::shuffle(ham.begin(), ham.end(), rng);
    ham.resize(spam.size());  // 只保留与 spam 数量相等的 ham 子集

    std::vector<SpamItem> balanced = ham;
    balanced.insert(balanced.end(), spam.begin(), spam.end());
    return balanced;
}

// 随机划分 70/10/20（seed 123），返回 train/val/test（代码清单 6-3）
struct SpamSplit {
    std::vector<SpamItem> train, val, test;
};

inline SpamSplit random_split(const std::vector<SpamItem>& data,
                              double train_frac, double validation_frac,
                              uint64_t seed = 123) {
    std::vector<SpamItem> shuffled = data;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    size_t train_end = static_cast<size_t>(shuffled.size() * train_frac);
    size_t validation_end = train_end + static_cast<size_t>(shuffled.size() * validation_frac);
    return {std::vector<SpamItem>(shuffled.begin(), shuffled.begin() + train_end),
            std::vector<SpamItem>(shuffled.begin() + train_end, shuffled.begin() + validation_end),
            std::vector<SpamItem>(shuffled.begin() + validation_end, shuffled.end())};
}

inline void save_csv(const std::string& path, const std::vector<SpamItem>& items) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("无法写入 CSV: " + path);
    for (const auto& it : items) {
        // 简单转义：文本中的引号翻倍、逗号用引号包裹
        std::string text = it.text;
        std::string escaped;
        bool need_quote = text.find(',') != std::string::npos ||
                          text.find('"') != std::string::npos;
        if (need_quote) {
            for (char c : text) {
                if (c == '"') escaped += "\"\"";
                else escaped += c;
            }
            escaped = "\"" + escaped + "\"";
        } else {
            escaped = text;
        }
        ofs << (it.label == 1 ? "spam" : "ham") << "," << escaped << "\n";
    }
}

// 读取准备阶段的 CSV（label,text）
inline std::vector<SpamItem> read_csv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("无法打开 CSV: " + path);
    std::vector<SpamItem> items;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string label = line.substr(0, comma);
        std::string text = line.substr(comma + 1);
        // 去掉包裹引号
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.size() - 2);
            std::string unescaped;
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] == '"' && i + 1 < text.size() && text[i + 1] == '"') {
                    unescaped += '"'; ++i;
                } else {
                    unescaped += text[i];
                }
            }
            text = unescaped;
        }
        items.push_back({label == "spam" ? 1 : 0, text});
    }
    return items;
}

// ==========================================================================
// 6.3 SpamDataset（代码清单 6-4）：编码 + 填充/截断到 max_length
// ==========================================================================
class SpamDataset {
public:
    SpamDataset(const std::string& csv_path, ch2::BpeTokenizer& tokenizer,
                int64_t max_length = -1, int64_t pad_token_id = 50256)
        : pad_token_id_(pad_token_id) {
        data_ = read_csv(csv_path);
        for (const auto& it : data_) {
            encoded_texts_.push_back(tokenizer.encode(it.text));
        }

        if (max_length < 0) {
            max_length_ = longest_encoded_length();
        } else {
            max_length_ = max_length;
            // 截断超长序列
            for (auto& e : encoded_texts_) {
                if (static_cast<int64_t>(e.size()) > max_length_) e.resize(max_length_);
            }
        }
        // 填充到 max_length
        for (auto& e : encoded_texts_) {
            e.resize(static_cast<size_t>(max_length_), pad_token_id_);
        }
    }

    int64_t max_length() const { return max_length_; }
    size_t size() const { return data_.size(); }
    const std::vector<int>& encoded(int64_t i) const { return encoded_texts_[i]; }
    int64_t label(int64_t i) const { return data_[i].label; }

private:
    int64_t longest_encoded_length() const {
        int64_t m = 0;
        for (const auto& e : encoded_texts_)
            m = std::max(m, static_cast<int64_t>(e.size()));
        return m;
    }

    std::vector<SpamItem> data_;
    std::vector<std::vector<int>> encoded_texts_;
    int64_t max_length_ = 0;
    int64_t pad_token_id_ = 50256;
};

// DataLoader（代码清单 6-5 的等价）：生成 (inputs, labels) 批次
struct SpamBatch {
    torch::Tensor inputs;  // [batch, max_length]
    torch::Tensor labels;  // [batch]
};

inline std::vector<SpamBatch> make_spam_loader(SpamDataset& ds, int64_t batch_size,
                                               bool shuffle, bool drop_last,
                                               uint64_t seed = 0) {
    size_t n = ds.size();
    if (drop_last) n = n / batch_size * batch_size;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    std::vector<SpamBatch> batches;
    for (size_t b = 0; b < n; b += batch_size) {
        std::vector<torch::Tensor> in_ts;
        std::vector<torch::Tensor> lb_ts;
        for (size_t k = b; k < std::min(b + static_cast<size_t>(batch_size), n); ++k) {
            auto enc = ds.encoded(idx[k]);
            in_ts.push_back(torch::tensor(enc, torch::kLong));
            lb_ts.push_back(torch::tensor(ds.label(idx[k]), torch::kLong));
        }
        batches.push_back({torch::stack(in_ts), torch::stack(lb_ts)});
    }
    return batches;
}

// ==========================================================================
// 6.5 GPTClassifier：复用 GPTModel，替换输出层，支持冻结/解冻
// ==========================================================================
class GPTClassifierImpl : public torch::nn::Module {
public:
    GPTClassifierImpl(ch4::GPTModel& base, int64_t num_classes) {
        // 复用 GPTModel（不复制权重，引用同一模块）
        base_ = base;
        register_module("base", base_);
        // 新输出层：emb_dim -> num_classes（在 manual_seed(123) 后构造，与书一致）
        int64_t emb_dim = base_->out_head->weight.size(1);
        out_head_ = register_module("out_head",
            torch::nn::Linear(emb_dim, num_classes));

        // 冻结所有参数
        for (auto& np : base_->named_parameters()) {
            np.value().set_requires_grad(false);
        }
        // 解冻最后一个 Transformer 块与 final_norm（与书一致）
        auto n = base_->trf_blocks->size();
        auto& last = base_->trf_blocks->at<ch4::TransformerBlockImpl>(n - 1);
        for (auto& np : last.named_parameters()) {
            np.value().set_requires_grad(true);
        }
        for (auto& np : base_->final_norm->named_parameters()) {
            np.value().set_requires_grad(true);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto hidden = base_->forward_hidden(x);          // [b, n, d]
        return out_head_->forward(hidden);               // [b, n, num_classes]
    }

    // 可训练参数（供优化器使用）
    std::vector<torch::Tensor> trainable_parameters() const {
        std::vector<torch::Tensor> out;
        for (auto& np : named_parameters()) {
            if (np.value().requires_grad()) out.push_back(np.value());
        }
        return out;
    }

    // 解冻全部参数（练习 6.2）
    void unfreeze_all() {
        for (auto& np : named_parameters()) np.value().set_requires_grad(true);
    }

    ch4::GPTModel base_{nullptr};
    torch::nn::Linear out_head_{nullptr};
};
TORCH_MODULE(GPTClassifier);

// ==========================================================================
// 6.6 分类损失与准确率
// ==========================================================================

// calc_loss_batch：只关注最后一个输出词元（代码清单 6-10 内的调整）
inline torch::Tensor calc_loss_batch(const torch::Tensor& input_batch,
                                     const torch::Tensor& target_batch,
                                     GPTClassifier& model) {
    auto logits = model->forward(input_batch).index(
        {torch::indexing::Slice(), -1, torch::indexing::Slice()});  // [b, num_classes]
    // target_batch 移到与 logits 相同的设备（输入可能是 CUDA 张量）
    return torch::nn::functional::cross_entropy(
        logits, target_batch.to(logits.device()));
}

inline double calc_loss_loader(GPTClassifier& model,
                               const std::vector<SpamBatch>& loader,
                               c10::optional<int64_t> num_batches = c10::nullopt) {
    if (loader.empty()) return std::numeric_limits<double>::quiet_NaN();
    int64_t n = num_batches.has_value()
                    ? std::min(num_batches.value(), static_cast<int64_t>(loader.size()))
                    : static_cast<int64_t>(loader.size());
    double total = 0.0;
    torch::NoGradGuard g;
    model->eval();
    for (int64_t i = 0; i < n; ++i) {
        total += calc_loss_batch(loader[i].inputs, loader[i].labels, model).item<double>();
    }
    model->train();
    return total / n;
}

// calc_accuracy_loader（代码清单 6-8）
inline double calc_accuracy_loader(GPTClassifier& model,
                                   const std::vector<SpamBatch>& loader,
                                   c10::optional<int64_t> num_batches = c10::nullopt) {
    int64_t n = num_batches.has_value()
                    ? std::min(num_batches.value(), static_cast<int64_t>(loader.size()))
                    : static_cast<int64_t>(loader.size());
    int64_t correct = 0, total = 0;
    torch::NoGradGuard g;
    model->eval();
    for (int64_t i = 0; i < n; ++i) {
        auto logits = model->forward(loader[i].inputs).index(
            {torch::indexing::Slice(), -1, torch::indexing::Slice()});
        auto predicted = torch::argmax(logits, -1).to(torch::kCPU);  // GPU 上比较需回 CPU
        auto labels = loader[i].labels.to(torch::kCPU);
        correct += (predicted == labels).sum().item<int64_t>();
        total += predicted.size(0);
    }
    model->train();
    return total > 0 ? static_cast<double>(correct) / total : 0.0;
}

// evaluate_model（代码清单 6-10 中与预训练相同的版本）
inline std::pair<double, double> evaluate_model(
    GPTClassifier& model, const std::vector<SpamBatch>& train_loader,
    const std::vector<SpamBatch>& val_loader, int64_t eval_iter) {
    double train_loss = calc_loss_loader(model, train_loader, eval_iter);
    double val_loss = calc_loss_loader(model, val_loader, eval_iter);
    return {train_loss, val_loss};
}

// ==========================================================================
// 6.7 训练分类器（代码清单 6-10）
// ==========================================================================
inline void train_classifier_simple(GPTClassifier& model,
                                    const std::vector<SpamBatch>& train_loader,
                                    const std::vector<SpamBatch>& val_loader,
                                    torch::optim::AdamW& optimizer,
                                    int64_t num_epochs, int64_t eval_freq,
                                    int64_t eval_iter) {
    int64_t examples_seen = 0;
    int64_t global_step = -1;
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        for (const auto& batch : train_loader) {
            optimizer.zero_grad();
            auto loss = calc_loss_batch(batch.inputs, batch.labels, model);
            loss.backward();
            optimizer.step();
            examples_seen += batch.inputs.size(0);
            global_step += 1;

            if (global_step % eval_freq == 0) {
                auto [tl, vl] = evaluate_model(model, train_loader, val_loader, eval_iter);
                std::cout << "Ep " << (epoch + 1) << " (Step " << global_step << "): "
                          << "Train loss " << tl << ", Val loss " << vl << "\n";
            }
        }
        double train_acc = calc_accuracy_loader(model, train_loader, eval_iter);
        double val_acc = calc_accuracy_loader(model, val_loader, eval_iter);
        std::cout << "Training accuracy: " << (train_acc * 100.0) << "% | "
                  << "Validation accuracy: " << (val_acc * 100.0) << "%\n";
    }
}

// ==========================================================================
// 6.8 classify_review（代码清单 6-12）
// ==========================================================================
inline std::string classify_review(const std::string& text, GPTClassifier& model,
                                   ch2::BpeTokenizer& tokenizer,
                                   int64_t max_length, int64_t pad_token_id = 50256) {
    model->eval();
    auto input_ids = tokenizer.encode(text);
    // 截断到 max_length（或模型上下文长度，取较小者）
    int64_t supported = model->base_->pos_emb->weight.size(0);
    int64_t limit = std::min(max_length, supported);
    if (static_cast<int64_t>(input_ids.size()) > limit) input_ids.resize(limit);
    // 填充
    while (static_cast<int64_t>(input_ids.size()) < max_length) {
        input_ids.push_back(pad_token_id);
    }
    auto input_tensor = torch::tensor(input_ids, torch::kLong)
                            .to(model->base_->pos_emb->weight.device())  // 与模型同设备
                            .unsqueeze(0);
    torch::Tensor logits;
    {
        torch::NoGradGuard g;
        logits = model->forward(input_tensor).index(
            {torch::indexing::Slice(), -1, torch::indexing::Slice()});
    }
    int64_t predicted = torch::argmax(logits).to(torch::kCPU).item<int64_t>();
    model->train();
    return predicted == 1 ? "spam" : "not spam";
}

}  // namespace ch6
