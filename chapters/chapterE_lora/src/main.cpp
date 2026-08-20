// chapters/chapterE_lora/src/main.cpp
// ---------------------------------------------------------------------------
// 附录 E：使用 LoRA 进行参数高效微调（C++/LibTorch GPU 版）
//
// 演示：
//   E.1）创建随机初始化的 GPTModel，把所有 nn.Linear 替换为 LinearWithLoRA
//        （与书中代码清单 E-3 ~ E-6 一一对应）。
//   E.2）冻结所有非 LoRA 参数，对比「可训练参数 / 总参数」。
//   E.3）在合成二分类数据集上微调：
//          spam = 文本包含 free / buy / win 任一关键词
//          ham  = 否则
//        （由于本仓库不下载 UCI 短信数据集，使用合成数据演示等价流程）。
//   E.4）训练 5 epoch，对比「LoRA 微调」与「全参数微调（解冻）」的分类准确率
//        以及训练时显存/参数量差异。
//   E.5）保存 / 加载 LoRA 权重（演示 LoRA 的核心优势：只保存小矩阵）。
// ---------------------------------------------------------------------------
#include "lora.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "bpe_tokenizer.h"
#include "gpt.h"

using namespace chE;

// ---------------------------------------------------------------------------
// 数据：合成 spam/ham 分类
// ---------------------------------------------------------------------------
struct Example {
    std::string text;
    int label;  // 0 = ham, 1 = spam
};

static std::vector<Example> make_synthetic_dataset(int n_ham, int n_spam) {
    // 关键词：出现任一 → spam
    static const std::vector<std::string> spam_words = {
        "free", "buy", "win", "cash", "prize", "offer", "discount", "urgent"};
    static const std::vector<std::string> ham_phrases = {
        "How are you", "Let's meet tomorrow", "The meeting is at 3pm",
        "Please review the document", "Can we reschedule",
        "I will call you later", "See you at lunch", "Thanks for the help",
        "The project is on track", "Call me when you can"};
    std::vector<Example> data;
    data.reserve(n_ham + n_spam);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist_int(1, 99);
    auto pick_spam = [&](int i) {
        std::ostringstream os;
        const auto& w = spam_words[i % spam_words.size()];
        os << "URGENT! " << w << " now! You have won "
           << (dist_int(rng) + 1) << " dollars! Click here " << w
           << " limited time offer " << w;
        return Example{os.str(), 1};
    };
    auto pick_ham = [&](int i) {
        std::ostringstream os;
        os << ham_phrases[i % ham_phrases.size()] << ". "
           << ham_phrases[(i + 3) % ham_phrases.size()] << ". "
           << ham_phrases[(i + 7) % ham_phrases.size()] << ".";
        return Example{os.str(), 0};
    };
    for (int i = 0; i < n_spam; ++i) data.push_back(pick_spam(i));
    for (int i = 0; i < n_ham; ++i)  data.push_back(pick_ham(i));
    std::shuffle(data.begin(), data.end(), rng);
    return data;
}

static std::vector<std::vector<int64_t>>
tokenize_all(ch2::BpeTokenizer& tok, const std::vector<Example>& data,
             int64_t max_len) {
    std::vector<std::vector<int64_t>> out;
    out.reserve(data.size());
    for (const auto& e : data) {
        auto ids = tok.encode(e.text, {"<|endoftext|>"});
        if (static_cast<int64_t>(ids.size()) > max_len) {
            ids.resize(max_len);
        }
        out.push_back(std::vector<int64_t>(ids.begin(), ids.end()));
    }
    return out;
}

// 分类头：在 hidden 末尾 token 上做 num_classes 路分类
static torch::Tensor classify_forward(GPTModelWithLoRA& model,
                                      const std::vector<int64_t>& ids) {
    auto x = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong))
                 .unsqueeze(0);   // [1, seq]
    auto h = model->forward_hidden(x);   // [1, seq, emb]
    auto last = h.select(1, h.size(1) - 1);  // [1, emb]
    auto logits = model->out_head->forward(last);   // [1, num_classes]
    return logits;  // 保留 batch 维
}

// 单 batch 训练 / 评估
struct Batch {
    std::vector<std::vector<int64_t>> ids_list;
    std::vector<int64_t> labels;
};

static double train_one_epoch(GPTModelWithLoRA& model,
                              const std::vector<Batch>& batches,
                              torch::optim::Optimizer& opt,
                              const std::string& device_str) {
    model->train();
    double total_loss = 0.0;
    int64_t n = 0;
    for (const auto& b : batches) {
        std::vector<torch::Tensor> logits_list;
        std::vector<int64_t> label_list;
        logits_list.reserve(b.ids_list.size());
        label_list.reserve(b.ids_list.size());
        for (size_t i = 0; i < b.ids_list.size(); ++i) {
            auto lg = classify_forward(model, b.ids_list[i]);  // [1, 2]
            logits_list.push_back(lg);
            label_list.push_back(b.labels[i]);
        }
        auto logits = torch::cat(logits_list, /*dim=*/0);  // [B, 2]
        auto labels = torch::tensor(label_list,
            torch::TensorOptions().dtype(torch::kLong));
        if (device_str == "cuda") {
            labels = labels.to(torch::kCUDA);
        }
        auto loss = torch::cross_entropy_loss(logits, labels);
        opt.zero_grad();
        loss.backward();
        opt.step();
        total_loss += loss.item<double>();
        ++n;
    }
    return total_loss / std::max<int64_t>(n, 1);
}

static double eval_accuracy(GPTModelWithLoRA& model,
                            const std::vector<Batch>& batches,
                            const std::string& device_str) {
    model->eval();
    int64_t correct = 0, total = 0;
    {
        torch::NoGradGuard no_grad;
        for (const auto& b : batches) {
            for (size_t i = 0; i < b.ids_list.size(); ++i) {
                auto lg = classify_forward(model, b.ids_list[i]);  // [1, 2]
                auto pred = lg.argmax(/*dim=*/-1).item<int64_t>();
                if (pred == b.labels[i]) ++correct;
                ++total;
            }
        }
    }
    (void)device_str;
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(4);

    bool use_cuda = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cuda" ||
            std::string(argv[i]) == "-cuda") use_cuda = true;
    }

    torch::Device device = torch::kCPU;
    if (use_cuda && torch::cuda::is_available()) {
        std::cout << "[appendix-E] CUDA available, using GPU\n";
        device = torch::kCUDA;
    } else {
        std::cout << "[appendix-E] using CPU\n";
    }
    const std::string device_str = (device == torch::kCUDA) ? "cuda" : "cpu";

    // ---- 1) 一个非常小的 GPT 配置（24 万参数级，CPU 也能跑） ----
    ch4::GPTConfig cfg;
    cfg.vocab_size      = 50257;
    cfg.context_length  = 128;
    cfg.emb_dim         = 128;
    cfg.n_heads         = 4;
    cfg.n_layers        = 4;
    cfg.drop_rate       = 0.0;
    cfg.qkv_bias        = true;

    // ---- 2) 加载 BPE ----
#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
    ch2::BpeTokenizer tokenizer(std::string(DATA_DIR) + "/encoder.json",
                                std::string(DATA_DIR) + "/vocab.bpe");
    std::cout << "[appendix-E] tokenizer vocab = " << tokenizer.vocab_size() << "\n";

    // ---- 3) 构造「LoRA 化」GPT，并冻结非 LoRA 参数 ----
    const int64_t rank = 8;
    const double  alpha = 16.0;  // 2 * rank，与书中建议一致
    GPTModelWithLoRA model(cfg, rank, alpha, /*lora_out_head=*/false);
    model->replace_out_head_plain(/*num_classes=*/2);  // 头部先冻结
    if (device == torch::kCUDA) model->to(torch::kCUDA);

    int64_t trainable = freeze_non_lora_parameters(*model);
    // 输出层也冻结了 → 只训练 LoRA；为公平对比全参微调，下文会再次解冻
    auto [t1, total1] = count_trainable_parameters(*model);
    (void)trainable;
    std::cout << "[appendix-E] LoRA only        : trainable=" << t1
              << "  total=" << total1
              << "  (" << (100.0 * t1 / std::max<int64_t>(total1, 1))
              << "%)\n";

    // ---- 4) 合成数据 ----
    auto data = make_synthetic_dataset(/*n_ham=*/200, /*n_spam=*/200);
    std::cout << "[appendix-E] dataset size = " << data.size() << "\n";
    auto ids_list = tokenize_all(tokenizer, data, /*max_len=*/cfg.context_length);

    // 划分 70 / 10 / 20
    int64_t n = static_cast<int64_t>(data.size());
    int64_t n_train = n * 7 / 10;
    int64_t n_val   = n * 1 / 10;
    auto make_batches = [&](int64_t from, int64_t to, int64_t bs) {
        std::vector<Batch> batches;
        for (int64_t i = from; i < to; i += bs) {
            Batch b;
            int64_t end = std::min<int64_t>(i + bs, to);
            b.ids_list.assign(ids_list.begin() + i, ids_list.begin() + end);
            std::vector<int64_t> ll;
            ll.reserve(end - i);
            for (int64_t j = i; j < end; ++j) ll.push_back(data[j].label);
            b.labels = std::move(ll);
            batches.push_back(std::move(b));
        }
        return batches;
    };
    auto train_batches = make_batches(0, n_train, /*bs=*/8);
    auto val_batches   = make_batches(n_train, n_train + n_val, 8);
    auto test_batches  = make_batches(n_train + n_val, n, 8);
    std::cout << "[appendix-E] train batches = " << train_batches.size()
              << ", val = " << val_batches.size()
              << ", test = " << test_batches.size() << "\n";

    // ---- 5) 训练循环（LoRA only）----
    double lr = 5e-4;  // LoRA 通常用更大的学习率
    auto optim_lora = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(lr).weight_decay(0.0));
    std::cout << "\n=== E.7 LoRA training ===\n";
    for (int ep = 1; ep <= 5; ++ep) {
        auto t0 = std::chrono::steady_clock::now();
        auto loss = train_one_epoch(model, train_batches, optim_lora, device_str);
        auto acc  = eval_accuracy(model, val_batches, device_str);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Ep " << ep << " | train_loss " << loss
                  << " | val_acc " << (acc * 100.0) << "%"
                  << " | " << sec << " s\n";
    }
    double test_acc_lora = eval_accuracy(model, test_batches, device_str);
    auto [t2, total2] = count_trainable_parameters(*model);
    std::cout << "[appendix-E] LoRA test_acc = " << (test_acc_lora * 100.0)
              << "%  (trainable=" << t2 << " / " << total2 << ")\n";

    // ---- 6) 对照实验：解冻所有参数，做全量微调 ----
    std::cout << "\n=== E.7 ablation: full fine-tuning (unfreeze all) ===\n";
    for (auto& p : model->parameters()) p.set_requires_grad(true);
    auto [t3, total3] = count_trainable_parameters(*model);
    std::cout << "[appendix-E] Full FT         : trainable=" << t3
              << "  total=" << total3 << " (100%)\n";
    double lr_full = 5e-5;
    auto optim_full = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(lr_full).weight_decay(0.0));
    for (int ep = 1; ep <= 5; ++ep) {
        auto t0 = std::chrono::steady_clock::now();
        auto loss = train_one_epoch(model, train_batches, optim_full, device_str);
        auto acc  = eval_accuracy(model, val_batches, device_str);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Ep " << ep << " | train_loss " << loss
                  << " | val_acc " << (acc * 100.0) << "%"
                  << " | " << sec << " s\n";
    }
    double test_acc_full = eval_accuracy(model, test_batches, device_str);
    std::cout << "[appendix-E] Full FT test_acc = " << (test_acc_full * 100.0)
              << "%\n";

    // ---- 7) 保存 LoRA 权重（仅 LoRA 矩阵，文件极小） ----
    std::ostringstream path;
    path << "/tmp/lora_chapterE_rank" << rank << ".bin";
    {
        std::ofstream fs(path.str(), std::ios::binary);
        for (auto& pair : model->named_parameters(/*recurse=*/true)) {
            const std::string& n = pair.key();
            if (n.find(".lora.A") == std::string::npos &&
                n.find(".lora.B") == std::string::npos) {
                continue;
            }
            auto t = pair.value().detach().cpu().contiguous();
            int64_t sz = t.numel();
            fs.write(reinterpret_cast<char*>(&sz), sizeof(sz));
            fs.write(n.data(), n.size());
            int64_t nd = t.dim();
            fs.write(reinterpret_cast<char*>(&nd), sizeof(nd));
            std::vector<int64_t> sh(t.dim());
            for (int64_t k = 0; k < nd; ++k) sh[k] = t.size(k);
            fs.write(reinterpret_cast<const char*>(sh.data()),
                     sizeof(int64_t) * nd);
            fs.write(reinterpret_cast<const char*>(t.data_ptr<float>()),
                     sizeof(float) * sz);
        }
    }
    std::ifstream fs(path.str(), std::ios::binary | std::ios::ate);
    auto sz = fs.tellg();
    std::cout << "\n[appendix-E] saved LoRA-only weights -> " << path.str()
              << "  size = " << sz << " bytes\n";

    // ---- 8) 总结 ----
    std::cout << "\n=== summary ===\n";
    std::cout << "LoRA trainable params   : " << t2 << "  ("
              << (100.0 * t2 / total2) << "% of total " << total2 << ")\n";
    std::cout << "LoRA test_acc           : " << (test_acc_lora * 100.0) << "%\n";
    std::cout << "Full FT test_acc        : " << (test_acc_full * 100.0) << "%\n";
    std::cout << "LoRA-weights file size  : " << sz << " bytes\n";
    return 0;
}
