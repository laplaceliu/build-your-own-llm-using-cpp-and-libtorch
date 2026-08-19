// main.cpp
// 第 5 章：在无标签数据上进行预训练（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印：
//   5.1  评估文本生成模型（生成文本 / 交叉熵损失 / 困惑度 / 训练验证集损失）
//   5.2  训练大语言模型（AdamW 预训练 10 轮）
//   5.3  控制随机性的解码策略（温度缩放 / Top-k / multinomial）
//   5.4  使用 PyTorch 保存和加载模型权重
//   5.5  从 OpenAI 加载预训练权重（safetensors）并生成文本
//
// 用法：
//   ./chapter05_pretraining [data_dir] [epochs] [openai_safetensors]
//     data_dir            默认 data/（第 2 章数据目录，含 the-verdict.txt）
//     epochs              默认 10（训练轮数；设 0 跳过训练）
//     openai_safetensors  GPT-2 权重文件（默认 data/gpt2-model.safetensors；
//                         不存在时跳过 5.5）
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"
#include "dataloader.h"
#include "gpt.h"
#include "safetensors.h"
#include "training.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

namespace fs = std::filesystem;

namespace {

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("无法打开文件: " + path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// ==========================================================================
// 5.1.1 使用 GPT 生成文本
// ==========================================================================
void demo_generate_text(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer) {
    section("5.1.1 使用 GPT 生成文本");
    const std::string start_context = "Every effort moves you";
    auto encoded = ch5::text_to_token_ids(start_context, tokenizer);
    std::cout << "encoded shape: " << encoded.sizes() << "\n";

    auto token_ids = ch4::generate_text_simple(model, encoded, /*max_new_tokens=*/10,
                                               /*context_size=*/model->pos_emb->weight.size(0));
    std::cout << "Output text:\n " << ch5::token_ids_to_text(token_ids, tokenizer) << "\n";
}

// ==========================================================================
// 5.1.2 计算文本生成损失
// ==========================================================================
void demo_loss(ch4::GPTModel& model) {
    section("5.1.2 计算文本生成损失");
    auto inputs = torch::tensor({{16833L, 3626L, 6100L},  // ["every effort moves",
                                 {40L, 1107L, 588L}},     //  "I really like"]
                               torch::kLong);
    auto targets = torch::tensor({{3626L, 6100L, 345L},   // [" effort moves you",
                                  {1107L, 588L, 11311L}}, //  " really like chocolate"]
                                torch::kLong);

    torch::Tensor logits;
    {
        torch::NoGradGuard g;
        logits = model->forward(inputs);
    }
    auto probas = torch::softmax(logits, -1);
    std::cout << "probas.shape = " << probas.sizes() << "\n";

    // argmax 预测
    auto token_ids = torch::argmax(probas, -1, /*keepdim=*/true);
    std::cout << "Token IDs:\n" << token_ids << "\n";
    std::cout << "Targets batch 1: effort moves you\n"
              << "Outputs batch 1: (解码见完整运行输出)\n";

    // 目标 token 的概率
    torch::Tensor target_probas_1 = probas.index(
        {0, torch::indexing::Slice(), torch::tensor({3626L, 6100L, 345L})})
        .diagonal();
    std::cout << "Text 1: " << target_probas_1 << "\n";
    torch::Tensor target_probas_2 = probas.index(
        {1, torch::indexing::Slice(), torch::tensor({1107L, 588L, 11311L})})
        .diagonal();
    std::cout << "Text 2: " << target_probas_2 << "\n";

    // 对数概率 -> 平均 -> 负对数概率 = 交叉熵
    auto log_probas = torch::log(torch::cat({target_probas_1, target_probas_2}));
    std::cout << "log_probas = " << log_probas << "\n";
    auto avg_log_probas = torch::mean(log_probas);
    std::cout << "avg_log_probas = " << avg_log_probas << "\n";
    auto neg_avg_log_probas = avg_log_probas * -1;
    std::cout << "neg_avg_log_probas = " << neg_avg_log_probas << "\n";

    // PyTorch 内置交叉熵
    auto logits_flat = logits.flatten(0, 1);
    auto targets_flat = targets.flatten();
    std::cout << "Flattened logits: " << logits_flat.sizes()
              << ", Flattened targets: " << targets_flat.sizes() << "\n";
    auto loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
    std::cout << "Loss (cross_entropy): " << loss << "\n";
    auto perplexity = torch::exp(loss);
    std::cout << "Perplexity: " << perplexity << "\n";
}

// ==========================================================================
// 5.1.3 计算训练集和验证集的损失
// ==========================================================================
void demo_dataset_loss(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                       const std::string& raw_text) {
    section("5.1.3 计算训练集和验证集的损失");
    std::cout << "Characters: " << raw_text.size() << "\n";
    auto all_tokens = tokenizer.encode(raw_text);
    std::cout << "Tokens: " << all_tokens.size() << "\n";

    const double train_ratio = 0.90;
    size_t split_idx = static_cast<size_t>(train_ratio * raw_text.size());
    auto train_data = raw_text.substr(0, split_idx);
    auto val_data = raw_text.substr(split_idx);
    std::cout << "split_idx = " << split_idx << "\n";

    auto train_ids = tokenizer.encode(train_data);
    auto val_ids = tokenizer.encode(val_data);
    std::cout << "train_data tokens: " << train_ids.size()
              << ", val_data tokens: " << val_ids.size() << "\n";

    const int64_t batch_size = 2;
    const int64_t max_length = 256;   // GPT_CONFIG_124M["context_length"] 改为 256
    const int64_t stride = 256;

    torch::manual_seed(123);
    ch2::GPTDataLoader train_loader(train_ids, batch_size, max_length, stride,
                                    /*shuffle=*/true, /*drop_last=*/true, /*seed=*/123);
    ch2::GPTDataLoader val_loader(val_ids, batch_size, max_length, stride,
                                  /*shuffle=*/false, /*drop_last=*/false, /*seed=*/123);

    std::cout << "Train loader batches: " << train_loader.num_batches() << "\n";
    for (const auto& b : train_loader.batches()) {
        std::cout << "  x " << b.inputs.sizes() << ", y " << b.targets.sizes() << "\n";
    }
    std::cout << "Validation loader batches: " << val_loader.num_batches() << "\n";
    for (const auto& b : val_loader.batches()) {
        std::cout << "  x " << b.inputs.sizes() << ", y " << b.targets.sizes() << "\n";
    }

    double train_loss = ch5::calc_loss_loader(model, train_loader.batches());
    double val_loss = ch5::calc_loss_loader(model, val_loader.batches());
    std::cout << "Training loss: " << train_loss << "\n";
    std::cout << "Validation loss: " << val_loss << "\n";
}

// ==========================================================================
// 5.2 训练大语言模型
// ==========================================================================
void demo_training(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                   const std::string& raw_text, int64_t num_epochs) {
    section("5.2 训练大语言模型");
    const double train_ratio = 0.90;
    size_t split_idx = static_cast<size_t>(train_ratio * raw_text.size());
    auto train_ids = tokenizer.encode(raw_text.substr(0, split_idx));
    auto val_ids = tokenizer.encode(raw_text.substr(split_idx));

    const int64_t batch_size = 2;
    const int64_t max_length = 256;
    const int64_t stride = 256;
    torch::manual_seed(123);
    ch2::GPTDataLoader train_loader(train_ids, batch_size, max_length, stride,
                                    /*shuffle=*/true, /*drop_last=*/true, /*seed=*/123);
    ch2::GPTDataLoader val_loader(val_ids, batch_size, max_length, stride,
                                  /*shuffle=*/false, /*drop_last=*/false, /*seed=*/123);

    // AdamW 优化器（书中 lr=0.0004, weight_decay=0.1）
    torch::optim::AdamWOptions options(4e-4);
    options.weight_decay(0.1);
    auto optimizer = torch::optim::AdamW(model->parameters(), options);

    std::cout << "开始训练 " << num_epochs << " 轮（每轮 " << train_loader.num_batches()
              << " 个批次）...\n";
    ch5::train_model_simple(model, train_loader.batches(), val_loader.batches(), optimizer,
                            /*num_epochs=*/num_epochs, /*eval_freq=*/5, /*eval_iter=*/5,
                            /*start_context=*/"Every effort moves you", tokenizer);
}

// ==========================================================================
// 5.3 控制随机性的解码策略
// ==========================================================================
void demo_decode_strategies(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer) {
    section("5.3 控制随机性的解码策略");

    // --- 5.3.1 温度缩放：小词汇表示例 ---
    std::cout << "-- 5.3.1 温度缩放 --\n";
    auto next_token_logits = torch::tensor(
        {4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79});
    auto probas = torch::softmax(next_token_logits, 0);
    auto next_token_id = torch::argmax(probas).item<int64_t>();
    std::cout << "argmax 采样 -> id " << next_token_id << " (forward)\n";

    // multinomial 采样 1000 次
    torch::manual_seed(123);
    std::vector<int64_t> counts(9, 0);
    for (int i = 0; i < 1000; ++i) {
        int64_t s = torch::multinomial(probas, 1).item<int64_t>();
        counts[s]++;
    }
    std::cout << "multinomial 采样 1000 次分布:\n";
    std::vector<const char*> vocab = {"closer", "every", "effort", "forward", "inches",
                                      "moves", "pizza", "toward", "you"};
    for (int i = 0; i < 9; ++i) std::cout << "  " << counts[i] << " x " << vocab[i] << "\n";

    // 温度缩放 softmax
    auto softmax_with_temperature = [](const torch::Tensor& logits, double T) {
        return torch::softmax(logits / T, 0);
    };
    for (double T : {1.0, 0.1, 5.0}) {
        auto sp = softmax_with_temperature(next_token_logits, T);
        std::cout << "T=" << T << " -> forward 概率 " << sp.index({3}).item<double>()
                  << ", pizza 概率 " << sp.index({6}).item<double>() << "\n";
    }

    // --- 5.3.2 Top-k 采样 ---
    std::cout << "\n-- 5.3.2 Top-k 采样 --\n";
    auto topk = torch::topk(next_token_logits, 3);
    auto top_logits = std::get<0>(topk);
    auto top_pos = std::get<1>(topk);
    std::cout << "Top logits: " << top_logits << "\n";
    std::cout << "Top positions: " << top_pos << "\n";

    auto min_val = top_logits.index({-1});
    auto new_logits = torch::where(
        next_token_logits < min_val,
        torch::full_like(next_token_logits, -std::numeric_limits<double>::infinity()),
        next_token_logits);
    std::cout << "new_logits:\n" << new_logits << "\n";
    auto topk_probas = torch::softmax(new_logits, 0);
    std::cout << "topk_probas:\n" << topk_probas << "\n";

    // --- 5.3.3 修改后的 generate 函数 ---
    std::cout << "\n-- 5.3.3 generate（top_k=25, temperature=1.4）--\n";
    model->eval();
    torch::manual_seed(123);
    auto token_ids = ch5::generate(model,
                                   ch5::text_to_token_ids("Every effort moves you", tokenizer),
                                   /*max_new_tokens=*/15,
                                   /*context_size=*/model->pos_emb->weight.size(0),
                                   /*temperature=*/1.4, /*top_k=*/25);
    std::cout << "Output text:\n " << ch5::token_ids_to_text(token_ids, tokenizer) << "\n";
    model->train();
}

// ==========================================================================
// 5.4 保存和加载模型权重
// ==========================================================================
void demo_save_load(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                    const std::string& dir) {
    section("5.4 使用 PyTorch 保存和加载模型权重");
    std::string model_path = dir + "/model.pth";

    torch::save(model, model_path);
    std::cout << "已保存模型权重到 " << model_path << "（"
              << fs::file_size(model_path) / (1024.0 * 1024.0) << " MB）\n";

    // 加载到新实例并验证：生成相同文本
    ch4::GPTConfig cfg;
    cfg.context_length = 256;  // 与训练配置一致
    ch4::GPTModel model2(cfg);
    torch::load(model2, model_path);
    model2->eval();

    auto encoded = ch5::text_to_token_ids("Every effort moves you", tokenizer);
    model->eval();
    auto t1 = ch4::generate_text_simple(model, encoded, 5,
                                        model->pos_emb->weight.size(0));
    auto t2 = ch4::generate_text_simple(model2, encoded, 5,
                                        model2->pos_emb->weight.size(0));
    model->train();
    bool same = t1.equal(t2);
    std::cout << "加载后生成与训练模型一致: " << (same ? "yes" : "no") << "\n";
    std::cout << "  生成: " << ch5::token_ids_to_text(t2, tokenizer) << "\n";

    std::cout << "\n说明：C++ 版 torch::save(model, path) 保存整个模型（含 state_dict）。\n"
              << "书中保存 optimizer 状态（AdamW 动量等）在 C++ API 中无对应接口，\n"
              << "如要续训，需要自行序列化优化器状态。\n";
}

// ==========================================================================
// 5.5 从 OpenAI 加载预训练权重
// ==========================================================================
// 书中用 TensorFlow 解析 OpenAI checkpoint；本实现用 HuggingFace 的
// safetensors（gpt2 模型，权重与 OpenAI 相同），键名映射到书中 params 结构
// （wte/wpe/blocks[i].attn.c_attn.w 等），加载逻辑与代码清单 5-5 完全一致。
void demo_openai_weights(ch2::BpeTokenizer& tokenizer, const std::string& st_path) {
    section("5.5 从 OpenAI 加载预训练权重");
    if (!fs::exists(st_path)) {
        std::cout << "未找到 " << st_path << "，跳过 5.5。\n"
                  << "下载命令：\n"
                  << "  curl -fsSL -L -o " << st_path
                  << " https://huggingface.co/gpt2/resolve/main/model.safetensors\n";
        return;
    }

    std::cout << "加载 safetensors: " << st_path << "\n";
    auto st = ch5::load_safetensors(st_path);
    std::cout << "张量数量: " << st.size() << "\n";

    // 键名映射：书中 params 键名 -> HF gpt2 safetensors 原始键名。
    // 注意：HF 新版本键名无 "transformer." 前缀；attn.bias 是因果掩码 buffer（无需加载）；
    // c_attn.weight 为 [in, 3*emb]（与 OpenAI c_attn.w 存储顺序一致）。
    auto map_key = [](const std::string& k) -> std::string {
        if (k == "wte") return "wte.weight";
        if (k == "wpe") return "wpe.weight";
        if (k == "g") return "ln_f.weight";
        if (k == "b") return "ln_f.bias";
        // blocks.{i}.xxx -> h.{i}.xxx
        if (k.rfind("blocks.", 0) == 0) {
            std::string rest = k.substr(7);
            size_t dot = rest.find('.');
            std::string idx = rest.substr(0, dot);
            std::string tail = rest.substr(dot + 1);
            if (tail == "ln_1.g") return "h." + idx + ".ln_1.weight";
            if (tail == "ln_1.b") return "h." + idx + ".ln_1.bias";
            if (tail == "ln_2.g") return "h." + idx + ".ln_2.weight";
            if (tail == "ln_2.b") return "h." + idx + ".ln_2.bias";
            if (tail == "attn.c_attn.w") return "h." + idx + ".attn.c_attn.weight";
            if (tail == "attn.c_attn.b") return "h." + idx + ".attn.c_attn.bias";
            if (tail == "attn.c_proj.w") return "h." + idx + ".attn.c_proj.weight";
            if (tail == "attn.c_proj.b") return "h." + idx + ".attn.c_proj.bias";
            if (tail == "mlp.c_fc.w") return "h." + idx + ".mlp.c_fc.weight";
            if (tail == "mlp.c_fc.b") return "h." + idx + ".mlp.c_fc.bias";
            if (tail == "mlp.c_proj.w") return "h." + idx + ".mlp.c_proj.weight";
            if (tail == "mlp.c_proj.b") return "h." + idx + ".mlp.c_proj.bias";
        }
        return k;
    };

    // NEW_CONFIG：context 1024 + qkv_bias True（书中 5.5 节）
    ch4::GPTConfig cfg;
    cfg.context_length = 1024;
    cfg.qkv_bias = true;
    ch4::GPTModel gpt(cfg);
    gpt->eval();
    std::cout << "GPTModel(1024 ctx, qkv_bias) 创建完成\n";

    // assign + load_weights_into_gpt（代码清单 5-5）
    // 注：参数是 requires_grad 的叶子张量，in-place 赋值须在 no_grad 下进行
    // （等价 Python 的 with torch.no_grad(): param.copy_(...)）
    auto assign = [](torch::Tensor left, const torch::Tensor& right) {
        if (left.sizes() != right.sizes()) {
            throw std::runtime_error("Shape mismatch. Left: " +
                                     std::to_string(left.numel()) +
                                     ", Right: " + std::to_string(right.numel()));
        }
        {
            torch::NoGradGuard g;
            left.copy_(right);
        }
        return left;
    };
    auto params = [&](const std::string& name) -> const torch::Tensor& {
        return st.at(map_key(name));
    };

    // 词元/位置嵌入
    gpt->pos_emb->weight = assign(gpt->pos_emb->weight, params("wpe"));
    gpt->tok_emb->weight = assign(gpt->tok_emb->weight, params("wte"));

    for (int64_t b = 0; b < static_cast<int64_t>(cfg.n_layers); ++b) {
        auto& block = gpt->trf_blocks->at<ch4::TransformerBlockImpl>(b);
        // c_attn.w [emb, 3*emb] -> 按最后一维 split -> q/k/v [emb, emb] 各自转置
        // （与书 np.split(axis=-1) 一致）
        auto c_attn_w = params("blocks." + std::to_string(b) + ".attn.c_attn.w");
        auto split = c_attn_w.split(static_cast<int64_t>(cfg.emb_dim), /*dim=*/1);
        block.att->W_query->weight = assign(block.att->W_query->weight, split[0].t().contiguous());
        block.att->W_key->weight = assign(block.att->W_key->weight, split[1].t().contiguous());
        block.att->W_value->weight = assign(block.att->W_value->weight, split[2].t().contiguous());

        auto c_attn_b = params("blocks." + std::to_string(b) + ".attn.c_attn.b");
        auto split_b = c_attn_b.split(static_cast<int64_t>(cfg.emb_dim), 0);
        block.att->W_query->bias = assign(block.att->W_query->bias, split_b[0]);
        block.att->W_key->bias = assign(block.att->W_key->bias, split_b[1]);
        block.att->W_value->bias = assign(block.att->W_value->bias, split_b[2]);

        block.att->out_proj->weight = assign(
            block.att->out_proj->weight,
            params("blocks." + std::to_string(b) + ".attn.c_proj.w").t().contiguous());
        block.att->out_proj->bias = assign(
            block.att->out_proj->bias,
            params("blocks." + std::to_string(b) + ".attn.c_proj.b"));

        auto& ff0 = block.ff->layers->at<torch::nn::LinearImpl>(0);
        ff0.weight = assign(ff0.weight,
                            params("blocks." + std::to_string(b) + ".mlp.c_fc.w").t().contiguous());
        ff0.bias = assign(ff0.bias, params("blocks." + std::to_string(b) + ".mlp.c_fc.b"));
        auto& ff2 = block.ff->layers->at<torch::nn::LinearImpl>(2);
        ff2.weight = assign(
            ff2.weight, params("blocks." + std::to_string(b) + ".mlp.c_proj.w").t().contiguous());
        ff2.bias = assign(ff2.bias, params("blocks." + std::to_string(b) + ".mlp.c_proj.b"));

        block.norm1->scale_ = assign(block.norm1->scale_,
                                     params("blocks." + std::to_string(b) + ".ln_1.g"));
        block.norm1->shift_ = assign(block.norm1->shift_,
                                     params("blocks." + std::to_string(b) + ".ln_1.b"));
        block.norm2->scale_ = assign(block.norm2->scale_,
                                     params("blocks." + std::to_string(b) + ".ln_2.g"));
        block.norm2->shift_ = assign(block.norm2->shift_,
                                     params("blocks." + std::to_string(b) + ".ln_2.b"));
    }
    gpt->final_norm->scale_ = assign(gpt->final_norm->scale_, params("g"));
    gpt->final_norm->shift_ = assign(gpt->final_norm->shift_, params("b"));
    gpt->out_head->weight = assign(gpt->out_head->weight, params("wte"));  // weight tying

    std::cout << "权重加载完成，生成文本（top_k=50, temperature=1.5）:\n";
    gpt->eval();
    torch::manual_seed(123);
    auto token_ids = ch5::generate(gpt,
                                   ch5::text_to_token_ids("Every effort moves you", tokenizer),
                                   /*max_new_tokens=*/25,
                                   /*context_size=*/1024,
                                   /*temperature=*/1.5, /*top_k=*/50);
    std::cout << "Output text:\n " << ch5::token_ids_to_text(token_ids, tokenizer) << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = DATA_DIR;
    if (argc > 1) data_dir = argv[1];
    int64_t num_epochs = 10;
    if (argc > 2) num_epochs = std::stoll(argv[2]);
    std::string st_path = data_dir + "/gpt2-model.safetensors";
    if (argc > 3) st_path = argv[3];

    try {
        ch2::BpeTokenizer tokenizer(data_dir + "/encoder.json", data_dir + "/vocab.bpe");
        std::string raw_text;
        if (fs::exists(data_dir + "/the-verdict.txt")) {
            raw_text = read_file(data_dir + "/the-verdict.txt");
        } else {
            raw_text = std::string(1, ' ');
        }
        std::cout << "=== 第 5 章：在无标签数据上进行预训练（C++ + LibTorch）===\n"
                  << "数据目录: " << data_dir << "\n"
                  << "训练轮数: " << num_epochs << "\n"
                  << "设备: CPU, 线程 "
                  << torch::get_num_threads() << "\n";

        // 5.1.1/5.1.2 用 seed 123 的随机初始化模型（书中 context=256）
        ch4::GPTConfig cfg;
        cfg.context_length = 256;
        torch::manual_seed(123);
        ch4::GPTModel model(cfg);
        model->eval();

        demo_generate_text(model, tokenizer);
        demo_loss(model);

        if (raw_text.size() > 1) {
            demo_dataset_loss(model, tokenizer, raw_text);
            if (num_epochs > 0) {
                demo_training(model, tokenizer, raw_text, num_epochs);
            }
            demo_decode_strategies(model, tokenizer);
            demo_save_load(model, tokenizer, data_dir);
        } else {
            std::cout << "（未找到 the-verdict.txt，跳过 5.1.3/5.2/5.3/5.4）\n";
        }

        demo_openai_weights(tokenizer, st_path);

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 5 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
