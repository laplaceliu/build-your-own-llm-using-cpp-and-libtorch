// main.cpp
// 第 4 章：从头实现 GPT 模型进行文本生成（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印，方便对照验证：
//   4.1  DummyGPTModel 占位架构
//   4.2  层归一化 LayerNorm
//   4.3  GELU 激活函数 + FeedForward 前馈网络
//   4.4  快捷连接（对比有/无快捷连接的梯度）
//   4.5  TransformerBlock
//   4.6  GPTModel（1.63 亿参数 / weight tying 1.24 亿 / 621.83 MB）
//   4.7  文本生成（贪心解码）
// 练习 4.1 前馈 vs 注意力参数量；4.2 GPT-2 各规模参数；4.3 提示。
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"
#include "gpt.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

namespace {

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

// 统计模块参数量
int64_t count_params(const torch::nn::Module& m) {
    int64_t total = 0;
    for (const auto& p : m.parameters()) total += p.numel();
    return total;
}

void print_ids(const std::string& label, const std::vector<int>& ids) {
    std::cout << label;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "\n";
}

// ---- 4.1 DummyGPTModel ----
void demo_dummy_gpt(ch2::BpeTokenizer& tokenizer) {
    section("4.1 DummyGPTModel（占位符架构）");
    const ch4::GPTConfig cfg;

    // 书中 batch：两段文本 BPE 编码
    auto txt1 = tokenizer.encode("Every effort moves you");
    auto txt2 = tokenizer.encode("Every day holds a");
    std::cout << "txt1 = ";
    print_ids("", txt1);
    std::cout << "txt2 = ";
    print_ids("", txt2);
    auto batch = torch::stack({torch::tensor(txt1, torch::kLong),
                               torch::tensor(txt2, torch::kLong)}, 0);
    std::cout << "batch:\n" << batch << "\n";

    torch::manual_seed(123);
    ch4::DummyGPTModel model(cfg);
    auto logits = model->forward(batch);
    std::cout << "Output shape: " << logits.sizes() << "\n";
    // 打印每个 batch/位置的前 3 个 logits（与书中输出对照）
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t t = 0; t < 4; ++t) {
            std::cout << "  logits[" << b << "][" << t << "][:3] = "
                      << logits.index({b, t, torch::indexing::Slice(0, 3)}) << "\n";
        }
    }
}

// ---- 4.2 LayerNorm ----
void demo_layer_norm() {
    section("4.2 层归一化（LayerNorm）");
    torch::manual_seed(123);
    auto batch_example = torch::randn({2, 5});
    auto layer = torch::nn::Sequential(torch::nn::Linear(5, 6), torch::nn::ReLU());
    auto out = layer->forward(batch_example);
    std::cout << "out (Linear(5,6)+ReLU):\n" << out << "\n";

    // 逐步：mean / var（unbiased=True 默认，与书中演示一致）
    auto mean = out.mean(-1, /*keepdim=*/true);
    auto var = out.var({-1}, /*unbiased=*/true, /*keepdim=*/true);
    std::cout << "Mean:\n" << mean << "\n";
    std::cout << "Variance:\n" << var << "\n";

    // 归一化
    auto out_norm = (out - mean) / torch::sqrt(var);
    std::cout << "Normalized layer outputs:\n" << out_norm << "\n";
    std::cout << "Mean of normalized:\n" << out_norm.mean(-1, true) << "\n";
    std::cout << "Variance of normalized:\n"
              << out_norm.var({-1}, /*unbiased=*/true, /*keepdim=*/true) << "\n";

    // LayerNorm 类
    ch4::LayerNorm ln(5);
    auto out_ln = ln->forward(batch_example);
    std::cout << "LayerNorm(5)(batch_example) mean:\n"
              << out_ln.mean(-1, true) << "\n";
    std::cout << "LayerNorm(5)(batch_example) var:\n"
              << out_ln.var({-1}, /*unbiased=*/false, /*keepdim=*/true) << "\n";
}

// ---- 4.3 GELU + FeedForward ----
void demo_gelu_ff() {
    section("4.3 GELU 激活函数 + FeedForward 前馈网络");
    ch4::GELU gelu;
    auto x = torch::tensor({-3.0, -2.0, -1.0, -0.75, 0.0, 1.0, 2.0, 3.0});
    std::cout << "x      = " << x << "\n";
    std::cout << "GELU(x)= " << gelu->forward(x) << "\n";
    std::cout << "ReLU(x)= " << torch::relu(x) << "\n";

    const ch4::GPTConfig cfg;
    ch4::FeedForward ffn(cfg);
    auto fx = torch::rand({2, 3, cfg.emb_dim});
    auto fout = ffn->forward(fx);
    std::cout << "FeedForward: input " << fx.sizes() << " -> output " << fout.sizes()
              << "\n";
    std::cout << "FeedForward 参数量 = " << count_params(*ffn) << "\n";
}

// ---- 4.4 快捷连接 ----
void print_gradients(ch4::ExampleDeepNeuralNetwork& model, const torch::Tensor& x) {
    auto output = model->forward(x);
    auto target = torch::zeros({1, 1});
    auto loss = torch::mse_loss(output, target);
    model->zero_grad();
    loss.backward();
    for (auto& np : model->named_parameters()) {
        if (np.key().find("weight") != std::string::npos) {
            auto g = np.value().grad();
            std::cout << np.key() << " has gradient mean of "
                      << (g.defined() ? g.abs().mean().item<double>() : 0.0) << "\n";
        }
    }
}

void demo_shortcut() {
    section("4.4 添加快捷连接（对比梯度）");
    const std::vector<int64_t> layer_sizes{3, 3, 3, 3, 3, 1};
    auto sample_input = torch::tensor({{1.0f, 0.0f, -1.0f}});

    torch::manual_seed(123);
    ch4::ExampleDeepNeuralNetwork model_without_shortcut(layer_sizes, /*use_shortcut=*/false);
    std::cout << "-- 无快捷连接 --\n";
    print_gradients(model_without_shortcut, sample_input);

    torch::manual_seed(123);
    ch4::ExampleDeepNeuralNetwork model_with_shortcut(layer_sizes, /*use_shortcut=*/true);
    std::cout << "-- 有快捷连接 --\n";
    print_gradients(model_with_shortcut, sample_input);
}

// ---- 4.5 TransformerBlock ----
void demo_transformer_block() {
    section("4.5 TransformerBlock");
    const ch4::GPTConfig cfg;
    torch::manual_seed(123);
    auto x = torch::rand({2, 4, cfg.emb_dim});
    ch4::TransformerBlock block(cfg);
    auto output = block->forward(x);
    std::cout << "Input shape:  " << x.sizes() << "\n";
    std::cout << "Output shape: " << output.sizes() << "\n";
    std::cout << "Output[:1,0,:5] = " << output.index({0, 0, torch::indexing::Slice(0, 5)})
              << "\n";
}

// ---- 4.6 GPTModel ----
void demo_gpt_model(ch4::GPTModel& model) {
    section("4.6 GPTModel（1.24 亿参数 GPT-2 small）");
    auto batch = torch::tensor({{6109L, 3626L, 6100L, 345L},
                                {6109L, 1110L, 6622L, 257L}}, torch::kLong);
    std::cout << "Input batch:\n" << batch << "\n";

    auto out = model->forward(batch);
    std::cout << "Output shape: " << out.sizes() << "\n";


    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t t = 0; t < 4; ++t) {
            std::cout << "  out[" << b << "][" << t << "][:3] = "
                      << out.index({b, t, torch::indexing::Slice(0, 3)}) << "\n";
        }
    }

    // 参数统计
    int64_t total_params = count_params(*model);
    std::cout << "Total number of parameters: " << total_params << "\n";
    std::cout << "Token embedding layer shape: " << model->tok_emb->weight.sizes() << "\n";
    std::cout << "Output layer shape: " << model->out_head->weight.sizes() << "\n";

    int64_t out_head_params = count_params(*model->out_head);
    int64_t total_params_gpt2 = total_params - out_head_params;
    std::cout << "Number of trainable parameters considering weight tying: "
              << total_params_gpt2 << "\n";

    double total_size_mb = total_params * 4.0 / (1024.0 * 1024.0);
    std::cout << "Total size of the model: " << total_size_mb << " MB\n";

    // 练习 4.1：FeedForward vs MultiHeadAttention 参数量
    const ch4::GPTConfig cfg;
    ch4::FeedForward ffn(cfg);
    ch3::MultiHeadAttention mha(cfg.emb_dim, cfg.emb_dim, cfg.context_length,
                                cfg.drop_rate, cfg.n_heads, cfg.qkv_bias);
    std::cout << "\n[练习 4.1] FeedForward 参数量 = " << count_params(*ffn)
              << ", MultiHeadAttention 参数量 = " << count_params(*mha) << "\n";

    // 练习 4.2：GPT-2 各规模（用公式估算，避免实例化超大模型占用内存）
    auto estimate = [](int64_t vocab, int64_t context, int64_t emb, int64_t n_layers,
                       bool qkv_bias) -> int64_t {
        int64_t tok = vocab * emb;
        int64_t pos = context * emb;
        int64_t att_qkv = 3 * (emb * emb + (qkv_bias ? emb : 0));
        int64_t att_out = emb * emb + emb;
        int64_t ff1 = emb * 4 * emb + 4 * emb;
        int64_t ff2 = 4 * emb * emb + emb;
        int64_t norm = 4 * emb;
        int64_t per_layer = att_qkv + att_out + ff1 + ff2 + norm;
        return tok + pos + per_layer * n_layers + 2 * emb + emb * vocab;
    };
    std::cout << "\n[练习 4.2] 公式估算 vs 实际（124M）: " << estimate(50257, 1024, 768, 12, false)
              << " (实际 " << total_params << ")\n";
    struct Spec { const char* name; int64_t emb, n_layers, n_heads; };
    std::vector<Spec> specs{
        {"GPT-2 medium", 1024, 24, 16},
        {"GPT-2 large", 1280, 36, 20},
        {"GPT-2 xl", 1600, 48, 25},
    };
    for (const auto& s : specs) {
        int64_t p = estimate(50257, 1024, s.emb, s.n_layers, false);
        std::cout << "  " << s.name << " (emb=" << s.emb << ", layers=" << s.n_layers
                  << ", heads=" << s.n_heads << "): 总参数 " << p
                  << "，weight tying 后 " << p - 50257 * s.emb << "\n";
    }
}

// ---- 4.7 生成文本 ----
void demo_generate(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer) {
    section("4.7 生成文本（贪心解码）");
    const std::string start_context = "Hello, I am";
    auto encoded = tokenizer.encode(start_context);
    std::cout << "encoded: ";
    print_ids("", encoded);
    auto encoded_tensor = torch::tensor(encoded, torch::kLong).unsqueeze(0);
    std::cout << "encoded_tensor.shape: " << encoded_tensor.sizes() << "\n";

    model->eval();  // 关闭 dropout
    auto out = ch4::generate_text_simple(model, encoded_tensor, /*max_new_tokens=*/6,
                                         /*context_size=*/1024);
    std::cout << "Output: " << out << "\n";
    std::cout << "Output length: " << out.size(1) << "\n";

    std::vector<int> ids;
    for (int64_t i = 0; i < out.size(1); ++i) ids.push_back(out.index({0, i}).item<int>());
    std::cout << "Decoded text: \"" << tokenizer.decode(ids) << "\"\n";

    // 练习 4.3 提示：3 个 dropout 层位置
    std::cout << "\n[练习 4.3] GPT 模型中共有 3 处 dropout："
              << "drop_emb（嵌入后）、trf_blocks 每块的 drop_shortcut（快捷连接）、"
              << "MultiHeadAttention 内部的 dropout_。\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = DATA_DIR;
    if (argc > 1) data_dir = argv[1];

    try {
        ch2::BpeTokenizer tokenizer(data_dir + "/encoder.json", data_dir + "/vocab.bpe");
        std::cout << "=== 第 4 章：从头实现 GPT 模型进行文本生成（C++ + LibTorch）===\n"
                  << "数据目录: " << data_dir << "\n";

        demo_dummy_gpt(tokenizer);
        demo_layer_norm();
        demo_gelu_ff();
        demo_shortcut();
        demo_transformer_block();

        // 4.6 与 4.7 共用一个 seed 123 初始化的 GPTModel
        const ch4::GPTConfig cfg;
        torch::manual_seed(123);
        ch4::GPTModel model(cfg);
        demo_gpt_model(model);
        demo_generate(model, tokenizer);

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 4 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
