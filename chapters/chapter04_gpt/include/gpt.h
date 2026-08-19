// gpt.h
// 第 4 章：从头实现 GPT 模型进行文本生成 —— 模型构建块与 GPTModel
//
// 对应书中代码清单：
//   LayerNorm               代码清单 4-2
//   GELU                    代码清单 4-3（tanh 近似，与原始 GPT-2 一致）
//   FeedForward             代码清单 4-4
//   ExampleDeepNeuralNetwork 代码清单 4-5（快捷连接 + 梯度演示）
//   TransformerBlock        代码清单 4-6
//   GPTModel                代码清单 4-7
//   generate_text_simple    代码清单 4-8
//
// 依赖：ch3::MultiHeadAttention（见第 3 章 include/attention.h）
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <string>
#include <vector>

#include "attention.h"

namespace ch4 {

// 对应书中 GPT_CONFIG_124M 配置字典
struct GPTConfig {
    int64_t vocab_size = 50257;
    int64_t context_length = 1024;
    int64_t emb_dim = 768;
    int64_t n_heads = 12;
    int64_t n_layers = 12;
    double drop_rate = 0.1;
    bool qkv_bias = false;
};

// ==========================================================================
// LayerNorm（代码清单 4-2）：有偏方差（unbiased=False），与 TensorFlow GPT-2 兼容
// ==========================================================================
class LayerNormImpl : public torch::nn::Module {
public:
    explicit LayerNormImpl(int64_t emb_dim) {
        scale_ = register_parameter("scale", torch::ones(emb_dim));
        shift_ = register_parameter("shift", torch::zeros(emb_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto mean = x.mean(-1, /*keepdim=*/true);
        auto var = x.var({-1}, /*unbiased=*/false, /*keepdim=*/true);
        auto norm_x = (x - mean) / torch::sqrt(var + eps_);
        return scale_ * norm_x + shift_;
    }

    torch::Tensor scale_;
    torch::Tensor shift_;
    double eps_ = 1e-5;
};
TORCH_MODULE(LayerNorm);

// ==========================================================================
// GELU（代码清单 4-3）：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// ==========================================================================
class GELUImpl : public torch::nn::Module {
public:
    torch::Tensor forward(torch::Tensor x) {
        auto c = torch::sqrt(torch::tensor(2.0 / 3.14159265358979323846));
        return 0.5 * x * (1.0 + torch::tanh(c * (x + 0.044715 * torch::pow(x, 3))));
    }
};
TORCH_MODULE(GELU);

// ==========================================================================
// FeedForward（代码清单 4-4）：Linear(emb, 4*emb) -> GELU -> Linear(4*emb, emb)
// ==========================================================================
class FeedForwardImpl : public torch::nn::Module {
public:
    explicit FeedForwardImpl(const GPTConfig& cfg) {
        // 注意：不能写成 Sequential(Linear, GELU(), Linear) 的单表达式构造——
        // C++ 函数参数求值顺序未指定（GCC 从右到左），会导致第二个 Linear
        // 先于第一个初始化，RNG 顺序与 Python 不一致。因此逐个 push_back。
        layers = register_module("layers", torch::nn::Sequential());
        layers->push_back(torch::nn::Linear(cfg.emb_dim, 4 * cfg.emb_dim));
        layers->push_back(GELU());
        layers->push_back(torch::nn::Linear(4 * cfg.emb_dim, cfg.emb_dim));
    }

    torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }

    torch::nn::Sequential layers{nullptr};
};
TORCH_MODULE(FeedForward);

// ==========================================================================
// TransformerBlock（代码清单 4-6）：Pre-LayerNorm + 快捷连接
// ==========================================================================
class TransformerBlockImpl : public torch::nn::Module {
public:
    explicit TransformerBlockImpl(const GPTConfig& cfg) {
        att = register_module(
            "att", ch3::MultiHeadAttention(cfg.emb_dim, cfg.emb_dim, cfg.context_length,
                                           cfg.drop_rate, cfg.n_heads, cfg.qkv_bias));
        ff = register_module("ff", FeedForward(cfg));
        norm1 = register_module("norm1", LayerNorm(cfg.emb_dim));
        norm2 = register_module("norm2", LayerNorm(cfg.emb_dim));
        drop_shortcut = register_module("drop_shortcut", torch::nn::Dropout(cfg.drop_rate));
    }

    torch::Tensor forward(torch::Tensor x) {
        // 注意力子块（Pre-LayerNorm + 快捷连接）
        auto shortcut = x;
        x = norm1->forward(x);
        x = att->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;

        // 前馈子块
        shortcut = x;
        x = norm2->forward(x);
        x = ff->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;
        return x;
    }

    ch3::MultiHeadAttention att{nullptr};
    FeedForward ff{nullptr};
    LayerNorm norm1{nullptr};
    LayerNorm norm2{nullptr};
    torch::nn::Dropout drop_shortcut{nullptr};
};
TORCH_MODULE(TransformerBlock);

// ==========================================================================
// GPTModel（代码清单 4-7）：词元+位置嵌入 -> N 个 Transformer 块 -> LayerNorm -> 输出头
// ==========================================================================
class GPTModelImpl : public torch::nn::Module {
public:
    explicit GPTModelImpl(const GPTConfig& cfg) {
        tok_emb = register_module("tok_emb",
                                  torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim));
        pos_emb = register_module(
            "pos_emb", torch::nn::Embedding(cfg.context_length, cfg.emb_dim));
        drop_emb = register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate));

        trf_blocks = register_module("trf_blocks", torch::nn::Sequential());
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            trf_blocks->push_back(TransformerBlock(cfg));
        }

        final_norm = register_module("final_norm", LayerNorm(cfg.emb_dim));
        out_head = register_module(
            "out_head", torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,
                                                                   cfg.vocab_size)
                                              .bias(false)));
    }

    torch::Tensor forward(torch::Tensor in_idx) {
        const int64_t batch_size = in_idx.size(0);
        const int64_t seq_len = in_idx.size(1);

        auto tok_embeds = tok_emb->forward(in_idx);
        auto pos_embeds = pos_emb->forward(torch::arange(
            seq_len, torch::TensorOptions().device(in_idx.device()).dtype(torch::kLong)));

        auto x = tok_embeds + pos_embeds;
        x = drop_emb->forward(x);
        x = trf_blocks->forward(x);
        x = final_norm->forward(x);
        auto logits = out_head->forward(x);
        return logits;
    }

    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::Dropout drop_emb{nullptr};
    torch::nn::Sequential trf_blocks{nullptr};
    LayerNorm final_norm{nullptr};
    torch::nn::Linear out_head{nullptr};
};
TORCH_MODULE(GPTModel);

// ==========================================================================
// DummyGPTModel（代码清单 4-1）：占位符架构（Dummy TransformerBlock / LayerNorm）
// ==========================================================================
class DummyTransformerBlockImpl : public torch::nn::Module {
public:
    torch::Tensor forward(torch::Tensor x) { return x; }
};
TORCH_MODULE(DummyTransformerBlock);

class DummyLayerNormImpl : public torch::nn::Module {
public:
    torch::Tensor forward(torch::Tensor x) { return x; }
};
TORCH_MODULE(DummyLayerNorm);

class DummyGPTModelImpl : public torch::nn::Module {
public:
    explicit DummyGPTModelImpl(const GPTConfig& cfg) {
        tok_emb = register_module("tok_emb",
                                  torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim));
        pos_emb = register_module(
            "pos_emb", torch::nn::Embedding(cfg.context_length, cfg.emb_dim));
        drop_emb = register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate));
        trf_blocks = register_module("trf_blocks", torch::nn::Sequential());
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            trf_blocks->push_back(DummyTransformerBlock());
        }
        final_norm = register_module("final_norm", DummyLayerNorm());
        out_head = register_module(
            "out_head", torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,
                                                                   cfg.vocab_size)
                                              .bias(false)));
    }

    torch::Tensor forward(torch::Tensor in_idx) {
        const int64_t seq_len = in_idx.size(1);
        auto tok_embeds = tok_emb->forward(in_idx);
        auto pos_embeds = pos_emb->forward(torch::arange(
            seq_len, torch::TensorOptions().device(in_idx.device()).dtype(torch::kLong)));
        auto x = tok_embeds + pos_embeds;
        x = drop_emb->forward(x);
        x = trf_blocks->forward(x);
        x = final_norm->forward(x);
        auto logits = out_head->forward(x);
        return logits;
    }

    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::Dropout drop_emb{nullptr};
    torch::nn::Sequential trf_blocks{nullptr};
    DummyLayerNorm final_norm{nullptr};
    torch::nn::Linear out_head{nullptr};
};
TORCH_MODULE(DummyGPTModel);

// ==========================================================================
// ExampleDeepNeuralNetwork（代码清单 4-5）：快捷连接 vs 无快捷连接
// ==========================================================================
class ExampleDeepNeuralNetworkImpl : public torch::nn::Module {
public:
    ExampleDeepNeuralNetworkImpl(const std::vector<int64_t>& layer_sizes,
                                 bool use_shortcut)
        : use_shortcut_(use_shortcut) {
        // 用 ModuleList 逐层注册（自动编号 "0"~"4"，named_parameters
        // 键名与 Python 的 "layers.0.0.weight" 一致）
        layers_ = register_module("layers", torch::nn::ModuleList());
        for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
            auto seq = torch::nn::Sequential();
            seq->push_back(torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1]));
            seq->push_back(GELU());
            layers_->push_back(seq);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        for (size_t i = 0; i < layers_->size(); ++i) {
            auto& layer = layers_->at<torch::nn::SequentialImpl>(i);
            auto layer_output = layer.forward(x);
            if (use_shortcut_ && x.sizes() == layer_output.sizes()) {
                x = x + layer_output;  // 快捷连接
            } else {
                x = layer_output;
            }
        }
        return x;
    }

    torch::nn::ModuleList layers_{nullptr};
    bool use_shortcut_;
};
TORCH_MODULE(ExampleDeepNeuralNetwork);

// ==========================================================================
// generate_text_simple（代码清单 4-8）：贪心解码逐词元生成
// ==========================================================================
inline torch::Tensor generate_text_simple(GPTModel& model, torch::Tensor idx,
                                          int64_t max_new_tokens,
                                          int64_t context_size) {
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        // 截断到模型支持的最大上下文长度
        auto idx_cond =
            idx.index({torch::indexing::Slice(),
                       torch::indexing::Slice(-context_size, torch::indexing::None)});
        torch::Tensor logits;
        {
            torch::NoGradGuard no_grad;
            logits = model->forward(idx_cond);
        }
        logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
        auto probas = torch::softmax(logits, -1);
        auto idx_next = torch::argmax(probas, -1, /*keepdim=*/true);
        idx = torch::cat({idx, idx_next}, 1);
    }
    return idx;
}

}  // namespace ch4
