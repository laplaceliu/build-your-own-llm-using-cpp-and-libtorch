// attention.h
// 第 3 章：编码注意力机制 —— 各种注意力机制的 LibTorch 实现
//
// 对应书中代码清单：
//   SelfAttention_v1            代码清单 3-1（nn.Parameter + torch.rand）
//   SelfAttention_v2            代码清单 3-2（nn.Linear）
//   CausalAttention             代码清单 3-3（因果掩码 + dropout，支持 batch）
//   MultiHeadAttentionWrapper   代码清单 3-4（堆叠多个单头）
//   MultiHeadAttention          代码清单 3-5（权重划分 + 批矩阵乘）
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <vector>

namespace ch3 {

// ==========================================================================
// SelfAttention_v1：手动 Parameter + torch.rand 初始化（代码清单 3-1）
// （libtorch 2.13 的 C++ API 中 Parameter 即 Tensor，经 register_parameter 注册）
// ==========================================================================
class SelfAttentionV1Impl : public torch::nn::Module {
public:
    SelfAttentionV1Impl(int64_t d_in, int64_t d_out) {
        W_query = register_parameter("W_query", torch::rand({d_in, d_out}));
        W_key = register_parameter("W_key", torch::rand({d_in, d_out}));
        W_value = register_parameter("W_value", torch::rand({d_in, d_out}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto keys = x.matmul(W_key);
        auto queries = x.matmul(W_query);
        auto values = x.matmul(W_value);
        auto attn_scores = queries.matmul(keys.t());  // omega
        auto attn_weights = torch::softmax(
            attn_scores / std::sqrt(keys.size(-1)), -1);
        return attn_weights.matmul(values);
    }

    torch::Tensor W_query;
    torch::Tensor W_key;
    torch::Tensor W_value;
};
TORCH_MODULE(SelfAttentionV1);

// ==========================================================================
// SelfAttention_v2：nn.Linear（代码清单 3-2，权重为 kaiming 初始化）
// ==========================================================================
class SelfAttentionV2Impl : public torch::nn::Module {
public:
    SelfAttentionV2Impl(int64_t d_in, int64_t d_out, bool qkv_bias = false) {
        W_query = register_module("W_query",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_key = register_module("W_key",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_value = register_module("W_value",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto keys = W_key->forward(x);
        auto queries = W_query->forward(x);
        auto values = W_value->forward(x);
        auto attn_scores = queries.matmul(keys.t());
        auto attn_weights = torch::softmax(
            attn_scores / std::sqrt(keys.size(-1)), -1);
        return attn_weights.matmul(values);
    }

    torch::nn::Linear W_query{nullptr};
    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
};
TORCH_MODULE(SelfAttentionV2);

// ==========================================================================
// CausalAttention：因果掩码 + dropout，支持 batch（代码清单 3-3）
// ==========================================================================
class CausalAttentionImpl : public torch::nn::Module {
public:
    CausalAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length,
                        double dropout, bool qkv_bias = false)
        : d_out_(d_out) {
        W_query = register_module("W_query",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_key = register_module("W_key",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_value = register_module("W_value",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
        // register_buffer：掩码随模型移动设备
        mask_ = register_buffer("mask",
            torch::triu(torch::ones({context_length, context_length}), /*diagonal=*/1));
    }

    torch::Tensor forward(torch::Tensor x) {
        const int64_t b = x.size(0);
        const int64_t num_tokens = x.size(1);
        // const int64_t d_in = x.size(2);
        auto keys = W_key->forward(x);
        auto queries = W_query->forward(x);
        auto values = W_value->forward(x);

        auto attn_scores = queries.matmul(keys.transpose(1, 2));
        // 掩码对角线上方 -> -inf（mask 是 triu(diagonal=1)）
        auto mask_bool = mask_.to(torch::kBool)
                             .slice(0, 0, num_tokens)
                             .slice(1, 0, num_tokens);
        attn_scores = attn_scores.masked_fill(mask_bool, -std::numeric_limits<double>::infinity());

        auto attn_weights = torch::softmax(
            attn_scores / std::sqrt(keys.size(-1)), -1);
        attn_weights = dropout_->forward(attn_weights);
        return attn_weights.matmul(values);
    }

    int64_t d_out() const { return d_out_; }

    torch::nn::Linear W_query{nullptr};
    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::Tensor mask_;
    int64_t d_out_;
};
TORCH_MODULE(CausalAttention);

// ==========================================================================
// MultiHeadAttentionWrapper：堆叠多个 CausalAttention（代码清单 3-4）
// ==========================================================================
class MultiHeadAttentionWrapperImpl : public torch::nn::Module {
public:
    MultiHeadAttentionWrapperImpl(int64_t d_in, int64_t d_out,
                                  int64_t context_length, double dropout,
                                  int64_t num_heads, bool qkv_bias = false)
        : num_heads_(num_heads), d_out_(d_out) {
        for (int64_t i = 0; i < num_heads; ++i) {
            auto head = std::make_shared<CausalAttentionImpl>(
                d_in, d_out, context_length, dropout, qkv_bias);
            heads_.push_back(head);
            register_module("head_" + std::to_string(i), head);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> outs;
        outs.reserve(num_heads_);
        for (const auto& h : heads_) {
            outs.push_back(h->forward(x));
        }
        return torch::cat(outs, -1);
    }

    int64_t d_out() const { return d_out_; }

    std::vector<std::shared_ptr<CausalAttentionImpl>> heads_;
    int64_t num_heads_;
    int64_t d_out_;
};
TORCH_MODULE(MultiHeadAttentionWrapper);

// ==========================================================================
// MultiHeadAttention：权重划分 + 批矩阵乘（代码清单 3-5）
// ==========================================================================
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length,
                           double dropout, int64_t num_heads, bool qkv_bias = false)
        : d_out_(d_out), num_heads_(num_heads), head_dim_(d_out / num_heads) {
        TORCH_CHECK(d_out % num_heads == 0, "d_out must be divisible by num_heads");

        W_query = register_module("W_query",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_key = register_module("W_key",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_value = register_module("W_value",
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        out_proj = register_module("out_proj", torch::nn::Linear(d_out, d_out));
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
        mask_ = register_buffer("mask",
            torch::triu(torch::ones({context_length, context_length}), /*diagonal=*/1));
    }

    torch::Tensor forward(torch::Tensor x) {
        const int64_t b = x.size(0);
        const int64_t num_tokens = x.size(1);

        auto keys = W_key->forward(x);      // (b, n, d_out)
        auto queries = W_query->forward(x);
        auto values = W_value->forward(x);

        // 分割为多头的形状：(b, n, num_heads, head_dim) -> transpose -> (b, num_heads, n, head_dim)
        keys = keys.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);
        queries = queries.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);
        values = values.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);

        auto attn_scores = queries.matmul(keys.transpose(2, 3));  // (b, h, n, n)
        auto mask_bool = mask_.to(torch::kBool)
                             .slice(0, 0, num_tokens)
                             .slice(1, 0, num_tokens);
        attn_scores = attn_scores.masked_fill(
            mask_bool, -std::numeric_limits<double>::infinity());

        auto attn_weights = torch::softmax(
            attn_scores / std::sqrt(keys.size(-1)), -1);
        attn_weights = dropout_->forward(attn_weights);

        auto context_vec = attn_weights.matmul(values).transpose(1, 2);  // (b, n, h, hd)
        context_vec = context_vec.contiguous().view({b, num_tokens, d_out_});
        return out_proj->forward(context_vec);
    }

    torch::nn::Linear W_query{nullptr};
    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::Tensor mask_;
    int64_t d_out_;
    int64_t num_heads_;
    int64_t head_dim_;
};
TORCH_MODULE(MultiHeadAttention);

}  // namespace ch3
