// chapters/chapterE_lora/include/lora.h
// ---------------------------------------------------------------------------
// 附录 E：使用 LoRA 进行参数高效微调（Parameter-Efficient Fine-Tuning）
//
// 本文件实现附录 E 中的关键组件：
//   - LoRALayer       ：低秩分解 A·B（rank << min(in, out)），B 初始化为 0
//   - LinearWithLoRA  ：冻结的 Linear + 可训练 LoRA，前向 = Linear(x) + LoRA(x)
//   - MultiHeadAttentionWithLoRA / FeedForwardWithLoRA
//                     ：把多头注意力和 FFN 里的所有 nn.Linear 换成 LinearWithLoRA
//   - GPTModelWithLoRA：与第 4 章 GPTModel 等价的 LoRA 化版本
//   - freeze_non_lora_parameters() / count_trainable_parameters()
//                     ：冻结除 LoRA 外的所有参数，统计可训练参数数量
//
// 关键设计：
//   * 完全复用 ch4::GPTConfig / ch3::MultiHeadAttention / ch4::LayerNorm / GELU
//   * 与书中代码清单 E-3 ~ E-7 一一对应
//   * B 初始化为 0 → 训练开始时 LoRA 路径对前向输出贡献为 0，
//     因此训练前的行为与原模型完全一致。
// ---------------------------------------------------------------------------
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "attention.h"   // ch3::MultiHeadAttention（参考）
#include "gpt.h"         // ch4::{GPTConfig, LayerNorm, GELU}

namespace chE {

// ==========================================================================
// LoRALayer（对应书中 LoRALayer）：
//   ΔW ≈ A·B,    A ∈ R^{in × rank}（kaiming 初始化）
//                B ∈ R^{rank × out}（全 0 初始化，保证训练开始 ΔW = 0）
//   缩放因子 α/r：与原论文保持一致，使 rank 变化时无需重调学习率
// ==========================================================================
class LoRALayerImpl : public torch::nn::Module {
public:
    LoRALayerImpl(int64_t in_dim, int64_t out_dim, int64_t rank,
                  double alpha, double dropout = 0.0)
            : in_dim_(in_dim), out_dim_(out_dim), rank_(rank), alpha_(alpha) {
            TORCH_CHECK(rank > 0, "LoRALayer rank must be > 0 (use rank>=1)");
            // A 用 kaiming 初始化（与 nn.Linear 默认等价）
            A_ = register_parameter(
                "A", torch::empty({in_dim, rank},
                                  torch::TensorOptions().dtype(torch::kFloat32)));
            torch::nn::init::kaiming_uniform_(A_, std::sqrt(5.0));

            // B 初始化为 0 → 训练开始时 A·B = 0，前向行为不变
            B_ = register_parameter(
                "B", torch::zeros({rank, out_dim},
                                  torch::TensorOptions().dtype(torch::kFloat32)));

            if (dropout > 0.0) {
                dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
            }
        }

    // x: [..., in_dim] → [..., out_dim]
    torch::Tensor forward(torch::Tensor x) {
        if (dropout_) x = dropout_->forward(x);
        // 缩放因子 alpha/rank（与原论文公式一致）
        const double scale = alpha_ / static_cast<double>(rank_);
        // 等价 (x @ A) @ B，shape 一致：x @ A → [..., rank]；... @ B → [..., out_dim]
        return (x.matmul(A_)).matmul(B_) * scale;
    }

    int64_t in_dim()  const { return in_dim_; }
    int64_t out_dim() const { return out_dim_; }
    int64_t rank()    const { return rank_; }

    torch::Tensor A_, B_;
    torch::nn::Dropout dropout_{nullptr};

private:
    int64_t in_dim_, out_dim_, rank_;
    double alpha_;
};
TORCH_MODULE(LoRALayer);

// ==========================================================================
// LinearWithLoRA（对应书中 LinearWithLoRA）：
//   前向 = self.linear(x) + self.lora(x)
//   原 linear 的所有参数设为 requires_grad=false（被冻结），
//   只训练 lora.A / lora.B。
// ==========================================================================
class LinearWithLoRAImpl : public torch::nn::Module {
public:
    LinearWithLoRAImpl(torch::nn::Linear linear, int64_t rank,
                       double alpha, double dropout = 0.0) {
        linear_ = register_module("linear", linear);
        if (rank > 0) {
            lora_ = register_module(
                "lora",
                LoRALayer(linear->weight.size(1),  // in_features
                          linear->weight.size(0),  // out_features
                          rank, alpha, dropout));
            // 冻结原 linear 的参数（LoRA 训练开始时，前向行为不变）
            for (auto& p : linear_->parameters()) {
                p.set_requires_grad(false);
            }
        } else {
            // rank==0：纯 Linear wrapper（不接 LoRA），用于 plain 头部
            for (auto& p : linear_->parameters()) {
                p.set_requires_grad(true);
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        if (lora_) {
            return linear_->forward(x) + lora_->forward(x);
        }
        return linear_->forward(x);
    }

    // 内部访问接口（用于打印参数 / 状态字典查看）
    torch::nn::Linear linear() { return linear_; }
    LoRALayer        lora()   { return lora_;   }

    torch::nn::Linear linear_{nullptr};
    LoRALayer        lora_{nullptr};
};
TORCH_MODULE(LinearWithLoRA);

// ==========================================================================
// 工具：把 nn.Linear 替换为 LinearWithLoRA
//   替换后原 linear 的 requires_grad 自动为 false
// ==========================================================================
inline LinearWithLoRA replace_with_lora(torch::nn::Linear linear,
                                         int64_t rank,
                                         double alpha,
                                         double dropout = 0.0,
                                         const std::string& name = "") {
    (void)name;
    auto lwl = LinearWithLoRA(linear, rank, alpha, dropout);
    return lwl;
}

// ==========================================================================
// MultiHeadAttentionWithLoRA：与 ch3::MultiHeadAttention 接口完全一致，
// 但内部所有 nn.Linear（含 out_proj）都被替换为 LinearWithLoRA。
// ==========================================================================
class MultiHeadAttentionWithLoRAImpl : public torch::nn::Module {
public:
    MultiHeadAttentionWithLoRAImpl(int64_t d_in, int64_t d_out,
                                   int64_t context_length, double dropout,
                                   int64_t num_heads, int64_t rank,
                                   double alpha, bool qkv_bias = false)
        : d_out_(d_out), num_heads_(num_heads), head_dim_(d_out / num_heads) {
        TORCH_CHECK(d_out % num_heads == 0, "d_out must be divisible by num_heads");

        W_query = register_module("W_query", replace_with_lora(
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)),
            rank, alpha, dropout));
        W_key = register_module("W_key", replace_with_lora(
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)),
            rank, alpha, dropout));
        W_value = register_module("W_value", replace_with_lora(
            torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)),
            rank, alpha, dropout));
        out_proj = register_module("out_proj", replace_with_lora(
            torch::nn::Linear(d_out, d_out), rank, alpha, dropout));

        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
        mask_ = register_buffer("mask",
            torch::triu(torch::ones({context_length, context_length}),
                        /*diagonal=*/1));
    }

    torch::Tensor forward(torch::Tensor x) {
        const int64_t b = x.size(0);
        const int64_t num_tokens = x.size(1);

        auto keys = W_key->forward(x);
        auto queries = W_query->forward(x);
        auto values = W_value->forward(x);

        keys = keys.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);
        queries = queries.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);
        values = values.view({b, num_tokens, num_heads_, head_dim_}).transpose(1, 2);

        auto attn_scores = queries.matmul(keys.transpose(2, 3));
        auto mask_bool = mask_.to(torch::kBool)
                             .slice(0, 0, num_tokens)
                             .slice(1, 0, num_tokens);
        attn_scores = attn_scores.masked_fill(
            mask_bool, -std::numeric_limits<double>::infinity());

        auto attn_weights = torch::softmax(
            attn_scores / std::sqrt(keys.size(-1)), -1);
        attn_weights = dropout_->forward(attn_weights);

        auto context_vec = attn_weights.matmul(values).transpose(1, 2);
        context_vec = context_vec.contiguous().view({b, num_tokens, d_out_});
        return out_proj->forward(context_vec);
    }

    LinearWithLoRA W_query{nullptr}, W_key{nullptr}, W_value{nullptr}, out_proj{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::Tensor mask_;

private:
    int64_t d_out_, num_heads_, head_dim_;
};
TORCH_MODULE(MultiHeadAttentionWithLoRA);

// ==========================================================================
// FeedForwardWithLoRA：与 ch4::FeedForward 接口一致，内部 Linear 都换成 LoRA 版
// ==========================================================================
class FeedForwardWithLoRAImpl : public torch::nn::Module {
public:
    FeedForwardWithLoRAImpl(const ch4::GPTConfig& cfg,
                            int64_t rank, double alpha, double dropout = 0.0) {
        layers = register_module("layers", torch::nn::Sequential());
        layers->push_back(replace_with_lora(
            torch::nn::Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            rank, alpha, dropout));
        layers->push_back(ch4::GELU());
        layers->push_back(replace_with_lora(
            torch::nn::Linear(4 * cfg.emb_dim, cfg.emb_dim),
            rank, alpha, dropout));
    }

    torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }

    torch::nn::Sequential layers{nullptr};
};
TORCH_MODULE(FeedForwardWithLoRA);

// ==========================================================================
// TransformerBlockWithLoRA：与 ch4::TransformerBlock 接口一致
// ==========================================================================
class TransformerBlockWithLoRAImpl : public torch::nn::Module {
public:
    TransformerBlockWithLoRAImpl(const ch4::GPTConfig& cfg,
                                 int64_t rank, double alpha) {
        att = register_module("att", MultiHeadAttentionWithLoRA(
            cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.drop_rate,
            cfg.n_heads, rank, alpha, cfg.qkv_bias));
        ff = register_module("ff", FeedForwardWithLoRA(cfg, rank, alpha, cfg.drop_rate));
        norm1 = register_module("norm1", ch4::LayerNorm(cfg.emb_dim));
        norm2 = register_module("norm2", ch4::LayerNorm(cfg.emb_dim));
        drop_shortcut = register_module("drop_shortcut", torch::nn::Dropout(cfg.drop_rate));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto shortcut = x;
        x = norm1->forward(x);
        x = att->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;

        shortcut = x;
        x = norm2->forward(x);
        x = ff->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;
        return x;
    }

    MultiHeadAttentionWithLoRA att{nullptr};
    FeedForwardWithLoRA ff{nullptr};
    ch4::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout drop_shortcut{nullptr};
};
TORCH_MODULE(TransformerBlockWithLoRA);

// ==========================================================================
// GPTModelWithLoRA：与 ch4::GPTModel 接口一致；
//   输出头也可选地替换为 LinearWithLoRA（默认开启，与书中一致）
// ==========================================================================
class GPTModelWithLoRAImpl : public torch::nn::Module {
public:
    GPTModelWithLoRAImpl(const ch4::GPTConfig& cfg,
                         int64_t rank, double alpha,
                         bool lora_out_head = true) {
        tok_emb = register_module("tok_emb",
            torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim));
        pos_emb = register_module("pos_emb",
            torch::nn::Embedding(cfg.context_length, cfg.emb_dim));
        drop_emb = register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate));

        trf_blocks = register_module("trf_blocks", torch::nn::Sequential());
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            trf_blocks->push_back(TransformerBlockWithLoRA(cfg, rank, alpha));
        }

        final_norm = register_module("final_norm", ch4::LayerNorm(cfg.emb_dim));

        // 构造输出头（暂不 register，便于后续替换为分类头）
        if (lora_out_head) {
            out_head = replace_with_lora(
                torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size)
                                      .bias(false)),
                rank, alpha, /*dropout=*/0.0);
        } else {
            out_head = LinearWithLoRA(
                torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size)
                                      .bias(false)),
                /*rank=*/0, /*alpha=*/1.0, /*dropout=*/0.0);
        }
        register_module("out_head", out_head);
    }

    torch::Tensor forward(torch::Tensor in_idx) {
        return out_head->forward(forward_hidden(in_idx));
    }

    torch::Tensor forward_hidden(torch::Tensor in_idx) {
        const int64_t batch_size = in_idx.size(0);
        const int64_t seq_len = in_idx.size(1);
        auto tok_embeds = tok_emb->forward(in_idx);
        auto pos_embeds = pos_emb->forward(torch::arange(
            seq_len, torch::TensorOptions().device(in_idx.device()).dtype(torch::kLong)));
        auto x = tok_embeds + pos_embeds;
        x = drop_emb->forward(x);
        x = trf_blocks->forward(x);
        x = final_norm->forward(x);
        return x;
    }

    // 暴露所有 LinearWithLoRA，便于遍历冻结 / 收集
    std::vector<LinearWithLoRA> all_lora_linears() {
        std::vector<LinearWithLoRA> out;
        for (int64_t i = 0; i < trf_blocks->size(); ++i) {
            auto& blk = trf_blocks->at<TransformerBlockWithLoRAImpl>(i);
            out.push_back(blk.att->W_query);
            out.push_back(blk.att->W_key);
            out.push_back(blk.att->W_value);
            out.push_back(blk.att->out_proj);
            // FFN 内的两个 LinearWithLoRA 通过 named_parameters 也能遍历到，
            // 这里就不再单独 collect。
        }
        return out;
    }

    // 分类微调用：替换 out_head 为 num_classes 维 LinearWithLoRA
    void replace_out_head_for_classification(int64_t num_classes,
                                             int64_t rank, double alpha) {
        auto new_head = replace_with_lora(
            torch::nn::Linear(torch::nn::LinearOptions(
                /*in_features=*/final_norm->scale_.size(0), num_classes).bias(true)),
            rank, alpha, /*dropout=*/0.0);
        out_head = new_head;
        // 用 replace_module 直接覆盖
        replace_module("out_head", out_head);
    }

    // 分类微调用：直接替换为普通 Linear（参数保持固定）
    void replace_out_head_plain(int64_t num_classes) {
        LinearWithLoRA new_head(
            torch::nn::Linear(torch::nn::LinearOptions(
                final_norm->scale_.size(0), num_classes).bias(true)),
            /*rank=*/0, /*alpha=*/1.0, /*dropout=*/0.0);
        // 把 plain head 的参数冻结（仅训练 LoRA 时使用）
        for (auto& p : new_head->linear()->parameters()) p.set_requires_grad(false);
        out_head = new_head;
        replace_module("out_head", out_head);
    }

    torch::nn::Embedding tok_emb{nullptr}, pos_emb{nullptr};
    torch::nn::Dropout drop_emb{nullptr};
    torch::nn::Sequential trf_blocks{nullptr};
    ch4::LayerNorm final_norm{nullptr};
    LinearWithLoRA out_head{nullptr};
};
TORCH_MODULE(GPTModelWithLoRA);

// ==========================================================================
// 工具：把模型所有 requires_grad=true 的参数（除 LoRA 内部 A、B 外）冻结
//   与书中代码清单 E-6 等价：
//     for name, param in model.named_parameters():
//         if "weight" in name: param.requires_grad = False
//         elif "lora"  in name: param.requires_grad = True
// ==========================================================================
inline int64_t freeze_non_lora_parameters(torch::nn::Module& model) {
    int64_t trainable = 0;
    for (auto& pair : model.named_parameters(/*recurse=*/true)) {
        const std::string& name = pair.key();
        auto& p = pair.value();
        const bool is_lora = (name.find(".lora.A") != std::string::npos) ||
                             (name.find(".lora.B") != std::string::npos);
        if (is_lora) {
            p.set_requires_grad(true);
            trainable += p.numel();
        } else {
            p.set_requires_grad(false);
        }
    }
    return trainable;
}

// 统计总参 / 可训练参（用 named_parameters 走一遍，不依赖 options）
inline std::pair<int64_t, int64_t>
count_trainable_parameters(torch::nn::Module& model) {
    int64_t total = 0, trainable = 0;
    for (auto& pair : model.named_parameters(/*recurse=*/true)) {
        total += pair.value().numel();
        if (pair.value().requires_grad()) {
            trainable += pair.value().numel();
        }
    }
    return {trainable, total};
}

}  // namespace chE
