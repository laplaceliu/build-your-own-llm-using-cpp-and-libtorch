// main.cpp
// 第 3 章：编码注意力机制（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印，方便对照验证：
//   3.3  简单自注意力（无权重）与全输入推广
//   3.4  带可训练权重的自注意力（SelfAttention_v1 / v2）
//   3.5  因果注意力（掩码 + dropout，CausalAttention）
//   3.6  多头注意力（Wrapper 与权重划分 MultiHeadAttention）
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "attention.h"

namespace {

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

// 书中 3.3 节的 6 词输入："Your journey starts with one step."
torch::Tensor make_inputs() {
    return torch::tensor({{0.43f, 0.15f, 0.89f},  // Your   (x^1)
                          {0.55f, 0.87f, 0.66f},  // journey(x^2)
                          {0.57f, 0.85f, 0.64f},  // starts (x^3)
                          {0.22f, 0.58f, 0.33f},  // with   (x^4)
                          {0.77f, 0.25f, 0.10f},  // one    (x^5)
                          {0.05f, 0.80f, 0.55f}}  // step   (x^6)
    );
}

// ---- 3.3.1 没有可训练权重的简单自注意力 ----
void demo_naive_self_attention(const torch::Tensor& inputs) {
    section("3.3.1 没有可训练权重的简单自注意力");
    std::cout << "inputs (6x3):\n" << inputs << "\n";

    // 第 2 个输入词元作为查询
    torch::Tensor query = inputs.index({1});
    std::cout << "query = inputs[1] = " << query << "\n";

    // 注意力分数：query 与每个输入的 dot 积
    std::vector<torch::Tensor> scores;
    for (int64_t i = 0; i < inputs.size(0); ++i) {
        scores.push_back(torch::dot(inputs.index({i}), query));
    }
    auto attn_scores_2 = torch::stack(scores);
    std::cout << "attn_scores_2 = " << attn_scores_2 << "\n";

    // 归一化：除以总和
    auto attn_weights_tmp = attn_scores_2 / attn_scores_2.sum();
    std::cout << "Attention weights (sum 归一化): " << attn_weights_tmp << "\n";
    std::cout << "Sum: " << attn_weights_tmp.sum() << "\n";

    // softmax 归一化（naive）
    auto exp_scores = torch::exp(attn_scores_2);
    auto attn_weights_naive = exp_scores / exp_scores.sum(0);
    std::cout << "Attention weights (softmax_naive): " << attn_weights_naive << "\n";
    std::cout << "Sum: " << attn_weights_naive.sum() << "\n";

    // 推荐：torch::softmax
    auto attn_weights_2 = torch::softmax(attn_scores_2, 0);
    std::cout << "Attention weights (torch.softmax): " << attn_weights_2 << "\n";
    std::cout << "Sum: " << attn_weights_2.sum() << "\n";

    // 上下文向量 z^2
    torch::Tensor context_vec_2 = torch::zeros_like(query);
    for (int64_t i = 0; i < inputs.size(0); ++i) {
        context_vec_2 += attn_weights_2.index({i}) * inputs.index({i});
    }
    std::cout << "context_vec_2 = " << context_vec_2 << "\n";
}

// ---- 3.3.2 计算所有输入词元的注意力权重 ----
void demo_all_context_vectors(const torch::Tensor& inputs) {
    section("3.3.2 计算所有输入词元的注意力权重");

    // 双循环点积
    std::vector<torch::Tensor> rows;
    for (int64_t i = 0; i < inputs.size(0); ++i) {
        std::vector<torch::Tensor> r;
        for (int64_t j = 0; j < inputs.size(0); ++j) {
            r.push_back(torch::dot(inputs.index({i}), inputs.index({j})));
        }
        rows.push_back(torch::stack(r));
    }
    auto attn_scores_loop = torch::stack(rows);
    std::cout << "attn_scores (for 循环):\n" << attn_scores_loop << "\n";

    // 矩阵乘法 inputs @ inputs.T
    auto attn_scores = inputs.matmul(inputs.t());
    std::cout << "attn_scores (inputs @ inputs.T):\n" << attn_scores << "\n";

    // softmax 归一化每一行
    auto attn_weights = torch::softmax(attn_scores, -1);
    std::cout << "attn_weights (softmax dim=-1):\n" << attn_weights << "\n";
    std::cout << "所有行之和: " << attn_weights.sum(-1) << "\n";

    // 上下文向量 = attn_weights @ inputs
    auto all_context_vecs = attn_weights.matmul(inputs);
    std::cout << "all_context_vecs (6x3):\n" << all_context_vecs << "\n";
    std::cout << "第 2 行（与 3.3.1 context_vec_2 一致）: "
              << all_context_vecs.index({1}) << "\n";
}

// ---- 3.4.1 逐步计算带可训练权重的注意力 ----
void demo_trainable_step_by_step(const torch::Tensor& inputs) {
    section("3.4.1 逐步计算（可训练权重）");
    const int64_t d_in = inputs.size(1);  // 3
    const int64_t d_out = 2;

    torch::manual_seed(123);
    auto W_query = torch::rand({d_in, d_out});
    auto W_key = torch::rand({d_in, d_out});
    auto W_value = torch::rand({d_in, d_out});
    std::cout << "W_query (3x2):\n" << W_query << "\n";

    // 查询/键/值向量
    auto x_2 = inputs.index({1});
    auto query_2 = x_2.matmul(W_query);
    auto key_2 = x_2.matmul(W_key);
    auto value_2 = x_2.matmul(W_value);
    std::cout << "query_2 = " << query_2 << "\n";

    auto keys = inputs.matmul(W_key);
    auto values = inputs.matmul(W_value);
    std::cout << "keys.shape = " << keys.sizes() << "\n";
    std::cout << "values.shape = " << values.sizes() << "\n";

    // 注意力分数
    auto attn_score_22 = torch::dot(query_2, keys.index({1}));
    std::cout << "attn_score_22 = " << attn_score_22 << "\n";
    auto attn_scores_2 = query_2.matmul(keys.t());
    std::cout << "attn_scores_2 = " << attn_scores_2 << "\n";

    // 缩放 softmax
    auto d_k = static_cast<double>(keys.size(-1));
    auto attn_weights_2 = torch::softmax(attn_scores_2 / std::sqrt(d_k), -1);
    std::cout << "attn_weights_2 = " << attn_weights_2 << "\n";

    // 上下文向量
    auto context_vec_2 = attn_weights_2.matmul(values);
    std::cout << "context_vec_2 = " << context_vec_2 << "\n";
}

// ---- 3.4.2 SelfAttention 类 ----
void demo_self_attention_classes(const torch::Tensor& inputs) {
    section("3.4.2 SelfAttention_v1 / SelfAttention_v2 类");
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;

    torch::manual_seed(123);
    ch3::SelfAttentionV1 sa_v1(d_in, d_out);
    std::cout << "sa_v1(inputs):\n" << sa_v1->forward(inputs) << "\n";

    torch::manual_seed(789);
    ch3::SelfAttentionV2 sa_v2(d_in, d_out);
    std::cout << "sa_v2(inputs):\n" << sa_v2->forward(inputs) << "\n";

    // 练习 3.1：把 v2（nn.Linear，权重转置存储）的权重赋给 v1
    torch::manual_seed(789);
    ch3::SelfAttentionV2 sa_v2b(d_in, d_out);
    ch3::SelfAttentionV1 sa_v1b(d_in, d_out);
    sa_v1b->W_query = sa_v2b->W_query->weight.t();
    sa_v1b->W_key = sa_v2b->W_key->weight.t();
    sa_v1b->W_value = sa_v2b->W_value->weight.t();
    std::cout << "\n[练习 3.1] 把 v2 的 Linear 权重转置后赋给 v1\n";
    std::cout << "sa_v1(赋 v2 权重)(inputs):\n" << sa_v1b->forward(inputs) << "\n";
    std::cout << "sa_v2(inputs):\n" << sa_v2b->forward(inputs) << "\n";
}

// ---- 3.5.1 因果注意力掩码 ----
void demo_causal_mask(const torch::Tensor& inputs) {
    section("3.5.1 因果注意力的掩码实现");
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;

    torch::manual_seed(789);
    ch3::SelfAttentionV2 sa_v2(d_in, d_out);

    auto queries = sa_v2->W_query->forward(inputs);
    auto keys = sa_v2->W_key->forward(inputs);
    auto attn_scores = queries.matmul(keys.t());
    auto attn_weights = torch::softmax(
        attn_scores / std::sqrt(static_cast<double>(keys.size(-1))), -1);
    std::cout << "attn_weights (softmax):\n" << attn_weights << "\n";

    // 第 (2) 步：tril 掩码
    const int64_t context_length = attn_scores.size(0);
    auto mask_simple = torch::tril(torch::ones({context_length, context_length}));
    std::cout << "mask_simple (tril):\n" << mask_simple << "\n";

    auto masked_simple = attn_weights * mask_simple;
    std::cout << "masked_simple = attn_weights * mask_simple:\n"
              << masked_simple << "\n";

    // 第 (3) 步：行归一化
    auto row_sums = masked_simple.sum(-1, /*keepdim=*/true);
    auto masked_simple_norm = masked_simple / row_sums;
    std::cout << "masked_simple_norm:\n" << masked_simple_norm << "\n";

    // 更高效：softmax 前用 -inf 掩码
    auto mask = torch::triu(torch::ones({context_length, context_length}), 1);
    auto masked = attn_scores.masked_fill(mask.to(torch::kBool),
                                          -std::numeric_limits<double>::infinity());
    std::cout << "masked (attn_scores + -inf):\n" << masked << "\n";
    auto attn_weights_inf = torch::softmax(
        masked / std::sqrt(static_cast<double>(keys.size(-1))), 1);
    std::cout << "attn_weights (softmax(masked)):\n" << attn_weights_inf << "\n";
}

// ---- 3.5.2 dropout ----
void demo_dropout(const torch::Tensor& inputs) {
    section("3.5.2 利用 dropout 掩码额外的注意力权重");
    torch::manual_seed(123);
    auto dropout = torch::nn::Dropout(0.5);
    auto example = torch::ones({6, 6});
    std::cout << "dropout(ones(6,6)) 50%:\n" << dropout->forward(example) << "\n";

    // 对因果注意力权重做 dropout
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;
    torch::manual_seed(789);
    ch3::SelfAttentionV2 sa_v2(d_in, d_out);
    auto queries = sa_v2->W_query->forward(inputs);
    auto keys = sa_v2->W_key->forward(inputs);
    auto attn_scores = queries.matmul(keys.t());
    const int64_t context_length = attn_scores.size(0);
    auto mask = torch::triu(torch::ones({context_length, context_length}), 1);
    auto masked = attn_scores.masked_fill(mask.to(torch::kBool),
                                          -std::numeric_limits<double>::infinity());
    auto attn_weights = torch::softmax(
        masked / std::sqrt(static_cast<double>(keys.size(-1))), 1);

    torch::manual_seed(123);
    std::cout << "dropout(attn_weights) 50%:\n" << dropout->forward(attn_weights) << "\n";
}

// ---- 3.5.3 CausalAttention 类 ----
void demo_causal_attention_class(const torch::Tensor& inputs) {
    section("3.5.3 简化的因果注意力类 CausalAttention");
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;

    auto batch = torch::stack({inputs, inputs}, 0);
    std::cout << "batch.shape = " << batch.sizes() << "\n";

    torch::manual_seed(123);
    const int64_t context_length = batch.size(1);
    ch3::CausalAttention ca(d_in, d_out, context_length, /*dropout=*/0.0);
    auto context_vecs = ca->forward(batch);
    std::cout << "context_vecs.shape = " << context_vecs.sizes() << "\n";
    std::cout << "context_vecs:\n" << context_vecs << "\n";
}

// ---- 3.6.1 叠加多个单头注意力 ----
void demo_mha_wrapper(const torch::Tensor& inputs) {
    section("3.6.1 多头注意力（MultiHeadAttentionWrapper）");
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;
    auto batch = torch::stack({inputs, inputs}, 0);
    const int64_t context_length = batch.size(1);

    torch::manual_seed(123);
    ch3::MultiHeadAttentionWrapper mha(d_in, d_out, context_length, /*dropout=*/0.0,
                                       /*num_heads=*/2);
    auto context_vecs = mha->forward(batch);
    std::cout << "context_vecs.shape = " << context_vecs.sizes() << "\n";
    std::cout << "context_vecs:\n" << context_vecs << "\n";

    // 练习 3.2：d_out=1 -> 输出 [2,6,2]
    torch::manual_seed(123);
    ch3::MultiHeadAttentionWrapper mha2(d_in, /*d_out=*/1, context_length,
                                        /*dropout=*/0.0, /*num_heads=*/2);
    auto cv2 = mha2->forward(batch);
    std::cout << "\n[练习 3.2] d_out=1, num_heads=2 -> shape = " << cv2.sizes() << "\n";
}

// ---- 3.6.2 权重划分的多头注意力 ----
void demo_mha_weight_split(const torch::Tensor& inputs) {
    section("3.6.2 通过权重划分实现多头注意力");
    const int64_t d_in = inputs.size(1);
    const int64_t d_out = 2;
    auto batch = torch::stack({inputs, inputs}, 0);
    const int64_t context_length = batch.size(1);

    // 批矩阵乘演示（书中 (b,h,n,d)=(1,2,3,4) 的例子）
    auto a = torch::tensor({{{{0.2745f, 0.6584f, 0.2775f, 0.8573f},
                              {0.8993f, 0.0390f, 0.9268f, 0.7388f},
                              {0.7179f, 0.7058f, 0.9156f, 0.4340f}},
                             {{0.0772f, 0.3565f, 0.1479f, 0.5331f},
                              {0.4066f, 0.2318f, 0.4545f, 0.9737f},
                              {0.4606f, 0.5159f, 0.4220f, 0.5786f}}}});
    std::cout << "a.shape = " << a.sizes() << "\n";
    auto batched = a.matmul(a.transpose(2, 3));
    std::cout << "a @ a.transpose(2,3):\n" << batched << "\n";
    auto first_head = a.index({0, 0, torch::indexing::Slice(), torch::indexing::Slice()});
    std::cout << "first_head @ first_head.T:\n" << first_head.matmul(first_head.t()) << "\n";
    auto second_head = a.index({0, 1, torch::indexing::Slice(), torch::indexing::Slice()});
    std::cout << "second_head @ second_head.T:\n"
              << second_head.matmul(second_head.t()) << "\n";

    // MultiHeadAttention 类
    torch::manual_seed(123);
    ch3::MultiHeadAttention mha(d_in, d_out, context_length, /*dropout=*/0.0,
                                /*num_heads=*/2);
    auto context_vecs = mha->forward(batch);
    std::cout << "context_vecs.shape = " << context_vecs.sizes() << "\n";
    std::cout << "context_vecs:\n" << context_vecs << "\n";

    // 练习 3.3：GPT-2 大小
    torch::manual_seed(123);
    ch3::MultiHeadAttention gpt2_mha(/*d_in=*/768, /*d_out=*/768, /*context_length=*/1024,
                                     /*dropout=*/0.0, /*num_heads=*/12);
    auto probe = torch::randn({2, 8, 768});
    auto out = gpt2_mha->forward(probe);
    std::cout << "\n[练习 3.3] GPT-2 规模 (d=768, heads=12, context=1024)\n";
    std::cout << "  输入 [2,8,768] -> 输出 " << out.sizes() << "\n";
    std::cout << "  参数数量: " << gpt2_mha->parameters().size() << " 组参数\n";
}

}  // namespace

int main() {
    try {
        std::cout << "=== 第 3 章：编码注意力机制（C++ + LibTorch）===\n";
        auto inputs = make_inputs();

        demo_naive_self_attention(inputs);
        demo_all_context_vectors(inputs);
        demo_trainable_step_by_step(inputs);
        demo_self_attention_classes(inputs);
        demo_causal_mask(inputs);
        demo_dropout(inputs);
        demo_causal_attention_class(inputs);
        demo_mha_wrapper(inputs);
        demo_mha_weight_split(inputs);

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 3 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
