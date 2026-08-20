// training.h
// 附录 D：增强 LLM 的训练循环功能
//   D.1 学习率预热（线性预热）
//   D.2 余弦衰减
//   D.3 梯度裁剪
//   D.4 修改的训练函数
//
// 对应书中代码：
//   代码清单 D.1：带预热的余弦衰减学习率调度
//   代码清单 D.2：替换 train_model_simple 的 train_model
//
// 所有函数都是 header-only 风格，跟前面章节保持一致。
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "dataloader.h"
#include "gpt.h"
#include "training.h"  // ch5::calc_loss_batch / evaluate_model / generate_and_print_sample

namespace chD {

// ----------------------------------------------------------------------------
// D.1 / D.2 — 学习率预热 + 余弦衰减  (代码清单 D.1)
// ----------------------------------------------------------------------------
//
// 公式：
//   lr(step) = min( lr_max * (step+1) / warmup_steps ,
//                   lr_min + 0.5 * (lr_max - lr_min) *
//                            (1 + cos(pi * (step - warmup_steps) / (max_steps - warmup_steps))) )
//
// 当 step <  warmup_steps：线性递增（避免早期梯度爆炸导致训练不稳）
// 当 step >= warmup_steps：余弦衰减（前期衰减慢，后期衰减快）
// 当 step >  max_steps  ：退化为 lr_min（恒定下限，避免 lr 出现负值）
inline double get_lr(int64_t step, double warmup_steps, double max_steps,
                     double lr_min, double lr_max) {
    // 1) 线性预热阶段
    if (step < warmup_steps) {
        return lr_max * (static_cast<double>(step) + 1.0) / warmup_steps;
    }
    // 2) 超过 max_steps 直接退化为 lr_min（避免 cos 内参数变为负数）
    if (step > max_steps) {
        return lr_min;
    }
    // 3) 余弦衰减阶段
    double decay_ratio =
        static_cast<double>(step - warmup_steps) / (max_steps - warmup_steps);
    double coeff = 0.5 * (1.0 + std::cos(M_PI * decay_ratio));
    return lr_min + coeff * (lr_max - lr_min);
}

// 简易打印：将当前 step 的学习率打印出来（便于观测）
inline void print_lr(int64_t step, double warmup_steps, double max_steps,
                     double lr_min, double lr_max, int64_t eval_every) {
    if (step % eval_every == 0) {
        std::cout << "  step " << step
                  << " | lr = " << get_lr(step, warmup_steps, max_steps, lr_min, lr_max)
                  << "\n";
    }
}

// ----------------------------------------------------------------------------
// D.3 — 梯度裁剪
// ----------------------------------------------------------------------------
// 计算所有参数的 L2 范数，按 max_norm 做缩放。
// PyTorch 等价：torch.nn.utils.clip_grad_norm_(parameters, max_norm)
inline double calc_norm_of_grads(const std::vector<torch::Tensor>& parameters) {
    double total_sq = 0.0;
    for (const auto& p : parameters) {
        if (p.grad().defined()) {
            auto n = p.grad().norm(2).item<double>();
            total_sq += n * n;
        }
    }
    return std::sqrt(total_sq);
}

// 在原地做 L2-norm 裁剪，返回裁剪前的范数（用于观测）
inline double clip_gradients(std::vector<torch::Tensor> parameters,
                             double max_norm) {
    double total_norm = calc_norm_of_grads(parameters);
    double clip_factor = max_norm / (total_norm + 1e-5);
    if (clip_factor < 1.0) {
        for (auto& p : parameters) {
            if (p.grad().defined()) {
                p.grad().mul_(clip_factor);
            }
        }
    }
    return total_norm;
}

// ----------------------------------------------------------------------------
// D.4 — 修改的训练函数 train_model
// ----------------------------------------------------------------------------
// 与 ch5::train_model_simple 的区别：
//   - 每步都重新计算学习率（warmup + cosine）
//   - 梯度裁剪（max_norm）
//   - 打印每步的 norm 和 lr，方便观测
//
// 其它逻辑（eval、生成样本、记录 losses）保持不变。
inline std::pair<std::vector<double>, std::vector<double>> train_model(
    ch4::GPTModel& model,
    const std::vector<ch2::GPTBatch>& train_loader,
    const std::vector<ch2::GPTBatch>& val_loader,
    ch2::BpeTokenizer& tokenizer,
    int64_t num_epochs, int64_t eval_freq, int64_t eval_iter,
    int64_t warmup_steps, int64_t max_steps,
    double lr_min, double lr_max, double max_norm,
    const std::string& start_context = "Every effort moves you") {

    std::vector<double> train_losses;
    std::vector<double> val_losses;

    int64_t global_step = -1;
    int64_t total_steps = max_steps;  // 一个 epoch 走 max_steps 步

    // 收集所有可训练参数（对应 PyTorch 的 model.parameters()）
    std::vector<torch::Tensor> parameters;
    for (auto& p : model->parameters()) {
        if (p.requires_grad()) parameters.push_back(p);
    }

    torch::optim::AdamW optimizer(parameters,
                                 torch::optim::AdamWOptions(lr_max).betas({0.9, 0.95}).weight_decay(0.1));

    int64_t tokens_seen = 0;

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        std::cout << "Epoch " << (epoch + 1) << "\n";

        for (size_t i = 0; i < train_loader.size(); ++i) {
            // ---------- 1) 重新计算学习率 ----------
            double lr = get_lr(global_step, warmup_steps, max_steps, lr_min, lr_max);
            for (auto& g : optimizer.param_groups()) {
                static_cast<torch::optim::AdamWOptions&>(g.options()).lr(lr);
            }

            // ---------- 2) 前向 + 反向 ----------
            optimizer.zero_grad();
            auto& batch = train_loader[i];
            auto loss = ch5::calc_loss_batch(batch.inputs, batch.targets, model);
            loss.backward();

            // ---------- 3) 梯度裁剪 ----------
            double norm = clip_gradients(parameters, max_norm);

            // ---------- 4) 优化器步进 ----------
            optimizer.step();
            tokens_seen += batch.inputs.numel();
            global_step += 1;

            // ---------- 5) 周期性评估 ----------
            if (global_step % eval_freq == 0) {
                auto [tl, vl] = ch5::evaluate_model(model, train_loader, val_loader, eval_iter);
                std::cout << "Ep " << (epoch + 1)
                          << " | step " << global_step
                          << " | lr " << lr
                          << " | grad_norm " << norm
                          << " | train_loss " << tl
                          << " | val_loss " << vl
                          << "\n";
                train_losses.push_back(tl);
                val_losses.push_back(vl);
            }

            // ---------- 6) 周期性采样 ----------
            if (global_step % (5 * eval_freq) == 0 && global_step != 0) {
                ch5::generate_and_print_sample(model, tokenizer, start_context);
            }

            if (global_step >= total_steps) {
                break;
            }
        }
        if (global_step >= total_steps) break;
    }
    return {train_losses, val_losses};
}

}  // namespace chD