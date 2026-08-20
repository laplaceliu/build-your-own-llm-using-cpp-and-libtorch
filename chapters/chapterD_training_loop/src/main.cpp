// main.cpp — 附录 D 演示入口
// 主题：在小语料上展示学习率预热 + 余弦衰减 + 梯度裁剪 三件套的效果。
//
// 注意：本章不下载 GPT-2 完整权重（避免和小语料过拟合的演示目的冲突），
//      而是使用 GPT-2 124M 的小规模配置 + 随机初始化，专注演示训练循环
//      优化技巧本身的数值行为。
//
// 跑法：
//   cd build && cmake .. && make chapterD_training_loop
//   ./chapterD_training_loop            # CPU
//   ./chapterD_training_loop --cuda     # GPU（若可用）
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gpt.h"
#include "bpe_tokenizer.h"
#include "dataloader.h"
#include "training.h"   // ch5::{calc_loss_batch, evaluate_model, generate_and_print_sample}
#include "training_D.h"

namespace {

ch4::GPTConfig make_demo_config() {
    ch4::GPTConfig c;
    c.vocab_size = 50257;
    c.context_length = 256;
    c.emb_dim = 256;     // 缩小的 demo 配置
    c.n_heads = 4;
    c.n_layers = 4;
    c.drop_rate = 0.0;
    c.qkv_bias = false;
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    bool use_cuda = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cuda") use_cuda = true;
    }

    torch::manual_seed(123);
    auto device = torch::cuda::is_available() && use_cuda
                      ? torch::Device(torch::kCUDA)
                      : torch::Device(torch::kCPU);
    std::cout << "[appendix-D] device = " << device << "\n";

    // 强制 CUDA 精度（与第 6/7 章保持一致，避免 tf32 带来的非确定性）
    if (device.is_cuda()) {
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
    }

    // ---------- 1) 构造一个小型 GPT-2 模型 ----------
    auto cfg = make_demo_config();
    ch4::GPTModel model(cfg);
    model->to(device);
    std::cout << "[appendix-D] model constructed (vocab=" << cfg.vocab_size
              << ", ctx=" << cfg.context_length << ", d=" << cfg.emb_dim
              << ", layers=" << cfg.n_layers << ", heads=" << cfg.n_heads << ")\n";

    // ---------- 2) 准备一个小语料（"the verdict" 风格短文） ----------
    std::string text =
        "Every effort moves you forward. Every step is a step toward a goal.\n"
        "Hard work beats talent when talent doesn't work hard.\n"
        "The future belongs to those who believe in the beauty of their dreams.\n"
        "The only way to do great work is to love what you do.\n"
        "If you can dream it, you can do it. Your time is limited, so don't waste it.\n"
        "Success is not final, failure is not fatal: it is the courage to continue that counts.\n"
        "Believe you can and you're halfway there. The mind is everything.\n"
        "What you get by achieving your goals is not as important as what you become.\n";

    // 重复几次，让训练样本多一些
    std::string corpus;
    for (int i = 0; i < 30; ++i) corpus += text;

    // ---------- 3) BPE 分词 ----------
#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
    ch2::BpeTokenizer tokenizer(std::string(DATA_DIR) + "/encoder.json",
                                std::string(DATA_DIR) + "/vocab.bpe");
    auto ids = tokenizer.encode(corpus, {"<|endoftext|>"});
    std::cout << "[appendix-D] tokens = " << ids.size() << "\n";

    // ---------- 4) 构造数据加载器 ----------
    ch2::GPTDataLoader train_loader(ids, /*batch=*/4, /*max_length=*/128,
                                    /*stride=*/128, /*shuffle=*/true,
                                    /*drop_last=*/true, /*seed=*/123);
    ch2::GPTDataLoader val_loader(ids, /*batch=*/4, /*max_length=*/128,
                                  /*stride=*/128, /*shuffle=*/false,
                                  /*drop_last=*/false, /*seed=*/123);
    std::cout << "[appendix-D] train batches = " << train_loader.num_batches()
              << ", val batches = " << val_loader.num_batches() << "\n";

    // ---------- 5) 演示 D.1/D.2：学习率调度曲线 ----------
    std::cout << "\n=== D.1 + D.2: learning-rate schedule ===\n";
    int64_t total_steps = 30;
    int64_t warmup_steps = 10;
    double lr_max = 5e-4;
    double lr_min = 1e-6;

    std::cout << "step   lr\n";
    for (int64_t s = 0; s <= total_steps + 10; ++s) {
        double lr = chD::get_lr(s, warmup_steps, total_steps, lr_min, lr_max);
        std::cout << s << "  " << lr << "\n";
    }

    // ---------- 6) 训练（warmup + cosine + grad-clipping） ----------
    std::cout << "\n=== D.4: training with the new schedule ===\n";
    auto [train_losses, val_losses] = chD::train_model(
        model, train_loader.batches(), val_loader.batches(), tokenizer,
        /*num_epochs=*/1,
        /*eval_freq=*/5,  /*eval_iter=*/2,
        /*warmup_steps=*/10,
        /*max_steps=*/total_steps,
        /*lr_min=*/lr_min, /*lr_max=*/lr_max,
        /*max_norm=*/1.0,
        /*start_context=*/"Every effort moves you");

    // ---------- 7) 对比实验：关闭裁剪 / 关闭预热，看 loss 曲线 ----------
    std::cout << "\n=== D.3 ablation: turning off gradient clipping ===\n";
    torch::manual_seed(123);  // 同样初始化
    ch4::GPTModel model_no_clip(cfg);
    model_no_clip->to(device);

    // 复制初始参数以便在对照组实验中保持起点一致
    auto src_params = model->parameters();
    auto dst_params = model_no_clip->parameters();
    for (size_t i = 0; i < src_params.size(); ++i) {
        torch::NoGradGuard g;
        dst_params[i].copy_(src_params[i]);
    }

    // 复用 train_model，但通过 max_norm=0 关闭裁剪：clip_gradients 中
    // max_norm=0 仍然会执行 norm 计算，不会缩放任何梯度（clip_factor=1）。
    // 为了真正"关闭"裁剪，我们直接调一个最小版本：
    torch::optim::AdamW opt_nc(dst_params,
                               torch::optim::AdamWOptions(lr_max).betas({0.9, 0.95}).weight_decay(0.1));
    std::vector<torch::Tensor> params_nc;
    for (auto& p : model_no_clip->parameters()) if (p.requires_grad()) params_nc.push_back(p);

    int64_t global_step = -1;
    double total_loss_clip = 0.0, total_loss_noclip = 0.0;
    int64_t eval_every = 5;
    for (size_t i = 0; i < train_loader.batches().size(); ++i) {
        double lr = chD::get_lr(global_step, warmup_steps, total_steps, lr_min, lr_max);
        for (auto& g : opt_nc.param_groups()) static_cast<torch::optim::AdamWOptions&>(g.options()).lr(lr);

        auto& batch = train_loader.batches()[i];
        opt_nc.zero_grad();
        auto loss = ch5::calc_loss_batch(batch.inputs, batch.targets, model_no_clip);
        loss.backward();
        // ★ 与 train_model 不同：这里不裁剪
        opt_nc.step();
        global_step += 1;
        if (global_step % eval_every == 0 && global_step < total_steps) {
            auto tl = ch5::calc_loss_loader(model_no_clip, train_loader.batches(), 2);
            std::cout << "[no-clip] step " << global_step << " lr " << lr
                      << " train_loss " << tl << "\n";
        }
        if (global_step >= total_steps) break;
    }
    (void)total_loss_clip; (void)total_loss_noclip;  // unused placeholder

    // ---------- 8) 总结 ----------
    std::cout << "\n=== summary ===\n";
    std::cout << "losses_with_clip_first = ";
    for (size_t i = 0; i < std::min<size_t>(6, train_losses.size()); ++i)
        std::cout << train_losses[i] << " ";
    std::cout << "\n";
    std::cout << "losses_with_clip_last  = ";
    for (size_t i = 0; i < train_losses.size(); ++i)
        std::cout << train_losses[i] << " ";
    std::cout << "\n";

    return 0;
}