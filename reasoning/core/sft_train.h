#pragma once
// reasoning/core/sft_train.h
//
// 推理 SFT 训练：基于 R1 风格 思考/答案 数据的指令微调。
// 复用 ch7::InstructionLoader。
#include <torch/torch.h>
#include "gpt.h"
#include "training.h"
#include "instruction.h"
#include "bpe_tokenizer.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "reasoning_core.h"
#include "gen_utils.h"

namespace reasoning {

struct SFTTrainConfig {
  std::string data_path;
  std::string size = "small";
  std::string weights_path;
  std::string out_path;
  int64_t batch_size = 8;
  int64_t max_length = 1024;
  int64_t epochs = 1;
  double lr = 5e-5;
  int64_t eval_freq = 50;
  int64_t eval_iter = 5;
  double train_ratio = 0.85;
  unsigned seed = 123;
  bool no_cuda = false;
  bool shuffle = true;
};

inline ch4::GPTModel load_model_for_sft(const SFTTrainConfig& cfg, const torch::Device& device) {
  auto gpt_cfg = config_for_size(cfg.size);
  if (!cfg.weights_path.empty()) {
    try {
      auto loaded = load_pretrained(gpt_cfg, cfg.weights_path);
      loaded->to(device);
      std::cout << "[sft_train] 已加载预训练权重: " << cfg.weights_path << "\n";
      return loaded;
    } catch (const std::exception& e) {
      std::cerr << "[警告] 加载预训练失败（" << e.what() << "），使用随机初始化\n";
    }
  }
  auto m = make_model(gpt_cfg);
  m->to(device);
  return m;
}

inline void train_sft(const SFTTrainConfig& cfg, ch2::BpeTokenizer& tokenizer) {
  auto device = cfg.no_cuda ? torch::Device(torch::kCPU)
                            : (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                           : torch::Device(torch::kCPU));
  std::cout << "[sft_train] 设备: " << device << "\n";

  ch4::GPTModel model = load_model_for_sft(cfg, device);

  auto items = load_jsonl_or_json(cfg.data_path);
  std::cout << "[sft_train] 加载样本: " << items.size() << "\n";
  if (items.empty()) throw std::runtime_error("无训练数据");

  std::vector<ch7::InstructionEntry> entries;
  entries.reserve(items.size());
  // 防御 std::regex 栈溢出：先按长度粗筛，避免超长样本进入 BPE tokenizer
  // cfg.max_length 是 token 数；英文字符≈token×1.3，预留 1 字符~1 token 的下界再加 buffer
  const size_t MAX_CHAR = static_cast<size_t>(cfg.max_length) * 4;  // 4× 上限，超长样本直接砍
  size_t truncated = 0;
  for (const auto& it : items) {
    ch7::InstructionEntry e;
    e.instruction = it.instruction;
    e.input = it.input;
    e.output = it.output;
    if (e.instruction.size() + e.input.size() + e.output.size() > MAX_CHAR) {
      // 优先截断最长的 output
      if (e.output.size() > MAX_CHAR / 2) {
        e.output.resize(MAX_CHAR / 2);
        e.output += "...[truncated]";
        ++truncated;
      } else {
        ++truncated;
        continue;  // 整条丢掉
      }
    }
    entries.push_back(std::move(e));
  }
  std::cout << "[sft_train] 过滤/截断: 保留 " << entries.size()
              << ", 丢弃/截断 " << truncated << "\n";
  if (entries.empty()) throw std::runtime_error("过滤后无训练数据");
  std::mt19937 rng(cfg.seed);
  std::shuffle(entries.begin(), entries.end(), rng);
  size_t n_train = static_cast<size_t>(entries.size() * cfg.train_ratio);
  std::vector<ch7::InstructionEntry> train_set(entries.begin(), entries.begin() + n_train);
  std::vector<ch7::InstructionEntry> val_set(entries.begin() + n_train, entries.end());

  ch7::InstructionDataset train_ds(train_set, tokenizer);
  ch7::InstructionLoader train_loader(train_ds, cfg.batch_size, cfg.shuffle,
                                       true, 50256, cfg.max_length, device);
  std::cout << "[sft_train] 训练批次数: " << train_loader.num_batches()
            << ", 验证集样本: " << val_set.size() << "\n";

  torch::optim::AdamWOptions options(cfg.lr);
  options.weight_decay(0.01);
  torch::optim::AdamW optimizer(model->parameters(), options);

  int64_t tokens_seen = 0;
  int64_t global_step = -1;
  auto t_start = std::chrono::steady_clock::now();

  for (int64_t epoch = 0; epoch < cfg.epochs; ++epoch) {
    model->train();
    for (const auto& batch : train_loader.batches()) {
      optimizer.zero_grad();
      auto loss = ch5::calc_loss_batch(batch.inputs, batch.targets, model);
      loss.backward();
      optimizer.step();
      tokens_seen += batch.inputs.numel();
      ++global_step;

      if (global_step % cfg.eval_freq == 0) {
        double train_loss = loss.item<double>();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - t_start).count();
        std::cout << "[sft_train] Ep " << (epoch + 1) << " Step " << global_step
                  << ": loss=" << train_loss << " tokens=" << tokens_seen
                  << " elapsed=" << elapsed << "s\n";
      }
    }
    if (!val_set.empty()) {
      double val_loss = 0.0;
      int64_t n = 0;
      torch::NoGradGuard no_grad;
      model->eval();
      ch7::InstructionDataset vds(val_set, tokenizer);
      ch7::InstructionLoader vloader(vds, std::min<int64_t>(2, (int64_t)val_set.size()),
                                       false, true, 50256, cfg.max_length, device);
      for (const auto& b : vloader.batches()) {
        val_loss += ch5::calc_loss_batch(b.inputs, b.targets, model).item<double>();
        ++n;
      }
      std::cout << "[sft_train] Epoch " << epoch + 1
                << " 完成 (val_loss=" << (n ? val_loss / n : 0.0) << ")\n";
      model->train();
    } else {
      std::cout << "[sft_train] Epoch " << epoch + 1 << " 完成\n";
    }
  }

  torch::save(model, cfg.out_path);
  std::cout << "[sft_train] 已保存模型: " << cfg.out_path << "\n";
}

}  // namespace reasoning