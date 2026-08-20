#pragma once
// reasoning/core/rl_train.h
//
// 纯强化学习 (GRPO) 训练：DeepSeek-R1-Zero / TinyZero 路线
//
// 算法：
//   for step in 1..max_steps:
//     for each prompt p in batch:
//       对 p 采样 G 条 rollout（no-grad）
//     advantage = (reward - mean_group) / (std_group + eps)
//     重新 forward 整个 full_seq（带梯度），取生成段 log-prob
//     loss = -mean(advantage * log_prob)
//     loss.backward() -> AdamW.step()
#include <torch/torch.h>
#include "gpt.h"
#include "training.h"
#include "bpe_tokenizer.h"

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "reasoning_core.h"
#include "gen_utils.h"
#include "reward.h"

namespace reasoning {

struct RLConfig {
  std::string data_path;
  std::string size = "small";
  std::string weights_path;
  std::string init_path;
  std::string out_path;

  int64_t max_steps = 60;
  int64_t batch_size = 4;
  int64_t group_size = 4;
  int64_t max_new_tokens = 384;
  int64_t context_size = 1024;
  double lr = 1e-5;
  double temperature = 0.9;
  c10::optional<int64_t> top_k = 50;
  c10::optional<double> top_p = c10::nullopt;
  int64_t eos_token = 50256;
  double w_fmt = 0.5;
  double w_acc = 0.5;
  int64_t log_every = 1;
  int64_t sample_every = 5;
  unsigned seed = 123;
  bool no_cuda = false;
};

// 单 GRPO 步
inline double grpo_step(ch4::GPTModel model, ch2::BpeTokenizer& tokenizer,
                        const RLConfig& cfg, const std::vector<ReasoningItem>& batch,
                        torch::optim::Optimizer& optimizer,
                        const torch::Device& device, int64_t step_idx) {
  std::vector<Rollout> rollouts;
  std::vector<std::string> gen_texts;
  rollouts.reserve(batch.size() * cfg.group_size);
  gen_texts.reserve(batch.size() * cfg.group_size);

  for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); ++i) {
    const auto& item = batch[i];
    std::string prompt = make_r1_prompt(item.instruction, item.input);
    auto prompt_ids = ch5::text_to_token_ids(prompt, tokenizer).to(device);
    for (int64_t g = 0; g < cfg.group_size; ++g) {
      auto r = sample_rollout(model, prompt_ids, cfg.max_new_tokens, cfg.context_size,
                              cfg.temperature, cfg.top_k, cfg.top_p, cfg.eos_token);
      rollouts.push_back(r);
      std::string gen_text = ch5::token_ids_to_text(r.gen_tokens.to(torch::kCPU), tokenizer);
      gen_texts.push_back(prompt + gen_text);
    }
  }

  std::vector<double> rewards;
  std::vector<double> advantages;
  rewards.reserve(rollouts.size());
  advantages.reserve(rollouts.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    std::vector<double> group_r;
    for (int64_t g = 0; g < cfg.group_size; ++g) {
      size_t idx = i * cfg.group_size + g;
      double r = composite_reward(gen_texts[idx], batch[i].gold, cfg.w_fmt, cfg.w_acc);
      group_r.push_back(r);
    }
    double mean = std::accumulate(group_r.begin(), group_r.end(), 0.0) / group_r.size();
    double var = 0.0;
    for (double v : group_r) var += (v - mean) * (v - mean);
    var /= group_r.size();
    double std = std::sqrt(var + 1e-8);
    for (size_t g = 0; g < group_r.size(); ++g) {
      rewards.push_back(group_r[g]);
      advantages.push_back((group_r[g] - mean) / std);
    }
  }

  optimizer.zero_grad();
  torch::Tensor total_loss;
  int64_t loss_count = 0;
  double mean_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size();
  for (size_t i = 0; i < rollouts.size(); ++i) {
    auto& r = rollouts[i];
    if (r.gen_len == 0) continue;
    auto full_seq = r.full_seq.to(device);
    auto lp = compute_completion_log_probs(model, full_seq, r.prompt_len);
    double adv = advantages[i];
    auto adv_t = torch::tensor(adv, torch::kFloat32).to(device).expand_as(lp);
    auto term = -lp * adv_t;
    auto loss_t = term.mean();
    if (total_loss.defined()) {
      total_loss = total_loss + loss_t;
    } else {
      total_loss = loss_t;
    }
    ++loss_count;
  }

  if (loss_count == 0) {
    std::cout << "[rl_train] step=" << step_idx << " 所有 rollout gen_len=0，跳过\n";
    return mean_reward;
  }

  total_loss = total_loss / static_cast<double>(loss_count);
  total_loss.backward();
  torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
  optimizer.step();

  if (step_idx % cfg.log_every == 0) {
    std::cout << "[rl_train] step=" << step_idx
              << " mean_reward=" << mean_reward
              << " loss=" << total_loss.item<double>() << "\n";
  }
  if (step_idx % cfg.sample_every == 0 && !gen_texts.empty()) {
    auto best_it = std::max_element(rewards.begin(), rewards.end());
    size_t bi = std::distance(rewards.begin(), best_it);
    std::string sample_text = gen_texts[bi];
    std::cout << "  [best sample @ step " << step_idx << "]\n";
    std::string one_line;
    for (char c : sample_text) one_line += (c == '\n') ? ' ' : c;
    if (one_line.size() > 400) one_line = one_line.substr(0, 400) + "...";
    std::cout << "  " << one_line << "\n";
  }
  return mean_reward;
}

inline void train_rl(const RLConfig& cfg, ch2::BpeTokenizer& tokenizer) {
  auto device = cfg.no_cuda ? torch::Device(torch::kCPU)
                            : (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                           : torch::Device(torch::kCPU));
  std::cout << "[rl_train] 设备: " << device << "\n";

  auto gpt_cfg = config_for_size(cfg.size);
  ch4::GPTModel model = make_model(gpt_cfg);
  model->to(device);

  if (!cfg.init_path.empty()) {
    torch::load(model, cfg.init_path);
    std::cout << "[rl_train] loaded init model: " << cfg.init_path << "\n";
  } else if (!cfg.weights_path.empty()) {
    try {
      model = load_pretrained(gpt_cfg, cfg.weights_path);
      model->to(device);
      std::cout << "[rl_train] loaded pretrained: " << cfg.weights_path << "\n";
    } catch (const std::exception& e) {
      std::cerr << "[警告] 加载预训练失败: " << e.what() << "\n";
    }
  }

  auto items = load_jsonl_or_json(cfg.data_path);
  std::cout << "[rl_train] 训练样本: " << items.size() << "\n";
  if (items.empty()) throw std::runtime_error("无 RL 训练数据");

  torch::optim::AdamWOptions options(cfg.lr);
  options.weight_decay(0.0);
  torch::optim::AdamW optimizer(model->parameters(), options);

  std::mt19937 rng(cfg.seed);
  std::shuffle(items.begin(), items.end(), rng);
  auto t_start = std::chrono::steady_clock::now();
  for (int64_t step = 0; step < cfg.max_steps; ++step) {
    std::vector<ReasoningItem> batch;
    for (int64_t b = 0; b < cfg.batch_size; ++b) {
      size_t idx = rng() % items.size();
      batch.push_back(items[idx]);
    }
    grpo_step(model, tokenizer, cfg, batch, optimizer, device, step);
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - t_start).count();
  std::cout << "[rl_train] 训练完成, 总耗时=" << elapsed << "s\n";

  torch::save(model, cfg.out_path);
  std::cout << "[rl_train] 已保存: " << cfg.out_path << "\n";
}

}  // namespace reasoning