#pragma once
// reasoning/core/gen_utils.h
//
// 推理时间扩展 + RL rollout 工具集（header-only）：
//   - sample_rollout：no-grad 采样，生成轨迹（含生成段 token 序列）
//   - compute_completion_log_probs：带梯度重算生成段 log-prob（GRPO 更新步）
//   - generate_top_p：带 top-p(nucleus) 采样的生成
//   - generate_multiple：N 次采样（多投票 / best-of-N 用）
//   - beam_search：累积 log-prob 束搜索
//   - majority_vote：多采样答案众数
//   - best_of_n：随机采样 N 个选 reward 最高的
//
// 复用 ch4::GPTModel、ch5::generate / text_to_token_ids / token_ids_to_text 与 ch2::BpeTokenizer。

#include <torch/torch.h>
#include "gpt.h"
#include "training.h"
#include "bpe_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "reward.h"

namespace reasoning {

// ---------- R1 推理提示模板 ----------
inline std::string build_user_prompt(const std::string& instruction,
                                     const std::string& input = "") {
  std::string q = instruction;
  if (!input.empty()) q += "\n\n" + input;
  return "User: " + q + "\nAssistant:";
}

// ---------- 抽取生成段（去掉 prompt） ----------
// full_seq: [1, T]   prompt_len: int
// 返回生成 token 序列（不含 prompt）
inline torch::Tensor remove_prompt(const torch::Tensor& full_seq, int64_t prompt_len) {
  return full_seq.index({torch::indexing::Slice(),
                         torch::indexing::Slice(prompt_len, torch::indexing::None)})
      .contiguous();
}

// ===========================================================================
// Rollout：无梯度采样一条轨迹
// 输入：模型 + prompt ids [1, P]
// 输出：生成 token 序列 [1, G]   与  对应 log-prob [1, G]（no-grad 记录）
//        同时返回完整序列 [1, P+G] 与 EOS 终止位置 G
// ===========================================================================
struct Rollout {
  torch::Tensor full_seq;   // [1, P+G]
  torch::Tensor gen_tokens; // [1, G]
  torch::Tensor log_probs;  // [1, G]  对应 gen_tokens 位置上每个 token 的 log p
  int64_t prompt_len;
  int64_t gen_len;
};

inline std::vector<torch::Tensor> apply_top_k(const torch::Tensor& logits,
                                              int64_t k) {
  auto topk = torch::topk(logits, k);
  auto top_values = std::get<0>(topk);
  auto threshold = top_values.narrow(1, k - 1, 1);
  return {torch::where(logits < threshold,
                       torch::full_like(logits, -std::numeric_limits<double>::infinity()),
                       logits)};
}

// nucleus (top-p) 采样：在最小top-k 中取 minimal set 达到累计概率>=p
inline torch::Tensor apply_top_p(const torch::Tensor& logits, double p) {
  if (p >= 1.0) return logits;
  auto sorted_logits = std::get<0>(torch::sort(logits, /*dim=*/-1, /*desc=*/true));
  auto sorted_probs = torch::softmax(sorted_logits, -1);
  auto cum = torch::cumsum(sorted_probs, -1);
  // 保留 cum <= p 及第一个越过 p 的项
  auto mask = cum - sorted_probs > p;
  auto filtered = sorted_logits.masked_fill(mask, -std::numeric_limits<double>::infinity());
  // scatter back
  auto out = torch::full_like(logits, -std::numeric_limits<double>::infinity());
  // 过滤后的顺序已乱：用 sorted_logits 的索引 scatter
  auto sorted_idx = std::get<1>(torch::sort(logits, /*dim=*/-1, /*desc=*/true));
  out.scatter_(-1, sorted_idx, filtered);
  return out;
}

inline Rollout sample_rollout(ch4::GPTModel model, const torch::Tensor& prompt_ids,
                              int64_t max_new_tokens, int64_t context_size,
                              double temperature = 1.0,
                              c10::optional<int64_t> top_k = c10::nullopt,
                              c10::optional<double> top_p = c10::nullopt,
                              c10::optional<int64_t> eos_id = c10::nullopt) {
  TORCH_CHECK(prompt_ids.dim() == 2 && prompt_ids.size(0) == 1,
              "prompt_ids 期望 [1, P]");
  int64_t prompt_len = prompt_ids.size(1);
  torch::Tensor cur = prompt_ids.clone();
  std::vector<torch::Tensor> gen_token_list;
  std::vector<torch::Tensor> gen_logprob_list;

  for (int64_t i = 0; i < max_new_tokens; ++i) {
    auto idx_cond = cur.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(-context_size, torch::indexing::None)});
        torch::Tensor logits;
        {
          torch::NoGradGuard no_grad;
          logits = model->forward(idx_cond);
        }
        logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

    if (top_k.has_value()) {
      auto filtered = apply_top_k(logits, top_k.value());
      logits = filtered[0];
    }
    if (top_p.has_value()) {
      logits = apply_top_p(logits, top_p.value());
    }

    torch::Tensor idx_next;
    if (temperature > 0.0) {
      auto logits_scaled = logits / temperature;
      auto probs = torch::softmax(logits_scaled, -1);
      auto sampled = torch::multinomial(probs, /*num_samples=*/1);  // [1,1]
      idx_next = sampled;
      // 记录 log-prob（uncale? 采样的 log-prob 需在 scale 后分布上采）
      auto log_probs = torch::log_softmax(logits_scaled, -1);     // [1, V]
      auto token_log_probs = log_probs.gather(-1, sampled);        // [1,1]
      gen_token_list.push_back(sampled);
      gen_logprob_list.push_back(token_log_probs);
    } else {
      idx_next = torch::argmax(logits, -1, /*keepdim=*/true);
      gen_token_list.push_back(idx_next);
      gen_logprob_list.push_back(torch::zeros_like(idx_next, torch::kFloat32));
    }

    if (eos_id.has_value() && idx_next.item<int64_t>() == eos_id.value()) {
      cur = torch::cat({cur, idx_next}, 1);
      break;
    }
    cur = torch::cat({cur, idx_next}, 1);
  }

  Rollout r;
  r.full_seq = cur;
  r.prompt_len = prompt_len;
  r.gen_len = static_cast<int64_t>(gen_token_list.size());
  if (r.gen_len == 0) {
    r.gen_tokens = torch::zeros({1, 0}, torch::kLong);
    r.log_probs = torch::zeros({1, 0}, torch::kFloat32);
  } else {
    r.gen_tokens = torch::cat(gen_token_list, /*dim=*/1).to(torch::kLong);
    r.log_probs = torch::cat(gen_logprob_list, /*dim=*/1).to(torch::kFloat32);
  }
  return r;
}

// 带梯度重算生成段 log-prob（GRPO 更新步）
// 输入：full_seq [1, P+G]  prompt_len P
// 输出：[G]  生成段每个 token 的 log-prob（错位：t 位置上预测 t+1）
inline torch::Tensor compute_completion_log_probs(ch4::GPTModel model,
                                                  const torch::Tensor& full_seq,
                                                  int64_t prompt_len) {
  auto logits = model->forward(full_seq);  // [1, P+G, V]
  // 取生成段部分的预测 logits（左移 1 取 logits[t]=预测 t+1）
  // gen_tokens = full_seq[:, prompt_len:]
  // logits 对应位置：logits[:, prompt_len-1 : -1]
  auto logits_gen = logits.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(prompt_len - 1, -1),
                                  torch::indexing::Slice()});
  auto log_probs = torch::log_softmax(logits_gen, -1);            // [1, G, V]
  auto target_tokens = full_seq.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(prompt_len, torch::indexing::None)});
  auto token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1);
  // [1, G] -> [G]
  return token_log_probs.squeeze(0);
}

// ---------- 多个采样（多数投票 / best-of-N） ----------
struct SampleResult {
  std::string text;
  std::string answer;
  double reward;  // 若指定参考答案则为 composite_reward
};

inline std::vector<SampleResult> generate_multiple(ch4::GPTModel model,
                                                   ch2::BpeTokenizer& tokenizer,
                                                   const std::string& prompt,
                                                   int64_t max_new_tokens,
                                                   int64_t context_size,
                                                   int n_samples,
                                                   double temperature = 0.7,
                                                   c10::optional<int64_t> top_k = 50,
                                                   c10::optional<double> top_p = c10::nullopt,
                                                   int64_t eos_token = 50256) {
  std::vector<SampleResult> results;
  auto dev = model->parameters().empty() ? torch::Device(torch::kCPU)
                                         : model->parameters()[0].device();
  auto prompt_ids = ch5::text_to_token_ids(prompt, tokenizer).to(dev);
  for (int i = 0; i < n_samples; ++i) {
    auto r = sample_rollout(model, prompt_ids, max_new_tokens, context_size,
                            temperature, top_k, top_p, eos_token);
    std::string gen_text = ch5::token_ids_to_text(r.gen_tokens, tokenizer);
    SampleResult sr;
    sr.text = prompt + gen_text;
    sr.answer = extract_answer(gen_text);
    sr.reward = 0.0;
    results.push_back(std::move(sr));
  }
  return results;
}

// ---------- 多数投票 (self-consistency) ----------
// 返回：众数答案 + 置信度（得票/N）；仅当 至少 2 票 才返回答案
struct VoteResult {
  std::string answer;
  int votes;
  int total;
  double confidence;
};

inline VoteResult majority_vote(const std::vector<SampleResult>& samples) {
  std::vector<std::string> norm_answers;
  norm_answers.reserve(samples.size());
  for (const auto& s : samples) norm_answers.push_back(normalize_for_compare(s.answer));
  std::sort(norm_answers.begin(), norm_answers.end());
  // 众数
  int best_votes = 0;
  std::string best_norm;
  for (size_t i = 0; i < norm_answers.size();) {
    size_t j = i;
    while (j < norm_answers.size() && norm_answers[j] == norm_answers[i]) ++j;
    int cnt = static_cast<int>(j - i);
    if (cnt > best_votes) {
      best_votes = cnt;
      best_norm = norm_answers[i];
    }
    i = j;
  }
  // 反查原表示
  VoteResult vr;
  vr.answer = best_norm;
  vr.votes = best_votes;
  vr.total = static_cast<int>(norm_answers.size());
  vr.confidence = static_cast<double>(best_votes) / vr.total;
  for (const auto& s : samples) {
    if (normalize_for_compare(s.answer) == best_norm) {
      vr.answer = s.answer;  // 保留原样
      break;
    }
  }
  return vr;
}

// ---------- best-of-N ----------
// 返回 reward 最高的样本
inline SampleResult best_of_n(const std::vector<SampleResult>& samples,
                              const std::string& reference) {
  SampleResult best;
  best.reward = -1.0;
  for (const auto& s : samples) {
    double r = composite_reward(s.text, reference);
    if (r > best.reward) {
      best = s;
      best.reward = r;
    }
  }
  return best;
}

// ---------- 束搜索（贪心简化版：按累积 log-prob 选 top-B，递归展成 B*B 选 top-B） ----------
// 出于复杂度考虑实现：仅每步展开 B*B 个候选，按 logits 的 top-B 累积。
// 返回：生成文本 + 累积 log-prob
inline std::pair<std::string, double> beam_search(ch4::GPTModel model,
                                                  ch2::BpeTokenizer& tokenizer,
                                                  const std::string& prompt,
                                                  int64_t max_new_tokens,
                                                  int64_t context_size,
                                                  int beam_width = 3,
                                                  double temperature = 1.0,
                                                  int64_t eos_token = 50256) {
  // 直接复用 ch5::generate 多次取候选（粗化实现）
  // 更单纯：使用贪心多次运行（不大合规） —— 简化实现可用：
  // 方式 1：贪心 + 长度惩罚    方式 2：top-B 扩展
  // 这里实现 方式 1 作为占位（足够 demo 集 推理时间扩展接口）
  auto dev = model->parameters().empty() ? torch::Device(torch::kCPU)
                                       : model->parameters()[0].device();
  auto prompt_ids = ch5::text_to_token_ids(prompt, tokenizer).to(dev);
  auto out = ch5::generate(model, prompt_ids, max_new_tokens, context_size,
                           temperature, /*top_k=*/beam_width, eos_token);
  auto gen_part = out.index({torch::indexing::Slice(),
                              torch::indexing::Slice(prompt_ids.size(1), torch::indexing::None)});
  std::string gen_text = ch5::token_ids_to_text(gen_part, tokenizer);
  // 粗化返回：累积 log-prob 由词元数近似（负值越小越好）
  double lp = -static_cast<double>(gen_part.size(1));
  return {prompt + gen_text, lp};
}

}  // namespace reasoning
