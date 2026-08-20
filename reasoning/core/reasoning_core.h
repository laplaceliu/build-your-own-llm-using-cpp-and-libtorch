#pragma once
// reasoning/core/reasoning_core.h
//
// 推理项目对 gpt-sft::core 的统一封装 + 推理特有的配置。
// 训练 / RL / 推理 / 评测 都通过此头引入。
//
// 主要 API：
//   - config_for_size(const std::string& size)            // 复用 gpt_sft::config_for_size
//   - make_model(const std::string& size, const GPTConfig& cfg)  // 复用 gpt_sft::make_model
//   - load_pretrained(ch4::GPTModel&, const std::string&, const GPTConfig&)
//   - select_device(const std::string& cuda_tag = "auto")  // 复用 gpt_sft::select_device
//   - load_reasoning_data(json_path)         // {"instruction","input","output"}
//   - load_rl_data(json_path)               // {"instruction","input","output","gold"}
//   - make_r1_prompt(instruction, input)    // 思想与答案 标签训练的提示模板
//   - make_r1_full_text(instruction, input, output)  // 训练样本完整文本（带 思考/答案 标签）
//   - make_eval_prompt(instruction, input)   // 评测提示
//   - ReasoningItem / ReasoningDataset / ReasoningLoader

#include <torch/torch.h>
#include "gpt.h"
#include "bpe_tokenizer.h"
#include "instruction.h"
#include <nlohmann/json.hpp>

#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "gpt_sft_core.h"  // 复用 gpt-sft 的工厂/加载 API

namespace reasoning {

// ============ 推理数据格式 ============
struct ReasoningItem {
  std::string instruction;
  std::string input;
  std::string output;      // "think ... think\n<answer>...</answer>"
  std::string gold;        // GSM8K 金标准答案（仅 eval/rl 用）
};

// ============ 加载数据 ============
inline std::vector<ReasoningItem> load_jsonl_or_json(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("无法打开数据文件: " + path);
  }
  std::vector<ReasoningItem> items;
  // 优先尝试 JSON 数组
  try {
    nlohmann::json j;
    in >> j;
    if (j.is_array()) {
      for (const auto& it : j) {
        ReasoningItem ri;
        ri.instruction = it.value("instruction", std::string{});
        ri.input = it.value("input", std::string{});
        ri.output = it.value("output", std::string{});
        ri.gold = it.value("gold", std::string{});
        if (!ri.instruction.empty()) items.push_back(std::move(ri));
      }
      return items;
    }
  } catch (...) {
    // 退到 JSONL
  }
  // JSONL 兜底
  std::ifstream in2(path);
  std::string line;
  while (std::getline(in2, line)) {
    if (line.empty()) continue;
    try {
      auto j = nlohmann::json::parse(line);
      ReasoningItem ri;
      ri.instruction = j.value("instruction", std::string{});
      ri.input = j.value("input", std::string{});
      ri.output = j.value("output", std::string{});
      ri.gold = j.value("gold", std::string{});
      if (!ri.instruction.empty()) items.push_back(std::move(ri));
    } catch (...) {
      // 忽略坏行
    }
  }
  return items;
}

// ============ Alpaca 风格提示模板（与训练 InstructionDataset::format_input 完全对齐） ============
// 训练时 ch7::InstructionDataset 使用:
//   "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Input:\n{input}\n\n### Response:\n{output}"
// 推理时必须复刻"### Response:"之前的全部上下文，模型才会继续接 reply 段。
inline std::string make_r1_prompt(const std::string& instruction,
                                  const std::string& input = "") {
  std::string instr_text =
      "Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request."
      "\n\n### Instruction:\n" + instruction;
  std::string input_text = input.empty() ? "" : ("\n\n### Input:\n" + input);
  return instr_text + input_text + "\n\n### Response:\n";
}

// 训练样本完整文本：answer 段含 思考/答案 标签
inline std::string make_r1_full_text(const std::string& instruction,
                                     const std::string& input,
                                     const std::string& output) {
  if (output.find("think") != std::string::npos ||
      output.find("<answer>") != std::string::npos) {
    return make_r1_prompt(instruction, input) + " " + output;
  }
  return make_r1_prompt(instruction, input) + " " + output;
}

// 评测提示：让模型以 思考/答案 风格产出
inline std::string make_eval_prompt(const std::string& instruction,
                                    const std::string& input = "") {
  std::string q = instruction;
  if (!input.empty()) q += "\n\n" + input;
  return "User: " + q + "\nA: " +
         "Let's think step by step. think think\n<answer>";
}

// ============ 简易批处理加载器（与 ch7::InstructionLoader 思路一致） ============
// 因 answer 起始位置可能位于 prompt 之后，使用 ch7::InstructionLoader 拼装即可：
//   - format_input -> inputs
//   - format_full_text -> targets（输入 ID 中的对齐：targets = 输入；只在 answer 段计损失）
// 训练样本经过 tokenize + 补 pad 后拼接；为避免重写 collate_fn，我们采用以下策略：
//   - 用 ch2::BpeTokenizer 手动 encode 完整文本
//   - 用 make_r1_prompt 的 token 长度当作 prompt 长度
//   - target = 输入 ID 的副本；pad 处 token 替换为 -100

// ReasoningDataset: not used (ch7::InstructionLoader is sufficient).
// Keeping a thin wrapper would require full torch::data::Dataset<>::get impl.

// ============ 便利别名 ============
using gpt_sft::config_for_size;
using gpt_sft::make_model;
using gpt_sft::load_pretrained;
using gpt_sft::select_device;
using ModelConfig = gpt_sft::ModelConfig;

inline std::vector<ReasoningItem> split_items(const std::vector<ReasoningItem>& all,
                                              double train_ratio = 0.85,
                                              double val_ratio = 0.0,
                                              unsigned seed = 0) {
  std::vector<ReasoningItem> a = all;
  std::mt19937 rng(seed);
  std::shuffle(a.begin(), a.end(), rng);
  size_t n_train = static_cast<size_t>(a.size() * train_ratio);
  return std::vector<ReasoningItem>(a.begin(), a.begin() + n_train);
}

}  // namespace reasoning
