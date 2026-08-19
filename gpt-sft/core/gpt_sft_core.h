// gpt_sft_core.h
// 独立项目统一 API：GPT 模型工厂、预训练权重加载、指令微调、文本生成。
//
// 供 train/（sft_train）与 serve/（sft_serve, sft_chat）共用。
#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "instruction.h"
#include "safetensors.h"

namespace gpt_sft {

// ==========================================================================
// 模型配置（GPT-2 各规模；默认 medium，与第 7 章一致）
// ==========================================================================
struct ModelConfig {
  int64_t vocab_size = 50257;
  int64_t context_length = 1024;
  int64_t emb_dim = 1024;    // medium: 1024 / small: 768 / large: 1280
  int64_t n_heads = 16;      // medium: 16 / small: 12 / large: 20
  int64_t n_layers = 24;     // medium: 24 / small: 12 / large: 36
  double drop_rate = 0.0;
  bool qkv_bias = true;
};

// 由规模名构造配置：small / medium / large / xl
ModelConfig config_for_size(const std::string& size);

// ==========================================================================
// 模型工厂
// ==========================================================================

// 构造 GPT 模型（随机初始化）
ch4::GPTModel make_model(const ModelConfig& cfg);

// 加载 HF safetensors 预训练权重到模型（GPT-2 small/medium 通用）
// 返回模型（已是 cfg 配置），失败抛异常
ch4::GPTModel load_pretrained(const ModelConfig& cfg, const std::string& safetensors_path);

// 设备选择：优先 CUDA
torch::Device select_device();

// ==========================================================================
// 指令微调训练（train/sft_train 使用）
// ==========================================================================

// 指令数据加载（JSON 数组: instruction/input/output）
std::vector<ch7::InstructionEntry> load_instruction_data(const std::string& path);

// 划分训练/验证/测试（85/10/5）
void split_dataset(const std::vector<ch7::InstructionEntry>& data,
                   std::vector<ch7::InstructionEntry>& train,
                   std::vector<ch7::InstructionEntry>& val,
                   std::vector<ch7::InstructionEntry>& test);

// 指令微调（ignore_index=-100 掩码填充），返回最终训练损失
double train_instruction(ch4::GPTModel& model,
                         const std::vector<ch7::InstructionEntry>& train_data,
                         const std::vector<ch7::InstructionEntry>& val_data,
                         ch2::BpeTokenizer& tokenizer,
                         int64_t num_epochs, int64_t batch_size, double lr,
                         int64_t eval_freq, int64_t eval_iter,
                         const std::string& start_context,
                         const torch::Device& device);

// ==========================================================================
// 推理（serve/sft_chat 使用）
// ==========================================================================

// 指令回复生成（贪心解码 + eos 停止，返回回复文本）
// instruction/input 组装为 Alpaca 模板；input 为空时省略 ### Input:
std::string generate_response(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                              const std::string& instruction,
                              const std::string& input,
                              int64_t max_new_tokens,
                              const torch::Device& device);

}  // namespace gpt_sft
