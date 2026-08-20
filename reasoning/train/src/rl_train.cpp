// reasoning/train/src/rl_train.cpp
// GRPO 强化学习训练 CLI
// 用法: rl_train --data <rl.json> [--size small|medium] [--weights base.safetensors]
//                  [--init sft.pth]  (可选，从 SFT 检查点继续)
//                  [--max-steps N] [--batch N] [--group N] [--max-new N] [--lr 1e-5]
//                  [--w-fmt 0.5] [--w-acc 0.5] [--out rl.pth]
#include <gflags/gflags.h>
#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "training.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "reasoning_core.h"
#include "rl_train.h"

#define TOKENIZER_DIR_DEFAULT "/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/chapters/chapter02_text_data/data"

DEFINE_string(data, "", "RL 训练数据 JSON 路径（rl-train.json）");
DEFINE_string(size, "small", "模型规模: small|medium|...");
DEFINE_string(weights, "", "预训练基座 safetensors 路径");
DEFINE_string(init, "", "可选：从 .pth 加载（覆盖 weights）");
DEFINE_string(out, "data/rl.pth", "输出模型 .pth");
DEFINE_int64(max_steps, 60, "总训练步数");
DEFINE_int64(batch, 4, "每次 prompt 数");
DEFINE_int64(group, 4, "每 prompt 采样数 (G)");
DEFINE_int64(max_new, 384, "每条 rollout 最大长度");
DEFINE_double(lr, 1e-5, "学习率");
DEFINE_double(w_fmt, 0.5, "格式奖励权重");
DEFINE_double(w_acc, 0.5, "准确性奖励权重");
DEFINE_double(temperature, 0.9, "采样温度");
DEFINE_int64(top_k, 50, "top-k 采样");
DEFINE_int64(seed, 123, "随机种子");
DEFINE_bool(no_cuda, false, "强制使用 CPU");
DEFINE_string(tokenizer_dir, TOKENIZER_DIR_DEFAULT, "BPE tokenizer 数据目录");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_data.empty()) {
    std::cerr << "错误: 必须指定 --data\n";
    return 1;
  }
  torch::manual_seed(static_cast<uint32_t>(FLAGS_seed));
  std::string enc = FLAGS_tokenizer_dir + "/encoder.json";
  std::string bpe = FLAGS_tokenizer_dir + "/vocab.bpe";
  ch2::BpeTokenizer tokenizer(enc, bpe);

  reasoning::RLConfig cfg;
  cfg.data_path = FLAGS_data;
  cfg.size = FLAGS_size;
  cfg.weights_path = FLAGS_weights;
  cfg.init_path = FLAGS_init;
  cfg.out_path = FLAGS_out;
  cfg.max_steps = FLAGS_max_steps;
  cfg.batch_size = FLAGS_batch;
  cfg.group_size = FLAGS_group;
  cfg.max_new_tokens = FLAGS_max_new;
  cfg.lr = FLAGS_lr;
  cfg.w_fmt = FLAGS_w_fmt;
  cfg.w_acc = FLAGS_w_acc;
  cfg.temperature = FLAGS_temperature;
  cfg.top_k = FLAGS_top_k > 0 ? c10::optional<int64_t>(FLAGS_top_k) : c10::nullopt;
  cfg.seed = static_cast<unsigned>(FLAGS_seed);
  cfg.no_cuda = FLAGS_no_cuda;

  try {
    reasoning::train_rl(cfg, tokenizer);
  } catch (const std::exception& e) {
    std::cerr << "[rl_train] 训练失败: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
