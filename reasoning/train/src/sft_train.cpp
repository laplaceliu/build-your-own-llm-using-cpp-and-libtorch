// reasoning/train/src/sft_train.cpp
// 推理 SFT 训练 CLI
// 用法: sft_train --data <json> [--size small|medium] [--weights <safetensors>]
//                  [--epochs N] [--lr 5e-5] [--batch N] [--max-length N]
//                  [--out checkpoint.pth] [--no-cuda] [--seed 123]
#include <gflags/gflags.h>
#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "training.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "reasoning_core.h"
#include "sft_train.h"

#define TOKENIZER_DIR_DEFAULT "/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/chapters/chapter02_text_data/data"

DEFINE_string(data, "", "训练数据 JSON 路径（sft-train.json）");
DEFINE_string(size, "small", "模型规模: small|medium|large|xl");
DEFINE_string(weights, "", "预训练基座 safetensors 路径（可选）");
DEFINE_string(out, "data/sft.pth", "输出模型 .pth");
DEFINE_int64(epochs, 1, "训练轮数");
DEFINE_int64(batch, 4, "批大小");
DEFINE_int64(max_length, 1024, "序列最大长度");
DEFINE_double(lr, 5e-5, "学习率");
DEFINE_int64(eval_freq, 50, "评估频率（全局步）");
DEFINE_int64(eval_iter, 5, "评估迭代数");
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

  // 加载 tokenizer
  std::string enc = FLAGS_tokenizer_dir + "/encoder.json";
  std::string bpe = FLAGS_tokenizer_dir + "/vocab.bpe";
  ch2::BpeTokenizer tokenizer(enc, bpe);
  std::cout << "[sft_train] tokenizer vocab_size=" << tokenizer.vocab_size() << "\n";

  reasoning::SFTTrainConfig cfg;
  cfg.data_path = FLAGS_data;
  cfg.size = FLAGS_size;
  cfg.weights_path = FLAGS_weights;
  cfg.out_path = FLAGS_out;
  cfg.epochs = FLAGS_epochs;
  cfg.batch_size = FLAGS_batch;
  cfg.max_length = FLAGS_max_length;
  cfg.lr = FLAGS_lr;
  cfg.eval_freq = FLAGS_eval_freq;
  cfg.eval_iter = FLAGS_eval_iter;
  cfg.seed = static_cast<unsigned>(FLAGS_seed);
  cfg.no_cuda = FLAGS_no_cuda;

  try {
    reasoning::train_sft(cfg, tokenizer);
  } catch (const std::exception& e) {
    std::cerr << "[sft_train] 训练失败: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
