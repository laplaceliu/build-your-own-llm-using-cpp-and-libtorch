// sft_train.cpp
// 指令微调训练 CLI：加载预训练 GPT-2 -> 指令数据微调 -> 保存 .pth
//
// 用法:
//   sft_train [选项]
//     --data <json>        指令数据集（默认 data/instruction-data.json）
//     --weights <st>       预训练权重 safetensors（默认 data/gpt2-medium.safetensors）
//     --size <small|medium|large|xl>  模型规模（默认 medium）
//     --epochs <n>         训练轮数（默认 2）
//     --batch <n>          batch size（默认 8）
//     --lr <f>             学习率（默认 5e-5）
//     --out <path>         输出模型路径（默认 data/gpt2-medium-sft.pth）
//     --no-cuda            强制 CPU
#include <torch/torch.h>

#include <iostream>
#include <string>

#include "gpt_sft_core.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "../chapters/chapter02_text_data/data"
#endif

namespace {

struct Options {
  std::string data = DATA_DIR "/instruction-data.json";
  std::string weights = DATA_DIR "/gpt2-medium.safetensors";
  std::string size = "medium";
  std::string out = DATA_DIR "/gpt2-medium-sft.pth";
  int64_t epochs = 2, batch = 8;
  double lr = 5e-5;
  bool use_cuda = true;
};

Options parse_args(int argc, char* argv[]) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("缺少参数: ") + name);
      return argv[++i];
    };
    if (a == "--data") o.data = next("--data");
    else if (a == "--weights") o.weights = next("--weights");
    else if (a == "--size") o.size = next("--size");
    else if (a == "--epochs") o.epochs = std::stoll(next("--epochs"));
    else if (a == "--batch") o.batch = std::stoll(next("--batch"));
    else if (a == "--lr") o.lr = std::stod(next("--lr"));
    else if (a == "--out") o.out = next("--out");
    else if (a == "--no-cuda") o.use_cuda = false;
    else throw std::runtime_error("未知参数: " + a);
  }
  return o;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    at::globalContext().setAllowTF32CuBLAS(false);
    auto opts = parse_args(argc, argv);

    auto cfg = gpt_sft::config_for_size(opts.size);
    torch::Device device = opts.use_cuda ? gpt_sft::select_device()
                                         : torch::Device(torch::kCPU);
    std::cout << "=== sft_train（指令微调训练）===\n"
              << "模型规模: " << opts.size << " (emb=" << cfg.emb_dim
              << ", layers=" << cfg.n_layers << ", heads=" << cfg.n_heads << ")\n"
              << "设备: " << device << "\n"
              << "数据: " << opts.data << "\n"
              << "预训练权重: " << opts.weights << "\n";

    // 加载预训练权重
    std::cout << "[加载] GPT-2 " << opts.size << " 预训练权重...\n";
    auto model = gpt_sft::load_pretrained(cfg, opts.weights);
    model->to(device);
    model->eval();
    std::cout << "[完成] 权重加载\n";

    // 数据
    auto data = gpt_sft::load_instruction_data(opts.data);
    std::vector<ch7::InstructionEntry> train, val, test;
    gpt_sft::split_dataset(data, train, val, test);
    std::cout << "[数据] 共 " << data.size() << " 条: train=" << train.size()
              << ", val=" << val.size() << ", test=" << test.size() << "\n";

    ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                TOKENIZER_DIR "/vocab.bpe");

    // 训练
    std::cout << "[训练] 指令微调 " << opts.epochs << " 轮...\n";
    gpt_sft::train_instruction(model, train, val, tokenizer,
                               /*num_epochs=*/opts.epochs, /*batch_size=*/opts.batch,
                               /*lr=*/opts.lr, /*eval_freq=*/5, /*eval_iter=*/5,
                               /*start_context=*/"", device);

    // 保存
    model->eval();
    torch::save(model, opts.out);
    std::cout << "[保存] 微调模型 -> " << opts.out << "\n";

    // 保存后快速自检：用训练集第一条生成回复
    if (!train.empty()) {
      auto reply = gpt_sft::generate_response(model, tokenizer, train[0].instruction,
                                              train[0].input, /*max_new_tokens=*/64, device);
      std::cout << "[自检] 示例回复:\n" << reply << "\n";
    }

    std::cout << "=== 训练完成 ===" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}
