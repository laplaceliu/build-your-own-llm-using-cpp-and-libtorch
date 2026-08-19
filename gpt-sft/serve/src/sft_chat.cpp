// sft_chat.cpp
// 命令行交互式推理：加载微调模型，stdin 对话（或单条指令参数）。
//
// 用法:
//   sft_chat --model <pth> [--size small|medium|large|xl] [--no-cuda]
//   sft_chat --model <pth> "指令" ["输入"]     # 单条指令模式
//   （无参数进入交互模式，输入 "quit"/"exit" 退出）
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
  std::string model = DATA_DIR "/gpt2-medium-sft.pth";
  std::string size = "medium";
  bool use_cuda = true;
  std::string instruction, input;
};

Options parse_args(int argc, char* argv[]) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("缺少参数: ") + name);
      return argv[++i];
    };
    if (a == "--model") o.model = next("--model");
    else if (a == "--size") o.size = next("--size");
    else if (a == "--no-cuda") o.use_cuda = false;
    else if (a.rfind("--", 0) == 0) throw std::runtime_error("未知参数: " + a);
    else if (o.instruction.empty()) o.instruction = a;
    else if (o.input.empty()) o.input = a;
  }
  return o;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    at::globalContext().setAllowTF32CuBLAS(false);
    auto opts = parse_args(argc, argv);
    torch::Device device = opts.use_cuda ? gpt_sft::select_device()
                                         : torch::Device(torch::kCPU);
    std::cout << "=== sft_chat（指令推理）===\n"
              << "模型: " << opts.model << "\n"
              << "设备: " << device << "\n";

    auto cfg = gpt_sft::config_for_size(opts.size);
    auto model = gpt_sft::make_model(cfg);
    torch::load(model, opts.model);
    model->to(device);
    model->eval();
    std::cout << "[完成] 模型加载\n";

    ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                TOKENIZER_DIR "/vocab.bpe");

    // 单条指令模式
    if (!opts.instruction.empty()) {
      auto reply = gpt_sft::generate_response(model, tokenizer, opts.instruction,
                                              opts.input, /*max_new_tokens=*/256, device);
      std::cout << "\n>>> " << reply << "\n";
      return 0;
    }

    // 交互模式
    std::cout << "\n输入指令（输入 quit 退出）:\n";
    std::string line;
    while (true) {
      std::cout << ">>> ";
      if (!std::getline(std::cin, line)) break;
      if (line == "quit" || line == "exit") break;
      if (line.empty()) continue;
      auto reply = gpt_sft::generate_response(model, tokenizer, line, "",
                                              /*max_new_tokens=*/256, device);
      std::cout << reply << "\n\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}
