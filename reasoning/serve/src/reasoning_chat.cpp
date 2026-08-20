// reasoning/serve/src/reasoning_chat.cpp
// 推理模型 CLI 对话
//   - 加载 .pth 模型（含 SFT/RL/SFT+RL 任意检查点）
//   - 支持推理时间扩展策略：greedy / vote / beam / best-of
//   - 展示 思考 与 答案 段
//
// 用法:
//   reasoning_chat --model <pth> [--size small|medium] [--strategy greedy|vote|beam|best-of]
//                            [--samples N] [--max-new N] [--temperature T] [--top-k K]
//                            [--no-cuda] [--ref "123.45"]          # best-of 必填参考答案
//   reasoning_chat --model <pth> "指令" ["输入"]                    # 单条指令
//   退出交互模式：quit / exit
#include <gflags/gflags.h>
#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "training.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "gpt_sft_core.h"
#include "reasoning_core.h"
#include "gen_utils.h"
#include "reward.h"

#define TOKENIZER_DIR_DEFAULT "/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/chapters/chapter02_text_data/data"

DEFINE_string(model, "data/sft.pth", "模型 .pth 路径");
DEFINE_string(size, "small", "模型规模: small|medium|...");
DEFINE_string(strategy, "greedy", "推理策略: greedy|vote|beam|best-of");
DEFINE_int64(samples, 5, "vote/best-of 采样次数");
DEFINE_int64(max_new, 384, "最大生成 token 数");
DEFINE_double(temperature, 0.7, "采样温度");
DEFINE_int64(top_k, 50, "top-k 采样");
DEFINE_double(top_p, -1, "top-p 采样（>0 才启用）");
DEFINE_string(ref, "", "best-of 策略时必须的参考答案");
DEFINE_bool(no_cuda, false, "强制使用 CPU");
DEFINE_string(tokenizer_dir, TOKENIZER_DIR_DEFAULT, "BPE 数据目录");

void run_one(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
             const std::string& instruction, const std::string& input,
             const std::string& strategy, c10::optional<double> top_p_opt) {
  std::string prompt = reasoning::make_r1_prompt(instruction, input);
  auto dev = model->parameters().empty() ? torch::Device(torch::kCPU)
                                         : model->parameters()[0].device();
  (void)dev;

  auto t0 = std::chrono::steady_clock::now();
  if (strategy == "greedy") {
    auto samples = reasoning::generate_multiple(model, tokenizer, prompt,
                                                FLAGS_max_new, 1024,
                                                /*n=*/1, 0.0, c10::nullopt,
                                                top_p_opt, 50256);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n>>>  (" << strategy << ", " << ms << "ms)\n";
    std::string one_line;
    for (char c : samples[0].text) one_line += (c == '\n') ? ' ' : c;
    std::cout << one_line << "\n";
  } else if (strategy == "vote") {
    auto samples = reasoning::generate_multiple(model, tokenizer, prompt,
                                                FLAGS_max_new, 1024,
                                                FLAGS_samples, FLAGS_temperature,
                                                FLAGS_top_k > 0 ? c10::optional<int64_t>(FLAGS_top_k) : c10::nullopt,
                                                top_p_opt, 50256);
    auto vr = reasoning::majority_vote(samples);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n>>>  (" << strategy << ", " << vr.votes << "/" << vr.total
              << " 得票, " << ms << "ms)\n";
    std::cout << "答案: " << vr.answer << "\n";
    if (samples.size() <= 3) {
      for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "  -- 样本 " << (i + 1) << " 答案: " << samples[i].answer << "\n";
      }
    }
  } else if (strategy == "beam") {
    auto [text, lp] = reasoning::beam_search(model, tokenizer, prompt,
                                             FLAGS_max_new, 1024, 3, 0.0, 50256);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n>>>  (" << strategy << ", log-prob=" << lp << ", " << ms << "ms)\n";
    std::string one_line;
    for (char c : text) one_line += (c == '\n') ? ' ' : c;
    std::cout << one_line << "\n";
  } else if (strategy == "best-of") {
    if (FLAGS_ref.empty()) {
      std::cerr << "best-of 策略需要 --ref <参考答案>\n";
      return;
    }
    auto samples = reasoning::generate_multiple(model, tokenizer, prompt,
                                                FLAGS_max_new, 1024,
                                                FLAGS_samples, FLAGS_temperature,
                                                FLAGS_top_k > 0 ? c10::optional<int64_t>(FLAGS_top_k) : c10::nullopt,
                                                top_p_opt, 50256);
    auto best = reasoning::best_of_n(samples, FLAGS_ref);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n>>>  (" << strategy << ", reward=" << best.reward << ", " << ms << "ms)\n";
    std::string one_line;
    for (char c : best.text) one_line += (c == '\n') ? ' ' : c;
    std::cout << one_line << "\n";
  } else {
    std::cerr << "未知策略: " << strategy << "\n";
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  at::globalContext().setAllowTF32CuBLAS(false);
  torch::Device device = FLAGS_no_cuda
                              ? torch::Device(torch::kCPU)
                              : (torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
  std::cout << "=== reasoning_chat ===\n"
            << "模型: " << FLAGS_model << "\n"
            << "策略: " << FLAGS_strategy << "\n"
            << "设备: " << device << "\n";

  auto cfg = gpt_sft::config_for_size(FLAGS_size);
  auto model = gpt_sft::make_model(cfg);
  torch::load(model, FLAGS_model);
  model->to(device);
  model->eval();

  ch2::BpeTokenizer tokenizer(FLAGS_tokenizer_dir + "/encoder.json",
                              FLAGS_tokenizer_dir + "/vocab.bpe");

  c10::optional<double> top_p_opt = c10::nullopt;
  if (FLAGS_top_p > 0) top_p_opt = FLAGS_top_p;

  // 捕获非选项参数作为 instruction/input
  std::string instruction, input;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() && a[0] != '-' && instruction.empty()) instruction = a;
    else if (a.size() && a[0] != '-' && input.empty()) input = a;
  }

  try {
    if (!instruction.empty()) {
      run_one(model, tokenizer, instruction, input, FLAGS_strategy, top_p_opt);
      return 0;
    }
    std::cout << "\n输入指令（quit 退出）:\n";
    std::string line;
    while (true) {
      std::cout << ">>> ";
      if (!std::getline(std::cin, line)) break;
      if (line == "quit" || line == "exit") break;
      if (line.empty()) continue;
      run_one(model, tokenizer, line, "", FLAGS_strategy, top_p_opt);
      std::cout << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
