// reasoning/eval/src/eval_math.cpp
// GSM8K 推理评测器：给定 1 个或多个模型 + 多种推理策略，输出准确率/格式合规率/平均耗时。
//
// 用法:
//   eval_math --data eval-test.json [--size small|medium] [--models pth1,pth2,pth3]
//             [--strategies greedy,vote,beam] [--samples N] [--max-new N]
//             [--out compare.csv] [--no-cuda]
//
// 输出 CSV 列：model, strategy, total, correct, accuracy, format_rate, avg_ms, total_s
#include <gflags/gflags.h>
#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "training.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gpt_sft_core.h"
#include "reasoning_core.h"
#include "gen_utils.h"
#include "reward.h"

#define TOKENIZER_DIR_DEFAULT "/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/chapters/chapter02_text_data/data"

DEFINE_string(data, "data/eval-test.json", "评测 JSON");
DEFINE_string(size, "small", "模型规模");
DEFINE_string(models, "", "模型 .pth 列表，逗号分隔（空则使用 --model）");
DEFINE_string(model, "data/sft.pth", "单模型 .pth（当 --models 为空）");
DEFINE_string(strategies, "greedy", "策略列表：greedy,vote,beam,best-of（best-of 需 --ref 列）");
DEFINE_string(names, "", "模型显示名（与 --models 顺序一致，空则用文件名）");
DEFINE_int64(samples, 5, "vote/best-of 采样次数");
DEFINE_int64(max_new, 384, "最大生成 token");
DEFINE_double(temperature, 0.7, "采样温度");
DEFINE_int64(top_k, 50, "top-k 采样");
DEFINE_int64(limit, 0, "评测样本数上限（0 = 全部）");
DEFINE_string(out, "data/compare.csv", "输出 CSV");
DEFINE_bool(no_cuda, false, "强制 CPU");
DEFINE_string(tokenizer_dir, TOKENIZER_DIR_DEFAULT, "BPE 数据目录");

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) if (!item.empty()) out.push_back(item);
  return out;
}

struct Result {
  std::string model_name;
  std::string strategy;
  int total = 0;
  int correct = 0;
  int format_ok = 0;
  double total_ms = 0.0;
};

Result eval_one(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                const std::vector<reasoning::ReasoningItem>& items,
                const std::string& strategy, const std::string& model_name,
                int64_t max_new, int64_t samples, double temperature,
                c10::optional<int64_t> top_k) {
  Result r{model_name, strategy, 0, 0, 0, 0.0};
  // 探测模型所在设备：直接看参数的设备更稳，避免 forward 时撞 device/dtype 不匹配
  auto dev = model->parameters().empty() ? torch::Device(torch::kCPU)
                                         : model->parameters()[0].device();
  for (const auto& it : items) {
    std::string prompt = reasoning::make_r1_prompt(it.instruction, it.input);
    auto t0 = std::chrono::steady_clock::now();
    std::string text, answer;
    bool format_ok = false;
    if (strategy == "greedy") {
      auto sa = reasoning::generate_multiple(model, tokenizer, prompt, max_new, 1024,
                                              1, 0.0, c10::nullopt, c10::nullopt, 50256);
      text = sa[0].text; answer = sa[0].answer;
    } else if (strategy == "vote") {
      auto sa = reasoning::generate_multiple(model, tokenizer, prompt, max_new, 1024,
                                              samples, temperature, top_k, c10::nullopt, 50256);
      auto vr = reasoning::majority_vote(sa);
      text = sa[0].text; answer = vr.answer;
    } else if (strategy == "beam") {
      auto [t, lp] = reasoning::beam_search(model, tokenizer, prompt, max_new, 1024, 3, 0.0, 50256);
      text = t; answer = reasoning::extract_answer(t);
    } else {
      continue;
    }
    // 调试：每条打印前 300 字符的生成（截断避免刷屏）
    if (FLAGS_limit <= 3) {
      std::string snippet = text.size() > 300 ? text.substr(0, 300) + "..." : text;
      std::cout << "\n  [Q] " << (it.instruction.size() > 100 ? it.instruction.substr(0, 100) + "..." : it.instruction)
                << "\n  [A] gold=" << it.gold << " pred='" << answer << "' fmt=" << (answer.find('<') != std::string::npos ? "yes" : "no")
                << "\n  [T] " << snippet << "\n";
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    r.total++;
    r.total_ms += ms;
    if (reasoning::answers_match(answer, it.gold)) r.correct++;
    if (reasoning::format_reward(text) >= 1.0) r.format_ok++;
  }
  return r;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  at::globalContext().setAllowTF32CuBLAS(false);
  torch::Device device = FLAGS_no_cuda
                              ? torch::Device(torch::kCPU)
                              : (torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
  std::cout << "=== eval_math ===\n数据: " << FLAGS_data << "\n设备: " << device << "\n";

  auto items = reasoning::load_jsonl_or_json(FLAGS_data);
  if (FLAGS_limit > 0 && FLAGS_limit < (int64_t)items.size()) {
    items.resize(FLAGS_limit);
  }
  std::cout << "评测样本: " << items.size() << "\n";
  if (items.empty()) { std::cerr << "无评测数据\n"; return 1; }

  ch2::BpeTokenizer tokenizer(FLAGS_tokenizer_dir + "/encoder.json",
                              FLAGS_tokenizer_dir + "/vocab.bpe");

  std::vector<std::string> model_paths = FLAGS_models.empty() ?
      std::vector<std::string>{FLAGS_model} : split(FLAGS_models, ',');
  std::vector<std::string> names = FLAGS_names.empty() ? model_paths :
      split(FLAGS_names, ',');
  while (names.size() < model_paths.size()) names.push_back(model_paths[names.size()]);

  std::vector<std::string> strategies = split(FLAGS_strategies, ',');
  c10::optional<int64_t> top_k = FLAGS_top_k > 0 ? c10::optional<int64_t>(FLAGS_top_k) : c10::nullopt;

  std::vector<Result> results;
  for (size_t mi = 0; mi < model_paths.size(); ++mi) {
    std::cout << "\n--- 模型: " << names[mi] << " (" << model_paths[mi] << ") ---\n";
    auto cfg = gpt_sft::config_for_size(FLAGS_size);
    auto model = gpt_sft::make_model(cfg);
    try {
      torch::load(model, model_paths[mi]);
    } catch (const std::exception& e) {
      std::cerr << "  [警告] 加载失败: " << e.what() << "，跳过\n";
      continue;
    }
    model->to(device);
    model->eval();
    for (const auto& st : strategies) {
      auto r = eval_one(model, tokenizer, items, st, names[mi],
                        FLAGS_max_new, FLAGS_samples, FLAGS_temperature, top_k);
      std::cout << "  " << st << ": acc=" << (r.total ? (double)r.correct / r.total : 0.0)
                << " fmt=" << (r.total ? (double)r.format_ok / r.total : 0.0)
                << " avg=" << (r.total ? r.total_ms / r.total : 0.0) << "ms\n";
      results.push_back(r);
    }
  }

  // 输出 CSV
  std::ofstream csv(FLAGS_out);
  csv << "model,strategy,total,correct,accuracy,format_rate,avg_ms,total_s\n";
  for (const auto& r : results) {
    double acc = r.total ? (double)r.correct / r.total : 0.0;
    double fmt = r.total ? (double)r.format_ok / r.total : 0.0;
    double avg_ms = r.total ? r.total_ms / r.total : 0.0;
    double total_s = r.total_ms / 1000.0;
    csv << r.model_name << "," << r.strategy << "," << r.total << ","
        << r.correct << "," << acc << "," << fmt << "," << avg_ms << ","
        << total_s << "\n";
  }
  std::cout << "\nCSV 已写入: " << FLAGS_out << "\n";
  return 0;
}
