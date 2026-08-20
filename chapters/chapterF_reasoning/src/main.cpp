// chapters/chapterF_reasoning/src/main.cpp
// ---------------------------------------------------------------------------
// 附录 F：理解推理大语言模型 —— CoT / 自一致性 / 推理时扩展（C++/LibTorch GPU）
//
// 本程序演示附录 F 总结的 4 个核心思想：
//   F.1  直接回答（baseline）
//   F.2  思维链（CoT）单样本
//   F.3  自一致性（self-consistency）：N 次采样 + 多数投票
//   F.4  过程奖励模型（PRM）：在 N 个候选里选最佳 CoT
//   F.5  对四种策略在玩具数学题上的准确率 / 延迟做汇总
//
// 注：本仓库不下载 OpenAI GPT-4/4o / DeepSeek R1 等外部推理模型，
//      而是基于确定性伪推理器（chF::ToyReasoner）演示**方法论**。
//      这与书中"Black-box reasoning"的论点完全一致。
// ---------------------------------------------------------------------------
#include "reasoning.h"

#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(2);

    bool use_cuda = false;
    int  n_problems = 20;
    int  n_samples  = 7;
    double error_rate = 0.4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cuda") use_cuda = true;
        else if (a == "-n" && i + 1 < argc) n_problems = std::stoi(argv[++i]);
        else if (a == "-k" && i + 1 < argc) n_samples  = std::stoi(argv[++i]);
        else if (a == "-e" && i + 1 < argc) error_rate = std::stod(argv[++i]);
    }

    torch::Device device = torch::kCPU;
    if (use_cuda && torch::cuda::is_available()) {
        std::cout << "[appendix-F] CUDA available, using GPU\n";
        device = torch::kCUDA;
    } else {
        std::cout << "[appendix-F] using CPU\n";
    }
    (void)device;  // 伪推理器不在 GPU 上跑（只演示方法论）

    // ---------------- F.0 演示：单个例子 ----------------
    std::cout << "\n========== F.0 demo: one math problem ==========\n";
    {
        std::string q = "A farmer has 12 chickens. He buys 5 more, then sells 3. "
                        "How many chickens does he have now?";
        std::cout << "Q: " << q << "\n\n";

        std::cout << "--- F.1 direct answer ---\n";
        chF::ToyReasoner r0(1);
        std::cout << r0.solve_direct(q) << "\n";

        std::cout << "--- F.2 chain-of-thought (single) ---\n";
        chF::ToyReasoner r1(2);
        auto cot = chF::cot_single(q, r1);
        std::cout << "Prompt: " << cot.prompt << "\n";
        std::cout << "Generated:\n" << cot.generated << "\n";
        std::cout << "Parsed answer: " << cot.final_answer
                  << "  (latency " << cot.latency_ms << " ms)\n";

        std::cout << "--- F.3 self-consistency (k=" << n_samples << ") ---\n";
        chF::ToyReasoner r2(3);
        auto sc = chF::self_consistency(q, r2, n_samples, error_rate);
        for (size_t i = 0; i < sc.samples.size(); ++i) {
            std::cout << " sample " << i << ": answer=" << sc.samples[i].final_answer
                      << "\n";
        }
        std::cout << "Vote histogram: ";
        for (auto v : sc.vote_hist) std::cout << v << " ";
        std::cout << "\nMajority answer: " << sc.majority_answer
                  << "  (latency " << sc.latency_ms << " ms)\n";

        std::cout << "--- F.4 PRM (process reward model) ---\n";
        chF::ToyReasoner r3(4);
        auto prm = chF::prm_select(q, r3, n_samples, error_rate);
        for (const auto& c : prm.candidates) {
            std::cout << "  cand answer=" << c.first.final_answer
                      << "  len=" << c.second.length_score
                      << "  kw="  << c.second.keyword_score
                      << "  total=" << c.second.total << "\n";
        }
        std::cout << "Chosen answer: " << prm.chosen_answer << "\n";
        std::cout << "Chosen text:\n" << prm.chosen.generated << "\n";
    }

    // ---------------- F.5 大批量评估 ----------------
    std::cout << "\n========== F.5 batch evaluation ==========\n";
    std::cout << "Generating " << n_problems << " synthetic math problems...\n";
    auto problems = chF::gen_math_problems(n_problems, /*seed=*/42);
    std::cout << "Examples (first 5):\n";
    for (int i = 0; i < 5 && i < n_problems; ++i) {
        std::cout << "  Q: " << problems[i].text << "\n";
        std::cout << "  A: " << problems[i].answer << "\n\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    auto es = chF::evaluate(problems, n_samples, error_rate, /*seed=*/7);
    auto t1 = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n===== summary =====\n";
    std::cout << "Problems: " << es.total << "\n";
    std::cout << "Samples per CoT (k): " << n_samples << "\n";
    std::cout << "Per-sample error rate: " << error_rate << "\n";
    std::cout << "------------------------------------------\n";
    std::cout << "Strategy      | accuracy | avg latency\n";
    std::cout << "------------------------------------------\n";
    std::cout << "Direct        | " << es.direct_acc << " %  | "
              << "  ~1 ms\n";
    std::cout << "CoT (single)  | " << es.cot_acc    << " %  | "
              << "  ~1 ms\n";
    std::cout << "CoT+SC (vote) | " << es.sc_acc     << " %  | "
              << es.avg_sc_latency_ms << " ms / problem\n";
    std::cout << "CoT+PRM       | " << es.prm_acc    << " %  | "
              << es.avg_prm_latency_ms << " ms / problem\n";
    std::cout << "------------------------------------------\n";
    std::cout << "Total time: " << total_sec << " s\n";

    return 0;
}
