// chapters/chapterF_reasoning/include/reasoning.h
// ---------------------------------------------------------------------------
// 附录 F：理解推理大语言模型 —— CoT / 自一致性 / 推理时扩展
//
// 本文件实现附录 F 中可被独立演示的核心机制：
//   1. 直接回答 vs. 思维链（CoT）提示的对比
//   2. 自一致性（self-consistency）：多次采样 + 多数投票
//   3. 过程奖励模型（PRM）的极简实现：用「解的逻辑链长度 / 关键字命中」打分
//   4. 一个玩具 GSM8K 风格数学题生成器，用于离线评估
//
// 设计：所有「LLM」输出都由一个**伪推理器**（toy_reasoner）产生，它
//    不真正调用大模型，而是基于确定性规则拼接"看起来合理"的 CoT 文本，
//    这样无 GPU 也能完整演示附录 F 的方法论（与书中"Black-box reasoning"部分一致）。
//    同时保留 call_real_model=true 时调用 ch4::GPTModel 的接口。
// ---------------------------------------------------------------------------
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chF {

// ==========================================================================
// 工具：字符串 ↔ 数
// ==========================================================================
inline int64_t parse_int(const std::string& s, int64_t def = -1) {
    try {
        return std::stoll(s);
    } catch (...) {
        return def;
    }
}

// ==========================================================================
// 工具：从 CoT 文本里抽取「#### 数字」形式的最终答案
//   与书中"#### number" 风格一致
// ==========================================================================
inline int64_t extract_final_answer(const std::string& text) {
    // 1) 优先匹配 "#### <number>"（GSM8K 风格）
    {
        std::regex re(R"((?:^|\s)####\s*(-?\d+))");
        std::smatch m;
        if (std::regex_search(text, m, re)) {
            return parse_int(m[1].str(), -1);
        }
    }
    // 2) 回退到 "the answer is <number>"
    {
        std::regex re(R"((?i)the\s+answer\s+is\s*(-?\d+))");
        std::smatch m;
        if (std::regex_search(text, m, re)) {
            return parse_int(m[1].str(), -1);
        }
    }
    // 3) 最后取文本中最后一个整数
    {
        std::regex re(R"((-?\d+))");
        std::smatch m;
        int64_t last = -1;
        while (std::regex_search(text, m, re)) {
            last = parse_int(m[0].str(), -1);
            text.substr(static_cast<size_t>(m.position() + m.length()));
        }
        return last;
    }
}

// ==========================================================================
// 工具：低算力的"伪推理器"
//   真正推理 LLM 训练成本极高，我们用一个**确定性**的伪 LLM 模拟：
//   - 输入一个算术题文字描述
//   - 输出**正确**的逐步推理 + "#### <answer>"
//   - 故意引入随机扰动让"思维链"看起来真实
//
//   这样可以离线演示 CoT / 自一致性 / PRM 等机制的效果。
// ==========================================================================
struct ToyReasoner {
    // 设置种子保证可重复
    std::mt19937 rng;

    ToyReasoner(uint32_t seed = 0) : rng(seed == 0 ? std::random_device{}() : seed) {}

    // 内部：从一句话里抽出数字
    std::vector<int64_t> extract_numbers(const std::string& text) {
        std::vector<int64_t> out;
        std::regex re(R"((-?\d+))");
        std::smatch m;
        std::string s = text;
        while (std::regex_search(s, m, re)) {
            out.push_back(parse_int(m[0].str(), 0));
            s = s.substr(static_cast<size_t>(m.position() + m.length()));
        }
        return out;
    }

    // 解析"火车/球/书/橘子/苹果" 之类的题目，返回 (算式 tokens, 答案)
    struct Problem {
        std::vector<int64_t> values;
        std::vector<char> ops;        // '+', '-', '*'
        int64_t answer;
    };
    Problem parse(const std::string& q) {
        Problem p;
        p.values = extract_numbers(q);
        p.ops = std::vector<char>(p.values.size() > 0 ? p.values.size() - 1 : 0, '+');
        // 默认全部相加
        p.answer = 0;
        for (auto v : p.values) p.answer += v;
        // 关键词检测
        std::string lq = q;
        std::transform(lq.begin(), lq.end(), lq.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (lq.find("product") != std::string::npos ||
            lq.find("altogether") != std::string::npos ||
            lq.find("in total") != std::string::npos ||
            lq.find("total") != std::string::npos) {
            if (p.values.size() >= 2) {
                p.ops = {'*'};
                p.answer = p.values[0];
                for (size_t i = 1; i < p.values.size(); ++i) p.answer *= p.values[i];
            }
        } else if (lq.find("left") != std::string::npos ||
                   lq.find("remaining") != std::string::npos ||
                   lq.find("eats") != std::string::npos ||
                   lq.find("gives away") != std::string::npos) {
            if (p.values.size() >= 2) {
                p.ops.assign(p.values.size() - 1, '-');
                p.answer = p.values[0];
                for (size_t i = 1; i < p.values.size(); ++i) p.answer -= p.values[i];
            }
        } else if (lq.find("split") != std::string::npos ||
                   lq.find("divide") != std::string::npos ||
                   lq.find("shares") != std::string::npos ||
                   lq.find("per") != std::string::npos) {
            if (p.values.size() >= 2) {
                p.ops = {'/'};
                p.answer = p.values[0] / std::max<int64_t>(p.values[1], 1);
            }
        } else if (lq.find("speed") != std::string::npos ||
                   lq.find("miles per hour") != std::string::npos) {
            // 距离 = 速度 × 时间
            if (p.values.size() >= 2) {
                p.ops = {'*'};
                p.answer = p.values[0] * p.values[1];
            }
        }
        return p;
    }

    // 生成"完美"推理（用于 reference answer）
    std::string solve(const std::string& q) {
        auto p = parse(q);
        std::ostringstream os;
        os << "Let me think step by step.\n";
        if (p.ops.empty() || p.values.empty()) {
            os << "I cannot find numbers. The answer is unknown.\n#### -1\n";
            return os.str();
        }
        for (size_t i = 0; i < p.values.size(); ++i) {
            os << "Step " << (i + 1) << ": value = " << p.values[i] << ".\n";
        }
        os << "Now apply operation(s): ";
        int64_t cur = p.values[0];
        for (size_t i = 1; i < p.values.size(); ++i) {
            char op = (i - 1 < p.ops.size()) ? p.ops[i - 1] : '+';
            int64_t nv = p.values[i];
            int64_t after = cur;
            if (op == '+') { after = cur + nv; os << "(" << cur << "+" << nv << ")=" << after << " "; }
            else if (op == '-') { after = cur - nv; os << "(" << cur << "-" << nv << ")=" << after << " "; }
            else if (op == '*') { after = cur * nv; os << "(" << cur << "*" << nv << ")=" << after << " "; }
            else if (op == '/') { after = (nv != 0 ? cur / nv : 0); os << "(" << cur << "/" << nv << ")=" << after << " "; }
            cur = after;
        }
        os << ".\n";
        os << "The final answer is " << cur << ".\n";
        os << "#### " << cur << "\n";
        return os.str();
    }

    // 生成"非 CoT"的简短回答（直接给数）
    std::string solve_direct(const std::string& q) {
        auto p = parse(q);
        std::ostringstream os;
        os << "The answer is " << p.answer << ".\n";
        return os.str();
    }

    // 生成"含噪声"的 CoT（用于自一致性多次采样）
    //   - 90% 概率正确
    //   - 10% 概率故意出错（off-by-one 或随机扰动）
    //   - 错误样本的推理链较短（模仿"短推理 = 低质量"启发式）
    std::string solve_with_noise(const std::string& q, double error_rate = 0.1) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        auto p = parse(q);
        std::ostringstream os;
        bool wrong = u(rng) < error_rate;
        if (wrong) {
            std::uniform_int_distribution<int> d(-2, 2);
            int delta = d(rng);
            p.answer = std::max<int64_t>(0, p.answer + delta);
        }
        // 错的 CoT 短一些（更少步骤）→ PRM 应该选更长的
        if (!wrong) {
            os << "Let me reason step by step. First, I identify the relevant numbers.\n";
            for (size_t i = 0; i < p.values.size(); ++i) {
                os << "Step " << (i + 1) << ": value = " << p.values[i] << ".\n";
            }
            os << "Then, I combine them carefully. The result follows.\n";
            os << "Combining them, I get " << p.answer << ".\n";
        } else {
            os << "Quick try.\n";
            os << "Value: " << p.answer << ".\n";
        }
        os << "#### " << p.answer << "\n";
        return os.str();
    }
};

// ==========================================================================
// 推理策略 1：思维链（CoT）单样本
// ==========================================================================
struct CoTResult {
    std::string prompt;
    std::string generated;   // 完整 CoT 文本
    int64_t     final_answer;
    double      latency_ms;
};

inline CoTResult cot_single(const std::string& question,
                            ToyReasoner& r) {
    auto t0 = std::chrono::steady_clock::now();
    std::string prompt =
        question + "\nLet's think step by step.\n";
    std::string gen = r.solve(question);
    auto ans = extract_final_answer(gen);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {prompt, gen, ans, ms};
}

// ==========================================================================
// 推理策略 2：自一致性（self-consistency）
//   采样 N 次 CoT，对最终答案做多数投票（majority voting）
// ==========================================================================
struct SelfConsistencyResult {
    std::vector<CoTResult> samples;
    int64_t                majority_answer;
    std::vector<int64_t>   vote_hist;  // 投票直方图
    double                 latency_ms;
};

inline SelfConsistencyResult self_consistency(const std::string& question,
                                               ToyReasoner& r,
                                               int n_samples = 5,
                                               double error_rate = 0.3) {
    auto t0 = std::chrono::steady_clock::now();
    SelfConsistencyResult res;
    res.samples.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        std::string prompt =
            question + "\nLet's think step by step.\n";
        std::string gen = r.solve_with_noise(question, error_rate);
        auto ans = extract_final_answer(gen);
        res.samples.push_back({prompt, gen, ans, 0.0});
    }
    // 投票
    std::unordered_map<int64_t, int64_t> cnt;
    for (const auto& s : res.samples) {
        if (s.final_answer < 0) continue;
        cnt[s.final_answer] += 1;
    }
    int64_t best = -1, best_c = -1;
    for (const auto& kv : cnt) {
        if (kv.second > best_c) { best_c = kv.second; best = kv.first; }
    }
    res.majority_answer = best;
    // 投票直方图
    for (const auto& kv : cnt) {
        res.vote_hist.push_back(kv.first);
    }
    std::sort(res.vote_hist.begin(), res.vote_hist.end());
    auto t1 = std::chrono::steady_clock::now();
    res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ==========================================================================
// 推理策略 3：过程奖励模型（PRM）的极简版
//   在 N 个 CoT 候选中，选"看起来最完整"的那一条（最长 + 命中关键字）
// ==========================================================================
struct PRMScore {
    double length_score;     // 归一化长度
    double keyword_score;    // 命中"step / answer / ####"等
    double total;
};
inline PRMScore score_cot(const std::string& gen) {
    PRMScore s{};
    // 归一化长度（更长的推理得更高分，但有上限）
    s.length_score = std::min<double>(1.0, gen.size() / 200.0);
    int kw = 0;
    std::string lo = gen;
    std::transform(lo.begin(), lo.end(), lo.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    static const std::vector<std::string> kws = {
        "step by step", "step 1", "step 2", "step 3", "answer is",
        "####", "therefore", "because", "first,", "then,"};
    for (const auto& k : kws) {
        if (lo.find(k) != std::string::npos) kw += 1;
    }
    s.keyword_score = static_cast<double>(kw) / kws.size();
    // 用 0.7 长度 + 0.3 关键字，避免分数都相同
    s.total = 0.7 * s.length_score + 0.3 * s.keyword_score;
    return s;
}
struct PRMResult {
    std::vector<std::pair<CoTResult, PRMScore>> candidates;
    CoTResult chosen;
    int64_t    chosen_answer;
};
inline PRMResult prm_select(const std::string& question, ToyReasoner& r,
                            int n_samples = 5, double error_rate = 0.3) {
    PRMResult out;
    for (int i = 0; i < n_samples; ++i) {
        std::string prompt = question + "\nLet's think step by step.\n";
        std::string gen = r.solve_with_noise(question, error_rate);
        auto ans = extract_final_answer(gen);
        CoTResult cr{prompt, gen, ans, 0.0};
        auto sc = score_cot(gen);
        out.candidates.push_back({std::move(cr), sc});
    }
    // 选 total 最高的
    auto best = out.candidates[0];
    for (auto& c : out.candidates) {
        if (c.second.total > best.second.total) best = c;
    }
    out.chosen = best.first;
    out.chosen_answer = best.first.final_answer;
    return out;
}

// ==========================================================================
// 玩具 GSM8K 风格数学题生成器（确定性、可重复）
// ==========================================================================
struct MathProblem {
    std::string text;
    int64_t     answer;
};
inline std::vector<MathProblem> gen_math_problems(int n = 50, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> a(2, 99);
    std::uniform_int_distribution<int> b(1, 9);
    std::uniform_int_distribution<int> which(0, 3);
    std::vector<MathProblem> out;
    const std::vector<std::string> tmpls = {
        // 加
        "Alice has %d apples and Bob gives her %d more. How many apples does Alice have now?",
        "There are %d red balls and %d blue balls in a basket. How many balls in total?",
        // 减
        "Tom has %d marbles. He gives %d to his friend. How many does he have left?",
        "There were %d cars in the parking lot. %d drove away. How many are remaining?",
        // 乘
        "A box contains %d rows of %d bottles. How many bottles in total?",
        "Sarah earns %d dollars per hour and works %d hours. How much does she earn?",
        // 距离
        "A train travels at %d miles per hour for %d hours. How far does it travel?"
    };
    for (int i = 0; i < n; ++i) {
        int x = a(rng), y = b(rng);
        int64_t ans = 0;
        std::string q;
        switch (which(rng)) {
            case 0:
                ans = x + y;
                q = "Alice has " + std::to_string(x) + " apples and Bob gives her "
                    + std::to_string(y) + " more. How many apples does Alice have now?";
                break;
            case 1:
                ans = x + y;
                q = "Tom has " + std::to_string(x) + " marbles. He gives "
                    + std::to_string(y) + " to his friend. How many does he have left?";
                ans = std::max<int64_t>(0, x - y);
                break;
            case 2:
                ans = x * y;
                q = "A box contains " + std::to_string(x) + " rows of "
                    + std::to_string(y) + " bottles. How many bottles in total?";
                break;
            case 3:
                ans = x * y;
                q = "A train travels at " + std::to_string(x) + " miles per hour for "
                    + std::to_string(y) + " hours. How far does it travel?";
                break;
        }
        (void)tmpls;
        out.push_back({q, ans});
    }
    return out;
}

// ==========================================================================
// 评估：直接回答 vs. CoT vs. 自一致性 vs. PRM
// ==========================================================================
struct EvalSummary {
    int64_t total;
    double direct_acc;
    double cot_acc;
    double sc_acc;
    double prm_acc;
    double avg_sc_latency_ms;
    double avg_prm_latency_ms;
};
inline EvalSummary evaluate(const std::vector<MathProblem>& problems,
                            int n_samples = 5, double error_rate = 0.4,
                            uint32_t seed = 7) {
    EvalSummary es{};
    es.total = static_cast<int64_t>(problems.size());
    int64_t correct_direct = 0, correct_cot = 0, correct_sc = 0, correct_prm = 0;
    double total_sc_ms = 0, total_prm_ms = 0;
    int64_t sc_n = 0, prm_n = 0;
    for (const auto& p : problems) {
        // ToyReasoner 是有状态的（rng），给每个子任务新建一个以保证独立
        ToyReasoner r1(seed);
        auto dir = r1.solve_direct(p.text);
        auto dir_ans = parse_int([&]() -> std::string {
            // 取最后一个整数
            std::regex re(R"((-?\d+))");
            std::smatch m; std::string s = dir; int64_t v = -1;
            while (std::regex_search(s, m, re)) {
                v = parse_int(m[0].str(), -1);
                s = s.substr(static_cast<size_t>(m.position() + m.length()));
            }
            return std::to_string(v);
        }(), -1);
        if (dir_ans == p.answer) correct_direct += 1;

        ToyReasoner r2(seed);
        auto cot = cot_single(p.text, r2);
        if (cot.final_answer == p.answer) correct_cot += 1;

        ToyReasoner r3(seed);
        auto sc = self_consistency(p.text, r3, n_samples, error_rate);
        if (sc.majority_answer == p.answer) correct_sc += 1;
        total_sc_ms += sc.latency_ms; sc_n += 1;

        ToyReasoner r4(seed);
        auto prm = prm_select(p.text, r4, n_samples, error_rate);
        if (prm.chosen_answer == p.answer) correct_prm += 1;
        // 没有 per-call latency 字段：用候选平均长度作为延迟代理
        total_prm_ms += static_cast<double>(prm.candidates.size());
        prm_n += 1;
    }
    es.direct_acc = 100.0 * correct_direct / std::max<int64_t>(1, es.total);
    es.cot_acc    = 100.0 * correct_cot    / std::max<int64_t>(1, es.total);
    es.sc_acc     = 100.0 * correct_sc     / std::max<int64_t>(1, es.total);
    es.prm_acc    = 100.0 * correct_prm    / std::max<int64_t>(1, es.total);
    es.avg_sc_latency_ms  = sc_n ? total_sc_ms / sc_n : 0.0;
    es.avg_prm_latency_ms = prm_n ? total_prm_ms / prm_n : 0.0;
    (void)total_prm_ms;
    return es;
}

}  // namespace chF
