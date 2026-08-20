#pragma once
// reasoning/core/reward.h
//
// 奖励函数与答案抽取。RL 训练与评测共用同一实现，保证评测口径一致。
//
// 主要 API：
//   - extract_answer(const std::string& text)
//   - answers_match(const std::string& predicted, const std::string& reference)
//   - format_reward(const std::string& text)
//   - accuracy_reward(const std::string& text, const std::string& reference)
//
// 标签格式约定（与 convert_data.py 的 思考/答案 转换一致）：
//   "think<REASONING>think\n<answer>FINAL</answer>"
//
// 兼容 GSM8K 原始格式：文本尾部的 "#### ANSWER" 也可被 extract_answer 识别。

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace reasoning {

// ---------- 文本归一化 ----------
inline std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

inline std::string to_lower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return r;
}

// 把答案字符串规范化为可比较的形式：
//   去首尾空白 -> 反小写 -> 移除 ', $ % , 等号, 末尾单位 -> 尝试取数值
inline std::string normalize_for_compare(const std::string& raw) {
  std::string s = trim(raw);
  s = to_lower(s);
  // 顶起最后连带 's 或 ' 前缀
  for (const std::string& suf : {"%", "$", "°", " degrees", "f", "c"}) {
    size_t p = s.rfind(suf);
    if (p != std::string::npos && p == s.size() - suf.size()) {
      s = s.substr(0, p);
    }
  }
  // 移除所有逗号与脱字符
  s.erase(std::remove(s.begin(), s.end(), ','), s.end());
  // 移除所有空格
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  // 末尾 "。"
  while (!s.empty() && s.back() == '.') s.pop_back();
  return s;
}

inline bool try_parse_double(const std::string& s, double& out) {
  try {
    size_t pos = 0;
    out = std::stod(s, &pos);
    return pos == s.size();
  } catch (...) {
    return false;
  }
}

// ---------- 答案抽取 ----------
// 优先级：<answer>...</answer>  ->  \boxed{...}  ->  "#### "  ->  末尾 "The answer is ..."
inline std::string extract_answer(const std::string& text) {
  // 1. <answer>...</answer>
  {
    auto p = text.find("<answer>");
    if (p != std::string::npos) {
      auto q = text.find("</answer>", p + 8);
      if (q != std::string::npos) return trim(text.substr(p + 8, q - p - 8));
    }
  }
  // 2. \boxed{...}
  {
    auto p = text.find("\\boxed{");
    if (p != std::string::npos) {
      int depth = 1;
      size_t i = p + 7;
      while (i < text.size() && depth > 0) {
        if (text[i] == '{') ++depth;
        else if (text[i] == '}') --depth;
        ++i;
      }
      if (depth == 0) return trim(text.substr(p + 7, i - p - 8));
    }
  }
  // 3. #### ANSWER (GSM8K)
  {
    auto p = text.rfind("####");
    if (p != std::string::npos) {
      std::string tail = text.substr(p + 4);
      return trim(tail);
    }
  }
  // 4. "The answer is XXX"
  {
    std::regex re(R"((?:the\s+answer\s+is\s*[:\.]?\s*)([^\.\n]+))",
                  std::regex::icase);
    std::smatch m;
    if (std::regex_search(text, m, re)) return trim(m[1].str());
  }
  // 5. 退到全文
  return trim(text);
}

// ---------- 答案比较 ----------
// 答案比较：文本准确 + 数值准确（容差） + 分数化简
inline bool answers_match(const std::string& predicted, const std::string& reference) {
  std::string p = normalize_for_compare(predicted);
  std::string r = normalize_for_compare(reference);
  if (p.empty() || r.empty()) return false;
  if (p == r) return true;

  // 数值容差
  double dp = 0.0, dr = 0.0;
  if (try_parse_double(p, dp) && try_parse_double(r, dr)) {
    if (std::abs(dp - dr) <= 1e-6 * std::max(1.0, std::abs(dr))) return true;
  }
  // 分数字符串 "a/b" 化简后比较
  auto parse_frac = [](const std::string& s, double& val) -> bool {
    auto slash = s.find('/');
    if (slash == std::string::npos) return false;
    double num, den;
    if (!try_parse_double(s.substr(0, slash), num)) return false;
    if (!try_parse_double(s.substr(slash + 1), den) || den == 0) return false;
    val = num / den;
    return true;
  };
  double fp = 0.0, fr = 0.0;
  if (parse_frac(p, fp) && parse_frac(r, fr)) {
    if (std::abs(fp - fr) <= 1e-6 * std::max(1.0, std::abs(fr))) return true;
  }
  return false;
}

// ---------- 奖励 ----------
// 格式奖励：包含 思考 与 <answer> 标签 = 1.0；仅包含其一 = 0.5；都没有 = 0
inline double format_reward(const std::string& text) {
  bool has_think = text.find("think") != std::string::npos;
  bool has_answer = text.find("<answer>") != std::string::npos &&
                    text.find("</answer>") != std::string::npos;
  if (has_think && has_answer) return 1.0;
  if (has_think || has_answer) return 0.5;
  return 0.0;
}

// 准确性奖励：答案匹配 = 1.0；不匹配 = 0.0
inline double accuracy_reward(const std::string& text, const std::string& reference) {
  if (reference.empty()) return 0.0;
  return answers_match(extract_answer(text), reference) ? 1.0 : 0.0;
}

// 综合奖励（已归一化到 [0,1]）：加权 w_fmt * format + w_acc * accuracy
inline double composite_reward(const std::string& text, const std::string& reference,
                              double w_fmt = 0.5, double w_acc = 0.5) {
  double f = format_reward(text);
  double a = accuracy_reward(text, reference);
  return w_fmt * f + w_acc * a;
}

}  // namespace reasoning
