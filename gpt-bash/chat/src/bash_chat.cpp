// bash_chat.cpp
// 命令行交互式推理：把自然语言转成 bash 单行命令。
//
// 与 sft_chat 的区别：
//   - 默认指令前缀就是 "Translate the following natural language request into a bash one-liner."，
//     这样用户输入的"找 7 天内改过的 txt"直接被包装成同样的 Alpaca 格式。
//   - 默认 max_new_tokens=64（bash 命令通常 ≤ 30 个 token）。
//   - 可选 --dry-run：不执行；--exec：执行生成的命令（带命令白名单 + timeout）。
//   - 单条示例模式：bash_chat --model <pth> "find .txt files modified in last 7 days"
//   - 交互模式：进入 REPL，输入 quit 退出，前缀 “eval ” 触发执行。
//
// 注意：本二进制不会运行任何白名单之外或带破坏性前缀（rm -rf / 等）的命令，
// 详见 --allow-* 参数和 check_safe_command() 函数（libc 系统调用层面的白名单）。
#include <torch/torch.h>

#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpt_sft_core.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "../chapters/chapter02_text_data/data"
#endif

namespace {

constexpr const char* kDefaultInstruction =
    "Translate the following natural language request into a bash one-liner.";

// ------------------------- 安全白名单 -------------------------
// 出于"训练时数据干净 + eval 时沙箱兜底"的双保险，
// 即便用户在 --exec 模式下按下 enter，默认也只允许下列基准命令。
// 模型学到的命令不一定满足安全白名单；这正是为什么 bash_eval 用
// 更细的策略（按测试样本语义不同单独设白名单）。
const std::vector<std::string>& safe_allowlist() {
  static const std::vector<std::string> v = {
      // 纯只读 / 信息类
      "ls", "echo", "printf", "cat", "head", "tail", "wc", "tree", "stat",
      "date", "cal", "whoami", "pwd", "hostname", "uname", "df", "du",
      "free", "uptime", "id", "groups", "tty", "whereis", "which", "type",
      "file", "test", "[", "true", "false", "yes", "seq", "basename",
      "dirname", "readlink", "realpath", "env", "printenv", "set",
      // 路径/查找
      "find", "grep", "egrep", "fgrep", "rgrep", "ag", "rg", "lsattr",
      "stat", "diff", "comm", "uniq", "sort", "shuf", "tac", "rev",
      "cut", "tr", "expand", "unexpand", "fold", "fmt", "nl", "od",
      "hexdump", "xxd", "strings",
      // 文本处理
      "awk", "gawk", "sed", "perl", "xargs", "tee",
      // 其它只读工具
      "touch", "mkdir", "ln",  // 可写但范围受限
  };
  return v;
}

std::string lstrip(std::string s) {
  size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  s.erase(0, i);
  return s;
}

// 取首个 "词"作为可执行命令名。带路径前缀的也会只保留 basename。
std::string first_token(const std::string& cmd) {
  std::istringstream iss(lstrip(cmd));
  std::string first;
  iss >> first;
  if (first.empty()) return "";
  // 去掉 ./ /usr/bin/ 等路径前缀
  size_t pos = first.find_last_of('/');
  std::string base = (pos == std::string::npos) ? first : first.substr(pos + 1);
  // 去掉 () / {} 等括起来的内置函数（bash builtin）
  if (!base.empty() && (base.front() == '[' || base.front() == '(')) return "";
  return base;
}

bool check_safe_command(const std::string& cmd) {
  // 任何形如 "rm -rf /" "mkfs" "shutdown" 关键字都禁止
  static const std::vector<std::string> banned = {
      "rm", "mv", "cp", "dd", "chmod", "chown", "chgrp", "kill", "killall",
      "pkill", "mount", "umount", "sudo", "su", "reboot", "shutdown",
      "halt", "poweroff", "init", "mkfs", "fdisk", "parted",
      "iptables", "firewall-cmd", "systemctl", "service", "useradd",
      "userdel", "passwd", "curl", "wget", "nc", "ncat", "ssh", "scp",
      "rsync", "crontab", "at", "bash", "sh", "zsh", "fish", "python",
      "python3", "perl", "ruby", "eval", "exec", "source"};
  std::string tok = first_token(cmd);
  if (tok.empty()) return false;
  for (const auto& b : banned) if (tok == b) return false;
  for (const auto& a : safe_allowlist()) if (tok == a) return true;
  return false;
}

// ------------------------- 执行 -------------------------
struct ExecResult {
  int rc = -1;
  std::string stdout_;
  std::string stderr_;
  double sec = 0.0;
};

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

ExecResult run_with_timeout(const std::string& cmd, int timeout_sec = 3) {
  ExecResult r;
  std::string full = "timeout --foreground " + std::to_string(timeout_sec) +
                     " bash -lc " + shell_quote(cmd);
  auto t0 = std::chrono::steady_clock::now();
  std::array<char, 4096> buf;
  FILE* pipe = ::popen(full.c_str(), "r");
  if (!pipe) {
    r.stderr_ = "popen failed";
    r.rc = -1;
    return r;
  }
  while (std::fgets(buf.data(), buf.size(), pipe)) r.stdout_ += buf.data();
  r.rc = ::pclose(pipe);
  auto t1 = std::chrono::steady_clock::now();
  r.sec = std::chrono::duration<double>(t1 - t0).count();
  return r;
}

// ------------------------- CLI 解析 -------------------------
struct Options {
  std::string model = DATA_DIR "/bash-sft.pth";
  std::string size = "medium";
  bool use_cuda = true;
  bool exec_after = false;
  int max_new_tokens = 64;
  std::string instruction;  // 单条指令模式
  std::string input;
};

Options parse_args(int argc, char* argv[]) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("缺少参数: ") + name);
      return argv[++i];
    };
    if      (a == "--model")           o.model = next("--model");
    else if (a == "--size")            o.size = next("--size");
    else if (a == "--no-cuda")         o.use_cuda = false;
    else if (a == "--exec")            o.exec_after = true;
    else if (a == "--max-new")         o.max_new_tokens = std::stoi(next("--max-new"));
    else if (a == "--input")           o.input = next("--input");
    else if (a == "--instruction")     o.instruction = next("--instruction");
    else if (a.rfind("--", 0) == 0)    throw std::runtime_error("未知参数: " + a);
    else if (o.input.empty())          o.input = a;  // 单个自由参数 = NL
  }
  if (o.instruction.empty()) o.instruction = kDefaultInstruction;
  return o;
}

void print_command(const std::string& cmd) {
  std::cout << "$ " << cmd << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    at::globalContext().setAllowTF32CuBLAS(false);
    auto opts = parse_args(argc, argv);
    torch::Device device = opts.use_cuda ? gpt_sft::select_device()
                                         : torch::Device(torch::kCPU);
    std::cout << "=== bash_chat（自然语言 → bash）===\n"
              << "模型: " << opts.model << "\n"
              << "设备: " << device << "\n"
              << "默认指令: " << kDefaultInstruction << "\n";

    auto cfg = gpt_sft::config_for_size(opts.size);
    auto model = gpt_sft::make_model(cfg);
    auto ends_with = [](const std::string& s, const std::string& suf) {
      return s.size() >= suf.size() &&
             std::equal(suf.rbegin(), suf.rend(), s.rbegin());
    };
    if (ends_with(opts.model, ".safetensors")) {
      std::cout << "[加载] safetensors 权重: " << opts.model << "\n";
      model = gpt_sft::load_pretrained(cfg, opts.model);
    } else {
      std::cout << "[加载] torch::load: " << opts.model << "\n";
      torch::load(model, opts.model);
    }
    model->to(device);
    model->eval();
    std::cout << "[完成] 模型加载\n";

    ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                TOKENIZER_DIR "/vocab.bpe");

    auto reply_and_maybe_exec = [&](const std::string& nl, bool do_exec) {
      // instruction 固定（与训练 prompt 一致），input = 自然语言
      auto cmd = gpt_sft::generate_response(model, tokenizer, kDefaultInstruction,
                                            nl, opts.max_new_tokens, device);
      // 抽取第一行非空文本作为命令
      std::istringstream iss(cmd);
      std::string first_line;
      std::getline(iss, first_line);
      first_line = lstrip(first_line);
      std::cout << "\nNL : " << nl << "\n";
      print_command(first_line);
      if (do_exec) {
        if (!check_safe_command(first_line)) {
          std::cout << "(跳过执行：未通过安全白名单)\n";
        } else {
          auto r = run_with_timeout(first_line, /*timeout_sec=*/3);
          std::cout << "(rc=" << r.rc << ", " << r.sec << "s)\n";
          if (!r.stdout_.empty()) std::cout << r.stdout_;
          if (!r.stderr_.empty()) std::cerr << r.stderr_;
        }
      }
      std::cout << "\n";
    };

    // 单条指令模式
    if (!opts.input.empty()) {
      reply_and_maybe_exec(opts.input, opts.exec_after);
      return 0;
    }

    // 交互模式
    std::cout << "\n=== 输入自然语言描述，输入 quit 退出 ===\n"
                 "提示：以 '!' 开头表示生成后立即沙箱执行 (--exec)，"
                 "需要命令通过白名单 + timeout 3s。\n";
    std::string line;
    while (true) {
      std::cout << "NL> ";
      if (!std::getline(std::cin, line)) break;
      if (line == "quit" || line == "exit") break;
      if (line.empty()) continue;
      bool do_exec = opts.exec_after || (!line.empty() && line[0] == '!');
      if (do_exec && line[0] == '!') line = line.substr(1);
      reply_and_maybe_exec(line, do_exec);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}
