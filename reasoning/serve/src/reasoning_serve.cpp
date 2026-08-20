// reasoning/serve/src/reasoning_serve.cpp
// 推理模型 HTTP 服务（极简实现）
// 端点:
//   GET  /health              -> {"status":"ok"}
//   POST /v1/reason            -> {"instruction":..., "input":..., "strategy":..., "samples":...}
//                          -> {"text":..., "answer":..., "elapsed_ms":..., "strategy":...}
//   POST /v1/answer            -> {"instruction":..., "input":..., "ref":...}
//                          -> {"best":..., "reward":..., "samples":[...]}
//
// 用法: reasoning_serve --model <pth> --size small|medium --port 8080
#include <gflags/gflags.h>
#include <torch/torch.h>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "training.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "gpt_sft_core.h"
#include "reasoning_core.h"
#include "gen_utils.h"
#include "reward.h"

#define TOKENIZER_DIR_DEFAULT "/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/chapters/chapter02_text_data/data"

DEFINE_string(model, "data/sft.pth", "模型 .pth");
DEFINE_string(size, "small", "模型规模");
DEFINE_int32(port, 8080, "绑定端口");
DEFINE_bool(no_cuda, false, "强制 CPU");
DEFINE_string(tokenizer_dir, TOKENIZER_DIR_DEFAULT, "BPE 数据目录");

using json = nlohmann::json;

struct ServerCtx {
  ch4::GPTModel* model;
  ch2::BpeTokenizer* tokenizer;
  std::atomic<bool> stop;
};

std::string http_response(int code, const std::string& body, const std::string& ctype = "application/json") {
  std::ostringstream oss;
  oss << "HTTP/1.1 " << code << " OK\r\n"
      << "Content-Type: " << ctype << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
  return oss.str();
}

std::string extract_body(const std::string& req) {
  auto p = req.find("\r\n\r\n");
  if (p == std::string::npos) return "";
  return req.substr(p + 4);
}

std::string handle_reason(const json& body, ServerCtx& ctx) {
  std::string instruction = body.value("instruction", std::string{});
  std::string input = body.value("input", std::string{});
  std::string strategy = body.value("strategy", std::string{"greedy"});
  int samples = body.value("samples", 5);
  int max_new = body.value("max_new", 384);
  double temperature = body.value("temperature", 0.7);
  int top_k = body.value("top_k", 50);
  std::string prompt = reasoning::make_r1_prompt(instruction, input);
  auto t0 = std::chrono::steady_clock::now();
  json out;
  out["strategy"] = strategy;
  if (strategy == "vote" || strategy == "best-of") {
    auto s = reasoning::generate_multiple(*ctx.model, *ctx.tokenizer, prompt,
                                          max_new, 1024, samples, temperature,
                                          top_k > 0 ? c10::optional<int64_t>(top_k) : c10::nullopt,
                                          c10::nullopt, 50256);
    if (strategy == "vote") {
      auto vr = reasoning::majority_vote(s);
      out["answer"] = vr.answer;
      out["votes"] = vr.votes;
      out["total"] = vr.total;
    } else {
      std::string ref = body.value("ref", std::string{});
      auto best = reasoning::best_of_n(s, ref);
      out["best"] = best.text;
      out["reward"] = best.reward;
      out["answer"] = best.answer;
    }
  } else {
    auto s = reasoning::generate_multiple(*ctx.model, *ctx.tokenizer, prompt,
                                          max_new, 1024, 1, 0.0, c10::nullopt,
                                          c10::nullopt, 50256);
    out["text"] = s[0].text;
    out["answer"] = s[0].answer;
  }
  auto t1 = std::chrono::steady_clock::now();
  out["elapsed_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  return out.dump();
}

void handle_client(int cfd, ServerCtx& ctx) {
  std::string req;
  char buf[4096];
  while (true) {
    ssize_t n = recv(cfd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    req.append(buf, n);
    if (req.find("\r\n\r\n") != std::string::npos) break;
  }
  if (req.empty()) { close(cfd); return; }
  std::string resp;
  if (req.find("GET /health") != std::string::npos) {
    resp = http_response(200, json({{"status", "ok"}}).dump());
  } else if (req.find("POST /v1/reason") != std::string::npos) {
    try {
      auto j = json::parse(extract_body(req));
      resp = http_response(200, handle_reason(j, ctx));
    } catch (const std::exception& e) {
      resp = http_response(400, json({{"error", e.what()}}).dump());
    }
  } else {
    resp = http_response(404, json({{"error", "not found"}}).dump());
  }
  send(cfd, resp.data(), resp.size(), 0);
  close(cfd);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  at::globalContext().setAllowTF32CuBLAS(false);
  torch::Device device = FLAGS_no_cuda
                              ? torch::Device(torch::kCPU)
                              : (torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
  std::cout << "=== reasoning_serve ===\n模型: " << FLAGS_model << "\n端口: " << FLAGS_port
            << "\n设备: " << device << "\n";

  auto cfg = gpt_sft::config_for_size(FLAGS_size);
  auto model = gpt_sft::make_model(cfg);
  torch::load(model, FLAGS_model);
  model->to(device);
  model->eval();
  ch2::BpeTokenizer tokenizer(FLAGS_tokenizer_dir + "/encoder.json",
                              FLAGS_tokenizer_dir + "/vocab.bpe");

  ServerCtx ctx{&model, &tokenizer, false};

  int sfd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(FLAGS_port));
  bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  listen(sfd, 32);
  std::cout << "[ready] 监听 0.0.0.0:" << FLAGS_port << "\n";

  while (!ctx.stop) {
    sockaddr_in cli{};
    socklen_t cl = sizeof(cli);
    int cfd = accept(sfd, reinterpret_cast<sockaddr*>(&cli), &cl);
    if (cfd < 0) continue;
    std::thread(handle_client, cfd, std::ref(ctx)).detach();
  }
  close(sfd);
  return 0;
}
