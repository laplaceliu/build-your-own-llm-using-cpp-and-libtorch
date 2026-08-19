// sft_serve.cpp
// 推理 HTTP 服务：加载微调模型，对外提供 REST API。
//
// 端点:
//   GET  /health        健康检查 -> {"status":"ok","model":...}
//   POST /v1/chat       推理 -> 请求: {"instruction","input"?,"max_tokens"?}
//                       响应: {"response": "...", "prompt": "...", "tokens": n}
//   POST /v1/generate   generate_text 兼容（OpenAI 风格消息）
//
// 用法:
//   sft_serve --model <pth> [--port 8080] [--size small|medium|large|xl]
//            [--tokenizer <dir>] [--no-cuda]
//
// 示例:
//   curl -X POST localhost:8080/v1/chat \
//        -d '{"instruction":"将以下句子改为被动语态：厨师每天做饭。"}'
#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "gpt_sft_core.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "../chapters/chapter02_text_data/data"
#endif

namespace {

struct ServeOptions {
  std::string model = DATA_DIR "/gpt2-medium-sft.pth";
  std::string size = "medium";
  int port = 8080;
  bool use_cuda = true;
};

ServeOptions parse_args(int argc, char* argv[]) {
  ServeOptions o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("缺少参数: ") + name);
      return argv[++i];
    };
    if (a == "--model") o.model = next("--model");
    else if (a == "--size") o.size = next("--size");
    else if (a == "--port") o.port = std::stoi(next("--port"));
    else if (a == "--no-cuda") o.use_cuda = false;
    else throw std::runtime_error("未知参数: " + a);
  }
  return o;
}

std::string url_decode(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        return (c >= '0' && c <= '9') ? c - '0'
             : (c >= 'a' && c <= 'f') ? c - 'a' + 10
             : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
      };
      out += static_cast<char>(hex(s[i + 1]) * 16 + hex(s[i + 2]));
      i += 2;
    } else {
      out += s[i];
    }
  }
  return out;
}

struct HttpRequest {
  std::string method, path, body;
};

// 极简 HTTP/1.1 解析（支持 GET/POST + JSON body）
bool parse_http(const std::string& raw, HttpRequest& req) {
  auto header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) return false;
  std::string head = raw.substr(0, header_end);
  req.body = raw.substr(header_end + 4);

  std::istringstream hs(head);
  std::string line;
  std::getline(hs, line);
  std::istringstream reqline(line);
  reqline >> req.method >> req.path;

  // 读取 Content-Length 截断 body
  size_t content_length = 0;
  while (std::getline(hs, line) && !line.empty()) {
    if (line.size() > 16 && line.rfind("Content-Length:", 0) == 0) {
      content_length = std::stoul(line.substr(15));
    }
  }
  if (content_length > 0 && req.body.size() > content_length)
    req.body.resize(content_length);
  return true;
}

std::string http_response(int code, const std::string& content_type,
                          const std::string& body) {
  std::ostringstream ss;
  ss << "HTTP/1.1 " << code << " OK\r\n"
     << "Content-Type: " << content_type << "\r\n"
     << "Content-Length: " << body.size() << "\r\n"
     << "Connection: close\r\n\r\n"
     << body;
  return ss.str();
}

void handle_client(int client_fd, ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                   const torch::Device& device) {
  // 读取请求
  std::string raw;
  char buf[8192];
  ssize_t n;
  while ((n = read(client_fd, buf, sizeof(buf))) > 0) {
    raw.append(buf, n);
    if (raw.find("\r\n\r\n") != std::string::npos) break;  // 头读完即止（后续 body 一并读入）
  }

  HttpRequest req;
  std::string response;
  if (!parse_http(raw, req)) {
    response = http_response(400, "application/json", "{\"error\":\"bad request\"}");
  } else if (req.method == "GET" && req.path == "/health") {
    response = http_response(200, "application/json",
                             "{\"status\":\"ok\",\"model\":\"gpt-sft\"}");
  } else if (req.method == "POST" && (req.path == "/v1/chat" ||
                                      req.path == "/v1/generate")) {
    try {
      auto j = nlohmann::json::parse(req.body);
      std::string instruction = j.value("instruction", "");
      std::string input = j.value("input", "");
      int64_t max_tokens = j.value("max_tokens", j.value("max_new_tokens", 128));

      // OpenAI 风格消息（可选）
      if (instruction.empty() && j.contains("messages")) {
        for (const auto& m : j["messages"]) {
          if (m["role"] == "user") { instruction = m["content"]; break; }
        }
      }

      auto reply = gpt_sft::generate_response(model, tokenizer, instruction, input,
                                              max_tokens, device);
      nlohmann::json out = {{"response", reply},
                            {"instruction", instruction},
                            {"input", input}};
      response = http_response(200, "application/json", out.dump());
    } catch (const std::exception& e) {
      response = http_response(400, "application/json",
                               std::string("{\"error\":\"") + e.what() + "\"}");
    }
  } else {
    response = http_response(404, "application/json", "{\"error\":\"not found\"}");
  }
  send(client_fd, response.data(), response.size(), 0);
  close(client_fd);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    at::globalContext().setAllowTF32CuBLAS(false);
    auto opts = parse_args(argc, argv);
    torch::Device device = opts.use_cuda ? gpt_sft::select_device()
                                         : torch::Device(torch::kCPU);
    std::cout << "=== sft_serve（推理服务）===\n"
              << "模型: " << opts.model << "\n"
              << "设备: " << device << "\n";

    auto cfg = gpt_sft::config_for_size(opts.size);
    auto model = gpt_sft::make_model(cfg);
    torch::load(model, opts.model);  // 加载微调 .pth
    model->to(device);
    model->eval();
    std::cout << "[完成] 模型加载\n";

    ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                TOKENIZER_DIR "/vocab.bpe");

    // socket
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(opts.port));
    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      throw std::runtime_error("端口绑定失败: " + std::to_string(opts.port));
    }
    listen(listen_fd, 16);
    std::cout << "[监听] http://0.0.0.0:" << opts.port << "\n"
              << "  健康检查: curl " << opts.port << "/health\n"
              << "  推理:     curl -X POST localhost:" << opts.port
              << "/v1/chat -d '{\"instruction\":\"...\"}'\n";

    std::atomic<int> requests{0};
    for (;;) {
      int client = accept(listen_fd, nullptr, nullptr);
      if (client < 0) continue;
      requests++;
      std::thread([client, &model, &tokenizer, &device] {
        handle_client(client, model, tokenizer, device);
      }).detach();
    }
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}
