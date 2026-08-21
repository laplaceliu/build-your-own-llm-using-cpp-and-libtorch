# X.3 gpt-sft 框架：CLI + HTTP 服务化

> 扩展项目第 3 帖｜阅读 6 分钟｜口播 4 分钟
> 平台：★ 知乎

---

## 钩子

**训练好的模型怎么用？CLI 一行启动，HTTP 端口开放。**

---

## 项目结构

```
gpt-sft/
├── README.md
├── core/                      # 核心静态库
│   ├── model.h
│   ├── tokenizer.h
│   ├── dataset.h
│   └── trainer.h
├── train/                     # 训练 CLI
│   └── sft_train.cpp
├── chat/                      # 单轮推理
│   └── chat.cpp
├── serve/                     # HTTP 服务
│   └── sft_serve.cpp
└── scripts/
    ├── train.sh
    └── serve.sh
```

---

## core：所有逻辑的静态库

```cpp
// core/model.h
class GPTModel {
public:
    GPTModel(GPTConfig config);
    
    torch::Tensor forward(torch::Tensor ids);
    
    void load_weights(const std::string& path);
    void save_weights(const std::string& path);
};

// core/tokenizer.h
class BPETokenizer {
public:
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};

// core/trainer.h
class SFTTrainer {
public:
    void train(GPTModel& model,
               std::vector<InstructionExample>& data,
               TrainingConfig config);
};
```

**优势**：3 个项目（gpt-bash / gpt-toolcall / gpt-sft）都复用 core，互不耦合。

---

## sft_train：训练 CLI

```bash
$ ./build/sft_train --help
Usage: sft_train [OPTIONS]

Options:
  --data <path>           Training data (jsonl)
  --model <name>          gpt2-small / gpt2-medium / gpt2-large
  --epochs <n>            Number of training epochs (default: 3)
  --batch-size <n>        Batch size (default: 8)
  --lr <f>                Learning rate (default: 5e-5)
  --max-len <n>           Max sequence length (default: 1024)
  --output <dir>          Output directory (default: ./checkpoints)
  --device <dev>          cuda / cpu (default: auto)
```

示例：
```bash
./build/sft_train \
    --data data/instruction.jsonl \
    --model gpt2-medium \
    --epochs 3 \
    --batch-size 4 \
    --output ./checkpoints/gpt2-medium-sft
```

---

## chat：单轮推理

```bash
$ ./build/chat --model ./checkpoints/gpt2-medium-sft
[sft_chat] loaded gpt2-medium-sft (355M params)
[sft_chat] device: cuda:0
[sft_chat] tokenizer: gpt2
> 你好，请介绍一下你自己
我是基于 GPT-2 medium 微调的指令助手...

> 用 Python 写个 Hello World
print("Hello, World!")

> 退出
[sft_chat] goodbye!
```

---

## sft_serve：HTTP 服务

```cpp
// serve/sft_serve.cpp
#include <crow.h>  // 单文件 HTTP 库

int main() {
    GPTModel model(load_config("gpt2-medium-sft"));
    BPETokenizer tok;
    model.load_weights("./checkpoints/gpt2-medium-sft/model.safetensors");
    
    crow::SimpleApp app;
    
    // 健康检查
    CROW_ROUTE(app, "/health")
    ([](const crow::request&) {
        return crow::response(200, "{\"status\":\"ok\"}");
    });
    
    // OpenAI 兼容的 /v1/chat
    CROW_ROUTE(app, "/v1/chat").methods("POST"_method)
    ([&](const crow::request& req) {
        auto body = crow::json::load(req.body);
        std::string user_msg = body["messages"][0]["content"].s();
        
        // 1. 格式化 prompt
        auto prompt = format_alpaca(user_msg, "");
        auto ids = tok.encode(prompt);
        
        // 2. 推理
        auto output_ids = model.generate(ids, /*max_new=*/200);
        auto response = tok.decode(output_ids);
        
        // 3. 返回 JSON
        crow::json::wvalue result;
        result["content"] = response;
        return crow::response(result);
    });
    
    app.port(8000).multithreaded().run();
    return 0;
}
```

启动：
```bash
./build/sft_serve --model ./checkpoints/gpt2-medium-sft --port 8000
```

调用：
```bash
$ curl http://localhost:8000/health
{"status":"ok"}

$ curl -X POST http://localhost:8000/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"你好"}]}'

{"content":"你好！我是基于 GPT-2 的指令助手..."}
```

---

## 部署形态：单二进制 + 模型文件

```bash
# 最终产物
ls -lh deploy/
total 1.5G
-rwxr-xr-x  sft_serve           4.2M  ← 单二进制
-rw-r--r--  model.safetensors   1.4G  ← 权重
-rw-r--r--  encoder.json         1M
-rw-r--r--  vocab.bpe          450K

# 部署（拷到任何 Linux x86-64 机器）
scp deploy/* user@server:/app/
ssh user@server "/app/sft_serve --model /app/model.safetensors --port 8000"
```

**零依赖**：不需要 Python，不需要 PyTorch，只需要 C++ 运行时。

---

## 资源占用

| 资源 | 占用 |
|---|---|
| 启动时间 | < 200 ms |
| 内存（空闲） | ~1.5 GB |
| 内存（单请求） | +200 MB |
| GPU 显存（medium） | ~2 GB |
| 吞吐量 | ~10 req/s（batch=1） |
| 延迟（首 token） | ~150 ms |
| 延迟（200 token） | ~3 s |

---

## 三个项目的复用关系

```
gpt-sft/core (静态库)
       │
       ├── gpt-bash     （数据：nl2bash）
       ├── gpt-toolcall （数据：Hermes）
       └── gpt-sft/train/serve（CLI + HTTP）
```

**核心代码 0 重复**，3 个项目各只写自己的业务逻辑。

---

## 下一步

**X.4 总结**：从 124M 到 355M，下一步往哪走

---

## 互动

你最想要的部署形态是：
- 单二进制（最简）
- Docker 镜像（隔离）
- K8s + Helm（规模化）

评论告诉我。
