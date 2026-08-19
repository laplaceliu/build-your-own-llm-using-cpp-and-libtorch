# gpt-sft —— GPT-2 指令微调独立项目（训练 / 推理分离）

基于《Build Your Own LLM Using C++ and LibTorch》第 7 章的完整独立工程：
把最终模型（GPT-2 medium 指令微调）做成可对外提供服务的项目。

```
gpt-sft/
├── CMakeLists.txt            # 顶层：core 静态库 + train + serve
├── core/                     # 共享核心（模型/分词器/权重加载/指令数据）
│   ├── gpt_sft_core.h/.cpp   # 统一 API 门面（模型工厂/训练/推理）
│   ├── gpt.h / attention.h   # GPT 架构（第 4/3 章）
│   ├── bpe_tokenizer.*       # GPT-2 BPE 分词器（第 2 章）
│   ├── safetensors.*         # HF 权重解析（第 5 章）
│   ├── instruction.h         # 指令数据/批处理（第 7 章）
│   └── training.h / dataloader.h
├── train/                    # 训练模块（指令微调 CLI）
│   └── src/sft_train.cpp
├── serve/                    # 推理模块（HTTP 服务 + CLI 对话）
│   └── src/sft_serve.cpp     # REST API 服务
│   └── src/sft_chat.cpp      # 命令行交互推理
├── data/                     # 数据与产物（模型 .pth 不入库）
├── scripts/                  # build.sh / run.sh
└── docs/                     # 部署说明
```

## 快速开始

```bash
# 1. 准备数据（复用根项目脚本，需先进入根项目）
cd .. && ./scripts/download-data.sh && ./scripts/download-weights.sh medium && cd gpt-sft

# 2. 编译
./scripts/build.sh

# 3. 训练（指令微调，输出 .pth）
./scripts/run.sh train --epochs 2 --out data/gpt2-medium-sft.pth

# 4. 启动 HTTP 推理服务
./scripts/run.sh serve --model data/gpt2-medium-sft.pth --port 8080

# 5. 调用
curl --noproxy '*' localhost:8080/health
curl --noproxy '*' -X POST localhost:8080/v1/chat \
  -d '{"instruction":"Convert the active sentence to passive: The chef cooks the meal every day."}'
# -> {"response":"The meal is prepared by the chef."}

# 6. 命令行对话
./scripts/run.sh chat --model data/gpt2-medium-sft.pth
```

## 训练（train/sft_train）

```
sft_train [选项]
  --data <json>       指令数据集（默认 data/instruction-data.json）
  --weights <st>      预训练权重（默认 data/gpt2-medium.safetensors，可复用根项目下载的）
  --size <small|medium|large|xl>  模型规模（默认 medium）
  --epochs <n>        轮数（默认 2）        --batch <n> 批大小（默认 8）
  --lr <f>            学习率（默认 5e-5）    --out <path> 输出 .pth（默认 data/gpt2-medium-sft.pth）
  --no-cuda           强制 CPU
```

训练数据格式（JSON 数组）：
```json
[{"instruction": "...", "input": "...", "output": "..."}, ...]
```

## 推理服务（serve/sft_serve）

```
sft_serve --model <pth> [--port 8080] [--size small|medium|large|xl] [--no-cuda]
```

### REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 `{"status":"ok","model":"gpt-sft"}` |
| `/v1/chat` | POST | 指令推理 `{"instruction","input"?,"max_tokens"?}` |
| `/v1/generate` | POST | OpenAI 风格 `{"messages":[{"role":"user","content":...}],"max_tokens"}` |

响应：`{"response":"...", "instruction":"...", "input":"..."}`

### 部署说明（systemd）

```ini
# /etc/systemd/system/gpt-sft.service
[Unit]
Description=GPT-SFT inference service
After=network.target

[Service]
User=maigi
WorkingDirectory=/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/gpt-sft
Environment=LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib
ExecStart=/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/gpt-sft/build/serve/sft_serve --model /home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/gpt-sft/data/gpt2-medium-sft.pth --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

## 命令行推理（serve/sft_chat）

```bash
# 单条指令
./scripts/run.sh chat --model data/gpt2-medium-sft.pth "指令" ["输入"]
# 交互模式（Ctrl-D 退出）
./scripts/run.sh chat --model data/gpt2-medium-sft.pth
```

## 说明

* **模型规模**：默认 medium（355M，1024/16/24）。`--size small` 需配 small 权重，
  `.pth` 与 `--size` 必须匹配（结构不同不能混用）。
* **GPU**：自动检测 CUDA（RTX 4080 等），训练/推理均在 GPU 上；`--no-cuda` 强制 CPU。
* **权重一致性**：TF32 已禁用，GPU 数值与书一致。
* **依赖**：LibTorch 2.13（`/opt/libtorch`）、CUDA 13.0、GCC 11、nlohmann-json。
* **数据**：`instruction-data.json` 由根项目 `./scripts/download-data.sh` 提供。
