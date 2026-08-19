= gpt-sft 部署说明

独立项目 `gpt-sft/` 把最终模型（GPT-2 medium 指令微调）打包为可对外服务的
训练/推理分离工程。本文说明部署步骤。

== 1. 编译

[source,shell]
----
cd gpt-sft
./scripts/build.sh            # 输出 build/train/sft_train, build/serve/sft_serve, build/serve/sft_chat
----

== 2. 数据与权重

数据/权重由根项目脚本提供（见根 README）：

[source,shell]
----
./scripts/download-data.sh      # instruction-data.json
./scripts/download-weights.sh medium   # gpt2-medium.safetensors (1.52 GB)
----

== 3. 训练

[source,shell]
----
./scripts/run.sh train --epochs 2 --out data/gpt2-medium-sft.pth
----

训练约 3-4 分钟（RTX 4080）。产物 `data/gpt2-medium-sft.pth`（约 1.7 GB）。

== 4. 启动推理服务

[source,shell]
----
./scripts/run.sh serve --model data/gpt2-medium-sft.pth --port 8080
----

== 5. 调用

[source,shell]
----
curl --noproxy '*' localhost:8080/health
curl --noproxy '*' -X POST localhost:8080/v1/chat \
  -d '{"instruction":"Convert the active sentence to passive: The chef cooks the meal every day."}'
# {"response":"The meal is prepared by the chef."}
----

注意：若 shell 设置了 `ALL_PROXY` 等代理变量，调用本地服务请加 `--noproxy '*'`。

== 6. 生产部署（systemd）

见项目 README 中的 systemd 单元示例；如需对外开放，建议前置 Nginx 反代 +
鉴权（API Key），并将服务绑定到内网地址。

== 架构说明

[source,text]
----
train/  sft_train  加载预训练 safetensors -> 指令微调 -> 保存 .pth
                    │
serve/  sft_serve  torch::load(.pth) -> HTTP 服务（/health /v1/chat /v1/generate）
        sft_chat   torch::load(.pth) -> 命令行对话
                    │
core/   gpt_sft_core.*  统一 API：make_model / load_pretrained /
                        train_instruction / generate_response
        gpt.h  attention.h  bpe_tokenizer.*  safetensors.*  instruction.*
----

训练与推理解耦：训练产物只有 `.pth` 权重，推理服务不依赖训练代码，可独立部署。
