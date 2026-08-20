# gpt-bash：自然语言 → Bash 命令 SFT

把小型 GPT 模型（GPT-2 small/medium 124M/355M）通过指令微调，
学会把**自然语言描述**翻译成**对应的 bash 单行命令**。
相比数学推理任务，bash 命令**输出更短、可执行验证**，对小参
数模型更友好。

## 数据 + 权重 一站式下载

```bash
cd gpt-bash
./scripts/download-data.sh       # 拉 nl2bash 原始语料 (~2 MB)
./scripts/download-weights.sh    # 拉 gpt2-medium.safetensors (~1.4 GiB)
python3 scripts/convert_data.py --shuffle --out data/bash-instruction-data.json
```

镜像说明（所有下载脚本统一通过环境变量切换）：

| 变量 | 取值 | 说明 |
|---|---|---|
| `HF_MIRROR` | `hf-mirror` (默认) | 中国大陆加速，~10 MB/s |
| `HF_MIRROR` | `huggingface` | HF 官方，需海外网络 |
| `ALL_PROXY` | `socks5h://host:port` | 走代理（如 `localhost:30000`）|

例：`HF_MIRROR=huggingface ./scripts/download-weights.sh`

数据本身（GitHub raw）国内也能直连，慢一点；若遇 `connection timed out`，
可手动设置 `ALL_PROXY` 或下载后放到 `data/raw/all.{nl,cm}`。

## 训练

直接复用 `gpt-sft/train/sft_train` 二进制。把 `--data_path` 指向
bash 的 json，把 `--out` 指向 gpt-bash 的位置：

```bash
cd gpt-sft && ./scripts/build.sh build_train && cd ../..
./gpt-sft/scripts/run.sh train \
    --data_path gpt-bash/data/bash-instruction-data.json \
    --out_dir gpt-bash/data \
    --out_name bash-sft-small.pth \
    --size small --epochs 5 --batch_size 8 --lr 5e-5
```

更省事的一行（gpt-bash 内置 `train-background.sh`）：

```bash
./gpt-bash/scripts/run.sh train    # 透传到 gpt-sft/scripts/run.sh
SIZE=medium EPOCHS=3 BATCH=4 ./gpt-bash/scripts/train-background.sh
#  ← 自动检查权重，缺失时引导跑 download-weights.sh
```

> 因为 `convert_data.py` 输出和第 7 章的 `instruction-data.json`
> 同一种 Alpaca 格式，**不需要修改任何训练代码**。

## 推理（交互）

```bash
cd gpt-bash && cmake -B build && cmake --build build -j
./build/bash_chat --model data/bash-sft-small.pth --size small --no-cuda
# NL> list all the files in the current directory
# $ ls
```

可选 `--exec`：每条生成都通过白名单后在沙箱试跑（`timeout 3s`）。

## 评测

```bash
cd gpt-bash/eval
python bash_eval.py \
    --model ../data/bash-sft-small.pth --size small --n 200 \
    --out result-small.json
```

输出 `metrics = {em, ast_match, safety_pass, exec_safe}`。

## 与 gpt-sft / reasoning 的关系

| 模块 | 任务 | 输出长度 | 评测 |
|------|------|---------|------|
| `gpt-sft`  | 通用指令（Alpaca） | 中长 | LLM-as-judge (Ollama) |
| `reasoning`| 数学推理（GSM8K）| 长链 | exact_match 数字 |
| **`gpt-bash`**| bash 生成 | 短 | **执行 + AST + EM** |

gpt-bash 直接 `add_subdirectory(gpt-sft/core)` 复用训练基础设施。
