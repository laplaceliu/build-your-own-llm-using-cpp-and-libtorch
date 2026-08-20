# gpt-bash/data 目录说明

## 数据来源

* `raw/all.nl` — nl2bash 原始自然语言描述（每一行一条）
* `raw/all.cm` — 与 nl 一一对应的 bash 命令（每行一条，去过重 + bashlint 规范化）
* 来源仓库: https://github.com/TellinaTool/nl2bash/tree/master/data/bash

> 该原始版本共 **12,607** 条样本，等价于 ATLAS-Lab nl2bash-shell 的
> 训练集语义。原始 nl2bash-shell 仓库未镜像到 HuggingFace 公开数据
> 视图，但同样基于 TellinaTool 的语料。

## 处理后的训练数据

* `bash-instruction-data.json` — `scripts/convert_data.py` 产出，
  与 `chapters/chapter07_instruction_tuning/data/instruction-data.json`
  同一种 Alpaca 三元组格式（`instruction / input / output`）。
* 该 JSON 可以直接喂给 `gpt-sft/train/sft_train`：

  ```bash
  cd gpt-sft && ./scripts/run.sh train \
      --data_path ../gpt-bash/data/bash-instruction-data.json \
      --out ../gpt-bash/data/bash-sft.pth \
      --size small --epochs 5
  ```

## 用作评测

评测不需要重复训练，但这里也提供：

* 当前 `convert_data.py` **不切分 train/dev**，整文件用作训练集。
  评测时由 `eval/bash_eval.py` 自行随机抽样（默认 200 条）或跑全量（`--all`）。
  如需固定的 dev 集，可自行用 `jq` / `python -c '...'` 从 JSON 里切。

* 评测方法参见 `eval/bash_eval.py`：每条样本把模型生成的 bash 命令
  丢进沙箱（`subprocess` + `timeout` + **命令白名单**）执行，
  reward = `rc == 0 && stdout 命中期望输出（必要时 mask UID / PID）`。
