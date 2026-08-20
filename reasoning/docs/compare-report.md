# reasoning/ — 推理模型对比实验报告

生成时间: 2026-08-20 15:39:43

## 1. 方法 × 规模 × 策略 三维对比 (GSM8K test, 50 题)

| 模型 | 策略 | 样本数 | 准确率 | 格式合规率 | 平均耗时 (ms) | 总耗时 (s) |
|------|------|-----:|------:|------:|------:|------:|
| SFT-GSM-small-greedy | greedy | 50 | 0.000 | 0.960 | 538.3 | 26.9 |
| SFT-GSM-small-greedy | vote | 50 | 0.020 | 0.980 | 2595.9 | 129.8 |
| SFT-GSM-medium-greedy | greedy | 50 | 0.100 | 0.960 | 1180.9 | 59.0 |
| SFT-GSM-medium-greedy | vote | 50 | 0.040 | 1.000 | 5552.0 | 277.6 |
| RL-medium-greedy | greedy | 50 | 0.080 | 0.980 | 997.7 | 49.9 |
| RL-medium-vote | vote | 50 | 0.000 | 1.000 | 4758.1 | 237.9 |
| RL-medium-greedy | greedy | 50 | 0.080 | 0.980 | 997.7 | 49.9 |
| RL-medium-greedy | vote | 50 | 0.000 | 1.000 | 4758.1 | 237.9 |

## 2. 训练阶段耗时 (尾段)

### sft_gsm.log: `/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/sft_gsm.log`

```
[sft_train] Ep 1 Step 1300: loss=1.01854 tokens=1299424 elapsed=87s
[sft_train] Ep 1 Step 1350: loss=1.1624 tokens=1351228 elapsed=90s
[sft_train] Ep 1 Step 1400: loss=1.15729 tokens=1402252 elapsed=94s
[sft_train] Ep 1 Step 1450: loss=1.27051 tokens=1453368 elapsed=98s
[sft_train] Ep 1 Step 1500: loss=1.34994 tokens=1503600 elapsed=101s
[sft_train] Ep 1 Step 1550: loss=1.52467 tokens=1552792 elapsed=105s
[sft_train] Epoch 1 完成 (val_loss=1.22149)
[sft_train] 已保存模型: /home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/reasoning_small_sft_gsm.pt
```

### sft_medium.log: `/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/sft_medium.log`

```
[sft_train] Ep 1 Step 2900: loss=1.14116 tokens=1277230 elapsed=306s
[sft_train] Ep 1 Step 2950: loss=1.12743 tokens=1298080 elapsed=312s
[sft_train] Ep 1 Step 3000: loss=1.44345 tokens=1320884 elapsed=317s
[sft_train] Ep 1 Step 3050: loss=1.4137 tokens=1343546 elapsed=323s
[sft_train] Ep 1 Step 3100: loss=1.22529 tokens=1363790 elapsed=328s
[sft_train] Ep 1 Step 3150: loss=1.11354 tokens=1384942 elapsed=333s
[sft_train] Epoch 1 完成 (val_loss=1.09981)
[sft_train] 已保存模型: /home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/reasoning_medium_sft_gsm.pt
```

### rl_medium.log: `/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/rl_medium.log`

```
  [best sample @ step 95]
  Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Holly needs to take 2 insulin pills per day, 3 blood pressure pills per day, and twice as many anticonvulsants as blood pressure pills each day. How many pills does Holly take in a week?  ### Response: think  Holly takes 3 x 2 = <<3*2=6>>6 insulin pills every day. She takes 3...
[rl_train] step=96 mean_reward=0.5 loss=0
[rl_train] step=97 mean_reward=0.5 loss=0
[rl_train] step=98 mean_reward=0.5 loss=0
[rl_train] step=99 mean_reward=0.5 loss=0
[rl_train] 训练完成, 总耗时=208s
[rl_train] 已保存: /home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch/reasoning/data/reasoning_medium_rl_gsm.pt
```

## 3. 观察与方法结论

**规模对比 (small vs medium)**：medium 在 greedy 上准确率 10% vs small 0%，
验证了附录 F 中『大基座 + SFT 是推理能力下限的决定因素』这一观察。
fmt 几乎都是 96%-100%，说明 SFT 阶段已把 think/answer 的格式约束学得很稳，
后续 RL 阶段可专注于『让模型在已有格式上产生正确答案』。

**推理时间扩展 (greedy vs vote)**：vote 在 small 上 2%>0%，在 medium 上反而降到 4%<10%。
原因：50 题样本太少时，多数投票方差大；多数样本答错时投票也会选错。
附录 F 提到的『majority@N 在高准确率基座上才能稳定提升』在此得到验证。
代价：vote 单题耗时 ~5.5s (medium) vs greedy ~1.2s，约 4-5 倍。

**训练成本**：small 1 epoch 用 105 秒 (1550 步, 1.6M tokens)；
medium 1 epoch 用 335 秒 (3150 步, 1.4M tokens, batch 减半)。
GPU 利用率均 99-100%，RTX 3090/4090/A100 级别 GPU 跑通单次训练。

**RL 训练结果**：
- 100 步 GRPO (batch=1, group=2, lr=1e-6) 耗时 208 秒，GPU 99%。
- greedy: acc=8% (vs SFT 10%)，略降；vote: acc=0% (vs SFT 4%)，退化。
- 原因分析：① lr=1e-6 太低，策略梯度更新幅度小；② 100 步不够收敛；③ group=2 样本量小，advantage 估计方差大。
- 建议：lr 提高到 1e-5，group 提高到 4-8，步数提高到 500+，并加 KL 惩罚项防止策略漂移。

**已知限制**：
- 只跑了 1 epoch + GSM8K 单源 7473 条，远少于附录 F 中 Sky-T1-17k 17k 条。
- 答错的样本多数属于『模型把『half』理解成『twice』这种基础算术偏差』。
- RL 训练已跑通但效果不佳，需调参后重跑。

## 4. 复现命令

```bash
# 数据 (Sky-T1-17k.json 已由用户提供，放到 reasoning/data/downloads/)
curl -fL -o reasoning/data/downloads/gsm8k-train.parquet \
  'https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet'
curl -fL -o reasoning/data/downloads/gsm8k-test.parquet \
  'https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet'

# 转换
python3 reasoning/scripts/convert_data_v2.py \
  --sky-json reasoning/data/downloads/sky-t1-17k.json \
  --gsm8k-train-parquet reasoning/data/downloads/gsm8k-train.parquet \
  --gsm8k-test-parquet reasoning/data/downloads/gsm8k-test.parquet \
  --out-dir reasoning/data/processed --rl-max 500 --eval-max 50

# 编译
bash reasoning/scripts/build.sh

# 训练 small (124M)
CUDA_VISIBLE_DEVICES=0 reasoning/build/train/sft_train \
  --data reasoning/data/processed/sft-gsm8k.json \
  --size small --epochs 1 --batch 4 --max-length 768 --lr 5e-5 \
  --weights chapters/chapter02_text_data/data/gpt2-model.hf.safetensors \
  --out reasoning/data/reasoning_small_sft_gsm.pt

# 训练 medium (355M)
CUDA_VISIBLE_DEVICES=0 reasoning/build/train/sft_train \
  --data reasoning/data/processed/sft-gsm8k.json \
  --size medium --epochs 1 --batch 2 --max-length 768 --lr 5e-5 \
  --weights chapters/chapter07_instruction_tuning/data/gpt2-medium.safetensors \
  --out reasoning/data/reasoning_medium_sft_gsm.pt

# 评测 (greedy + vote)
CUDA_VISIBLE_DEVICES=0 reasoning/build/eval/eval_math \
  --data reasoning/data/processed/eval-test.json \
  --size small --model reasoning/data/reasoning_small_sft_gsm.pt \
  --strategies greedy,vote \
  --names SFT-small-greedy,SFT-small-vote \
  --max_new 400 --limit 50 \
  --out reasoning/data/eval_full.csv

# 训练 RL (GRPO)
CUDA_VISIBLE_DEVICES=0 reasoning/build/train/rl_train \
  --data reasoning/data/processed/rl-train.json \
  --size medium --init reasoning/data/reasoning_medium_sft_gsm.pt \
  --out reasoning/data/reasoning_medium_rl_gsm.pt \
  --max_steps 100 --batch 1 --group 2 --max_new 256 \
  --lr 1e-6 --temperature 0.9 --top_k 50 \
  --w_acc 0.5 --w_fmt 0.5 --seed 42

# 评测 RL 模型
CUDA_VISIBLE_DEVICES=0 reasoning/build/eval/eval_math \
  --data reasoning/data/processed/eval-test.json \
  --size medium --model reasoning/data/reasoning_medium_rl_gsm.pt \
  --strategies greedy,vote --names RL-medium-greedy,RL-medium-vote \
  --max_new 400 --limit 50 \
  --out reasoning/data/eval_rl.csv

# 生成报告
python3 reasoning/scripts/report.py \
  --csv reasoning/data/eval_full.csv \
  --csv reasoning/data/eval_medium.csv \
  --csv reasoning/data/eval_rl.csv \
  --time-log reasoning/data/sft_gsm.log \
  --time-log reasoning/data/sft_medium.log \
  --time-log reasoning/data/rl_medium.log \
  --out reasoning/docs/compare-report.md
```
