#!/usr/bin/env bash
# scripts/download-data.sh
# 下载各章所需的数据集/词表（不含大模型权重，权重见 download-weights.sh）
#
#   ch2: the-verdict.txt, encoder.json, vocab.bpe
#   ch6: SMS Spam Collection（zip 解压 -> SMSSpamCollection.tsv）
#   ch7: instruction-data.json（指令微调数据集）
#
# 用法: ./scripts/download-data.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "=== 下载各章数据集 ==="

# ---- 第 2 章：文本与 GPT-2 词表 ----
CH2="${PROJECT_ROOT}/chapters/chapter02_text_data/data"
download "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" \
         "${CH2}/the-verdict.txt" "第2章 the-verdict.txt"
download "https://huggingface.co/gpt2/resolve/main/vocab.json" \
         "${CH2}/encoder.json" "第2章 encoder.json(GPT-2 词表)"
download "https://huggingface.co/gpt2/resolve/main/merges.txt" \
         "${CH2}/vocab.bpe" "第2章 vocab.bpe(GPT-2 merge)"

# ---- 第 6 章：SMS Spam 数据集（zip 需解压重命名）----
CH6="${PROJECT_ROOT}/chapters/chapter06_finetuning/data"
if [ ! -f "${CH6}/SMSSpamCollection.tsv" ]; then
  download "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" \
           "${CH6}/sms_spam_collection.zip" "第6章 SMS Spam zip"
  if [ -f "${CH6}/sms_spam_collection.zip" ]; then
    echo "  [解压] sms_spam_collection.zip"
    (cd "${CH6}" && unzip -o sms_spam_collection.zip >/dev/null)
    mv -f "${CH6}/SMSSpamCollection" "${CH6}/SMSSpamCollection.tsv"
    echo "  [完成] SMSSpamCollection.tsv"
  fi
else
  echo "  [跳过] 第6章 SMSSpamCollection.tsv 已存在"
fi

# ---- 第 7 章：指令微调数据集 ----
CH7="${PROJECT_ROOT}/chapters/chapter07_instruction_tuning/data"
download "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json" \
         "${CH7}/instruction-data.json" "第7章 instruction-data.json"

echo ""
echo "=== 数据集就绪。大模型权重请运行: ./scripts/download-weights.sh ==="
