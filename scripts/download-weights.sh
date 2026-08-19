#!/usr/bin/env bash
# scripts/download-weights.sh
# 下载各章所需的预训练大模型权重（数百 MB，.gitignore 已排除）
#
#   gpt2  small   (355MB)  -> 第 5 章权重加载 / 第 6 章分类微调
#   gpt2  medium  (1.5GB)  -> 第 7 章指令微调
#
# 源: hf-mirror.com（无代理可达且快），失败自动回退 huggingface.co
#
# 用法: ./scripts/download-weights.sh [small|medium|all]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHAT="${1:-all}"
CH2="${PROJECT_ROOT}/chapters/chapter02_text_data/data"
CH7="${PROJECT_ROOT}/chapters/chapter07_instruction_tuning/data"

echo "=== 下载 GPT-2 预训练权重 ==="

dl() {
  local src="$1" out="$2" desc="$3"
  if [ -f "${out}" ]; then
    echo "  [跳过] ${desc} 已存在: ${out}"
    return 0
  fi
  echo "  [下载] ${desc}（约 $4）"
  mkdir -p "$(dirname "${out}")"
  if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY curl -fsSL --connect-timeout 30 \
       -o "${out}" "https://hf-mirror.com/${src}" 2>/dev/null; then
    echo "  [完成] hf-mirror"
  elif curl -fsSL --connect-timeout 30 -L -o "${out}" "https://huggingface.co/${src}" 2>/dev/null; then
    echo "  [完成] huggingface.co"
  elif curl -fsSL --connect-timeout 30 -x http://localhost:8888 -L -o "${out}" \
       "https://huggingface.co/${src}" 2>/dev/null; then
    echo "  [完成] huggingface.co (代理)"
  else
    echo "  [失败] ${src}"
    rm -f "${out}"
    return 1
  fi
}

case "${WHAT}" in
  small|gpt2)
    dl "gpt2/resolve/main/model.safetensors" "${CH2}/gpt2-model.hf.safetensors" \
       "GPT-2 small (第 5/6 章)" "548 MB"
    ;;
  medium|gpt2-medium)
    dl "gpt2-medium/resolve/main/model.safetensors" "${CH7}/gpt2-medium.safetensors" \
       "GPT-2 medium (第 7 章)" "1.52 GB"
    ;;
  all|"")
    dl "gpt2/resolve/main/model.safetensors" "${CH2}/gpt2-model.hf.safetensors" \
       "GPT-2 small (第 5/6 章)" "548 MB"
    dl "gpt2-medium/resolve/main/model.safetensors" "${CH7}/gpt2-medium.safetensors" \
       "GPT-2 medium (第 7 章)" "1.52 GB"
    ;;
  *)
    echo "用法: $0 [small|medium|all]"; exit 1 ;;
esac

echo ""
echo "=== 权重就绪 ==="
