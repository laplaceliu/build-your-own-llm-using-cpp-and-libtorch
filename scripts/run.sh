#!/usr/bin/env bash
# scripts/run.sh
# 设置运行环境并运行指定章节的程序
#
# 用法: ./scripts/run.sh <章节名> [程序参数...]
#
# 章节名（目录名即可）:
#   chapter01_hello_torch
#   chapter02_text_data
#   chapter03_attention
#   chapter04_gpt
#   chapter05_pretraining       [可选: epochs] 如 ./scripts/run.sh chapter05_pretraining 10
#   chapter06_finetuning
#   chapter07_instruction_tuning [可选: epochs]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CHAPTER="${1:-}"
if [ -z "${CHAPTER}" ]; then
  echo "用法: $0 <章节名> [程序参数...]"
  echo "可用章节:"; ls "${PROJECT_ROOT}/chapters" | grep chapter || true
  exit 1
fi

# 参数可能用完整目录名或去前缀名
EXE=""
for c in "${CHAPTER}" "chapter${CHAPTER}" "${CHAPTER#chapter_}"; do
  p="${PROJECT_ROOT}/build/chapters/${c}"
  if [ -f "${p}" ]; then EXE="${p}"; break; fi
  if [ -d "${p}" ] && [ -x "${p}/${c}" ]; then EXE="${p}/${c}"; break; fi
done
if [ -z "${EXE}" ]; then
  echo "[错误] 未找到可执行程序: ${CHAPTER}（请先运行 ./scripts/build.sh）"
  exit 1
fi

shift
echo "[运行] ${EXE} $*"
exec "${EXE}" "$@"
