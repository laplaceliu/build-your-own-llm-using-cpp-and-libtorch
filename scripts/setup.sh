#!/usr/bin/env bash
# scripts/setup.sh
# 一键准备工作：下载数据 -> 下载权重 -> 启动 Ollama -> 编译
#
# 用法: ./scripts/setup.sh [--data] [--weights] [--ollama] [--build] [--all]
#   --data      下载各章数据集（词表/文本/Spam/指令数据）
#   --weights   下载 GPT-2 small + medium 权重（约 2 GB）
#   --ollama    启动 Ollama 服务并拉取 llama3（约 4.7 GB）
#   --build     配置并编译全部章节
#   --all       上述全部（默认）
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DO_DATA=0 DO_WEIGHTS=0 DO_OLLAMA=0 DO_BUILD=0
if [ $# -eq 0 ] || [ "${1:-}" = "--all" ]; then
  DO_DATA=1 DO_WEIGHTS=1 DO_OLLAMA=1 DO_BUILD=1
else
  for arg in "$@"; do
    case "${arg}" in
      --data) DO_DATA=1 ;;
      --weights) DO_WEIGHTS=1 ;;
      --ollama) DO_OLLAMA=1 ;;
      --build) DO_BUILD=1 ;;
      *) echo "未知参数: ${arg}"; echo "用法: $0 [--data] [--weights] [--ollama] [--build] [--all]"; exit 1 ;;
    esac
  done
fi

echo "=== 项目准备工作（${PROJECT_ROOT}）==="

[ "${DO_DATA}" = 1 ] && { echo "\n>>> [1/4] 下载数据集"; "${SCRIPT_DIR}/download-data.sh"; }
[ "${DO_WEIGHTS}" = 1 ] && { echo "\n>>> [2/4] 下载权重"; "${SCRIPT_DIR}/download-weights.sh"; }
[ "${DO_OLLAMA}" = 1 ] && { echo "\n>>> [3/4] 启动 Ollama"; "${SCRIPT_DIR}/start-ollama.sh"; }
[ "${DO_BUILD}" = 1 ] && { echo "\n>>> [4/4] 编译"; "${SCRIPT_DIR}/build.sh"; }

echo ""
echo "=== 准备完成 ==="
