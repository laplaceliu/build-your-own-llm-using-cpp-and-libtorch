#!/usr/bin/env bash
# gpt-sft/scripts/run.sh
# 设置运行环境并执行 训练 / 推理服务 / CLI 推理
#
# 用法:
#   ./scripts/run.sh train [选项...]     # sft_train 指令微调
#   ./scripts/run.sh serve [选项...]     # sft_serve  HTTP 推理服务
#   ./scripts/run.sh chat [选项...]      # sft_chat  命令行推理
#
# 环境: LD_LIBRARY_PATH 自动包含 libtorch + CUDA；lib 路径可用环境变量覆盖。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIBTORCH="${TORCH_ROOT:-/opt/libtorch}"
CUDA_DIR="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_DIR}/lib64:${CUDA_DIR}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CMD="${1:-}"
[ -z "${CMD}" ] && { echo "用法: $0 [train|serve|chat] [选项...]"; exit 1; }
shift

case "${CMD}" in
  train) EXE="${ROOT}/build/train/sft_train" ;;
  serve) EXE="${ROOT}/build/serve/sft_serve" ;;
  chat)  EXE="${ROOT}/build/serve/sft_chat" ;;
  *) echo "未知命令: ${CMD}"; exit 1 ;;
esac

[ -x "${EXE}" ] || { echo "未找到 ${EXE}，请先运行 ./scripts/build.sh"; exit 1; }
echo "[运行] ${EXE} $*"
exec "${EXE}" "$@"
