#!/usr/bin/env bash
# reasoning/scripts/run.sh
# 设置运行环境并执行 sft_train / rl_train / reasoning_chat / reasoning_serve / eval_math
#
# 用法:
#   ./scripts/run.sh sft_train [选项...]
#   ./scripts/run.sh rl_train [选项...]
#   ./scripts/run.sh reasoning_chat [选项...]
#   ./scripts/run.sh reasoning_serve [选项...]
#   ./scripts/run.sh eval_math [选项...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIBTORCH="${TORCH_ROOT:-/opt/libtorch}"
CUDA_DIR="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${CUDA_DIR}/lib64:${CUDA_DIR}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CMD="${1:-}"
[ -z "${CMD}" ] && {
  cat <<EOF
用法: $0 <command> [选项...]
  命令:
    sft_train       推理 SFT 训练
    rl_train        GRPO 强化学习训练
    reasoning_chat  CLI 推理（CoT/投票/束搜索/best-of-N）
    reasoning_serve HTTP 推理服务
    eval_math       GSM8K 评测
EOF
  exit 1
}
shift

case "${CMD}" in
  sft_train)       EXE="${ROOT}/build/train/sft_train" ;;
  rl_train)        EXE="${ROOT}/build/train/rl_train" ;;
  reasoning_chat)  EXE="${ROOT}/build/serve/reasoning_chat" ;;
  reasoning_serve) EXE="${ROOT}/build/serve/reasoning_serve" ;;
  eval_math)       EXE="${ROOT}/build/eval/eval_math" ;;
  *) echo "未知命令: ${CMD}"; exit 1 ;;
esac

[ -x "${EXE}" ] || { echo "未找到 ${EXE}，请先运行 ./scripts/build.sh"; exit 1; }
echo "[运行] ${EXE} $*"
exec "${EXE}" "$@"
