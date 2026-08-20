#!/usr/bin/env bash
# gpt-bash/scripts/run.sh
# 一站式：下载 + 转换 + 训练 + 评测 + 聊天 + smoke
#
# 用法:
#   ./scripts/run.sh download              # 下载原始数据 (~2 MB)
#   ./scripts/run.sh download-weights      # 下载 gpt2-medium 权重 (~1.4 GiB)
#   ./scripts/run.sh convert               # 转 Alpaca JSON
#   ./scripts/run.sh train   --size small   # SFT 训练
#   ./scripts/run.sh eval    --model ...    # 评测
#   ./scripts/run.sh chat    --model ...    # 交互/单条
#   ./scripts/run.sh smoke                  # 用 smoke 数据快速验证 chat
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cmd="${1:-help}"; shift || true

case "${cmd}" in
  help|-h)
    cat <<EOF
用法:
  ${0} download          # 下载原始数据 (~2 MB)
  ${0} download-weights  # 下载 gpt2-medium 权重 (~1.4 GiB, 默认 hf-mirror)
  ${0} convert           # 转 Alpaca JSON
  ${0} train             # SFT 训练（参数透传给 gpt-sft/scripts/run.sh）
  ${0} eval              # 评测（参数透传给 eval/bash_eval.py）
  ${0} chat              # 交互/单条（参数透传给 build/bash_chat）
  ${0} smoke             # 用 smoke 数据快速验证 chat 二进制（不需训练）
EOF
    exit 0
    ;;
  download)
    bash "${SELF_DIR}/download-data.sh"
    ;;
  download-weights)
    bash "${SELF_DIR}/download-weights.sh"
    ;;
  convert)
    ${PYTHON} "${SELF_DIR}/convert_data.py" --shuffle \
        --out "${ROOT}/data/bash-instruction-data.json" "$@"
    ;;
  train)
    cd "${ROOT}/../gpt-sft"
    if [[ ! -x ./scripts/run.sh ]]; then
      echo "[错误] 找不到 gpt-sft/scripts/run.sh"; exit 1
    fi
    exec ./scripts/run.sh train "$@"
    ;;
  eval)
    cd "${ROOT}"
    DEFAULT_DATA="${ROOT}/data/bash-instruction-data.json"
    DEFAULT_BIN="${ROOT}/build/bash_chat"
    EXTRA=()
    # 自动默认：data_path 与 binary（允许覆盖）
    [[ "$*" != *"--data"*    ]] && EXTRA+=(--data "${DEFAULT_DATA}")
    [[ "$*" != *"--binary"*  ]] && EXTRA+=(--binary "${DEFAULT_BIN}")
    ${PYTHON} "${ROOT}/eval/bash_eval.py" "${EXTRA[@]}" "$@"
    ;;
  chat)
    cd "${ROOT}"
    if [[ ! -x build/bash_chat ]]; then
      echo "[提示] 第一次运行？构建: cmake -B build && cmake --build build -j"
      exit 1
    fi
    TORCH_LIB="${TORCH_ROOT:-/opt/libtorch}/lib"
    LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}" \
      exec build/bash_chat "$@"
    ;;
  smoke)
    # 用 12 条 smoke 数据快速验证 chat 二进制：不需要训练权重
    # 模型是随机初始化的 gpt2-small（vocab=50257，OK
    cd "${ROOT}"
    if [[ ! -x build/bash_chat ]]; then
      echo "[提示] 第一次运行？构建: cmake -B build && cmake --build build -j"
      exit 1
    fi
    SMOKE_DATA="${ROOT}/data/smoke/bash-instruction-data-tiny.json"
    if [[ ! -s "${SMOKE_DATA}" ]]; then
      echo "[错误] 缺少 ${SMOKE_DATA}"; exit 1
    fi
    echo "[smoke] 用 12 条 smoke 数据验证 chat 二进制（无训练权重，期望输出接近乱码即 OK）"
    ${PYTHON} "${ROOT}/eval/bash_eval.py" \
        --data "${SMOKE_DATA}" \
        --model "${ROOT}/data/bash-sft-medium.pth" \
        --binary "${ROOT}/build/bash_chat" \
        --size medium \
        --n 12 \
        --out "${ROOT}/logs/smoke-eval.json" \
        --skip-exec
    echo "[smoke] 输出: logs/smoke-eval.json"
    ;;
  *)
    echo "[错误] 未知命令: ${cmd}"; exit 1;;
esac