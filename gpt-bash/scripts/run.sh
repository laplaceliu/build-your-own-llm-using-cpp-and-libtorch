#!/usr/bin/env bash
# gpt-bash/scripts/run.sh
# 一站式：训练 + 评测 + 聊天
#
# 用法:
#   ./scripts/run.sh train   --size small --epochs 5 --batch 8
#   ./scripts/run.sh eval    --model data/bash-sft-small.pth --size small --n 200
#   ./scripts/run.sh chat    --model data/bash-sft-small.pth --size small
#
# 需要先：
#   ./scripts/download-data.sh         # ~2 MB
#   python3 scripts/convert_data.py    # ~12.5k Alpaca JSON
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cmd="${1:-help}"; shift || true

case "${cmd}" in
  help|-h)
    cat <<EOF
用法:
  ${0} download   # 下载原始数据
  ${0} convert    # 转 Alpaca JSON
  ${0} train      # SFT 训练（参数透传给 gpt-sft/scripts/run.sh）
  ${0} eval       # 评测（参数透传给 eval/bash_eval.py）
  ${0} chat       # 交互/单条（参数透传给 build/bash_chat）
EOF
    exit 0
    ;;
  download)
    bash "${SELF_DIR}/download-data.sh"
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
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/libtorch/lib" \
      exec build/bash_chat "$@"
    ;;
  *)
    echo "[错误] 未知命令: ${cmd}"; exit 1;;
esac
