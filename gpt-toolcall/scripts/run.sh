#!/usr/bin/env bash
# gpt-toolcall/scripts/run.sh
# 一站式：下载 + 转换 + 训练 + smoke + chat(dispatch)
#
# 用法:
#   ./scripts/run.sh download              # 下载 Hermes 原始 (~26 MB)
#   ./scripts/run.sh download-weights      # 下载 gpt2-medium (~1.4 GiB)
#   ./scripts/run.sh convert               # 转 Alpaca JSON
#   ./scripts/run.sh train                 # SFT 训练（参数透传给 gpt-sft/scripts/run.sh）
#   ./scripts/run.sh build                 # 编译 tool_chat 二进制
#   ./scripts/run.sh chat --query "..."    # 单轮 Function Calling dispatcher
#   ./scripts/run.sh smoke                 # 用 smoke 数据快速验证构建
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cmd="${1:-help}"; shift || true

case "${cmd}" in
  help|-h)
    cat <<EOF
用法:
  ${0} download          # 下载 Hermes 原始数据 (~26 MB)
  ${0} download-weights  # 下载 gpt2-medium 权重 (~1.4 GiB, 默认 hf-mirror)
  ${0} convert           # Hermes → Alpaca JSON
  ${0} train             # SFT 训练（参数透传给 gpt-sft/scripts/run.sh）
  ${0} build             # cmake 构建 tool_chat
  ${0} chat --query ...  # 多轮 dispatcher（调 dispatch.py）
  ${0} smoke             # 用 smoke 数据快速验证 chat 二进制
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
    ${PYTHON} "${SELF_DIR}/convert_data.py" \
        --in "${ROOT}/data/raw/hermes-func-calling.json" \
        --out "${ROOT}/data/toolcall-data.json" \
        --shuffle "$@"
    ;;
  train)
    cd "${ROOT}/../gpt-sft"
    if [[ ! -x ./scripts/run.sh ]]; then
      echo "[错误] 找不到 gpt-sft/scripts/run.sh"; exit 1
    fi
    exec ./scripts/run.sh train "$@"
    ;;
  build)
    cd "${ROOT}"
    if [[ ! -f CMakeLists.txt ]]; then
      echo "[错误] 缺 CMakeLists.txt"; exit 1
    fi
    if [[ ! -d build ]]; then
      cmake -B build -DCMAKE_BUILD_TYPE=Release
    fi
    cmake --build build -j --target tool_chat
    echo "[完成] $([[ -x build/tool_chat ]] && echo OK || echo FAIL): build/tool_chat"
    ;;
  chat)
    cd "${ROOT}"
    if [[ ! -x build/tool_chat ]]; then
      echo "[提示] 第一次运行？构建: ./scripts/run.sh build"; exit 1
    fi
    DEFAULT_MODEL="${ROOT}/data/toolcall-sft-medium.pth"
    [[ "$*" != *"--model"* ]] && set -- --model "${DEFAULT_MODEL}" "$@"
    [[ "$*" != *"--binary"* ]] && set -- --binary "${ROOT}/build/tool_chat" "$@"
    [[ "$*" != *"--size"*  ]] && set -- --size medium "$@"
    exec ${PYTHON} "${ROOT}/chat/dispatch.py" "$@"
    ;;
  smoke)
    cd "${ROOT}"
    if [[ ! -x build/tool_chat ]]; then
      echo "[提示] 第一次运行？构建: ./scripts/run.sh build"; exit 1
    fi
    SMOKE_DATA="${ROOT}/data/smoke/toolcall-smoke.json"
    if [[ ! -s "${SMOKE_DATA}" ]]; then
      echo "[错误] 缺少 ${SMOKE_DATA}"; exit 1
    fi
    # smoke 验证：纯跑 prompt + 取生成结果，不要求能调通工具
    echo "[smoke] 用 12 条手写 smoke 数据验证 chat 二进制"
    # 优先用训练好的 .pth；没有则回落到未训练的 medium.safetensors（期望输出乱码）
    MODEL="${ROOT}/data/toolcall-sft-medium.pth"
    if [[ ! -s "${MODEL}" ]]; then
      MODEL="${ROOT}/../chapters/chapter07_instruction_tuning/data/gpt2-medium.safetensors"
      if [[ -s "${MODEL}" ]]; then
        echo "[注意] 未训练权重，使用 medium.safetensors（预期输出接近乱码即 OK）"
      fi
    fi
    # 智能选择: 训练好的 .pth 走 GPU（快），否则 --no-cuda
    if [[ "${MODEL}" == *"toolcall-sft"* ]]; then
      DEVICE_ARGS_STR="[]"
    else
      DEVICE_ARGS_STR="['--no-cuda']"
    fi
    # smoke 数据是 JSON 数组（不是 JSONL），用 python json.load 一次读出
    ${PYTHON} - <<EOF
import json, subprocess, sys, os
data = json.load(open("${SMOKE_DATA}"))
binary = "${ROOT}/build/tool_chat"
model = "${MODEL}"
device_args = ${DEVICE_ARGS_STR}   # 转成 python list
for i, sample in enumerate(data):
    instr = sample["instruction"]
    expected = sample["output"]
    with open("/tmp/_smoke_prompt.txt", "w") as f:
        f.write(instr)
    print(f"\n--- sample {i+1}/{len(data)} ---")
    print(f"prompt 末 80 字: ...{instr[-80:]!r}")
    print(f"期望输出:        {expected!r}")
    try:
        proc = subprocess.run(
            [binary, "--model", model, "--size", "medium",
             "--prompt-file", "/tmp/_smoke_prompt.txt",
             "--max-new", "96"] + device_args + ["--json"],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            print(f"FAIL exit={proc.returncode} stderr={proc.stderr[:200]}")
            continue
        out = json.loads(proc.stdout)
        print(f"模型输出 ({len(out['text'])} 字, stopped_on={out['stopped_on']!r}, {out.get('elapsed_sec','?')}s):")
        print(f"  {out['text'][:200]!r}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (180s)")
    except Exception as e:
        print(f"FAIL: {e}")
EOF
    echo "[smoke] 验证完成"
    ;;
  *)
    echo "[错误] 未知命令: ${cmd}"; exit 1;;
esac