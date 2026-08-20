#!/usr/bin/env bash
# gpt-toolcall/scripts/download-data.sh
#
# 下载 NousResearch/hermes-function-calling-v1 (JSON 格式) 到 data/raw/hermes.jsonl
# 原始数据 ~26 MB，含 12,565 条多轮工具调用对话。
#
# 镜像（按 HF_MIRROR 同 gpt-bash 约定）：
#   hf-mirror (默认) / huggingface / ghproxy
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

MIRROR="${HF_MIRROR:-hf-mirror}"
case "${MIRROR}" in
  hf-mirror)   BASE="https://hf-mirror.com" ;;
  huggingface) BASE="https://huggingface.co" ;;
  *)
    echo "[警告] 未知镜像 '${MIRROR}'，回退到 hf-mirror" >&2
    BASE="https://hf-mirror.com" ;;
esac

# NousResearch/hermes-function-calling-v1 main 分支根目录下的多轮 function calling 数据集
# 实测文件清单（2026-08 hf-mirror 抓取）：
#   func-calling.json         20.6 MB   多轮 + 单轮 function calling 主数据集
#   func-calling-singleturn.json 17.8 MB   纯单轮
#   json-mode-agentic.json    5.02 MB   agentic JSON mode（不是 function calling 格式，跳过）
#   json-mode-singleturn.json 2.95 MB   JSON mode 单轮（跳过）
#   glaive-function-calling-5k.json 20.4 MB   Glaive 语料（多轮，跳过避免与 func-calling 重复）
#
# 这里默认拉 func-calling.json（多 + 单轮混合，最具代表性）
URL="${BASE}/datasets/NousResearch/hermes-function-calling-v1/resolve/main/func-calling.json"
DEST="${RAW}/hermes-func-calling.json"
MIN_SIZE=$((15 * 1024 * 1024))   # 15 MiB（实测 20.6 MB）

# 跳过
if [[ -s "${DEST}" ]] && [[ $(wc -c < "${DEST}") -ge ${MIN_SIZE} ]]; then
  echo "[跳过] ${DEST} ($(wc -c < "${DEST}") 字节)"
  exit 0
fi

echo "[镜像] ${MIRROR} → ${BASE}"
echo "[下载] ${URL}"
PART="${DEST}.part"
curl -fSL -C - --connect-timeout 30 --max-time 1800 \
     --retry 5 --retry-delay 5 --retry-all-errors \
     "${URL}" -o "${PART}"

got=$(wc -c < "${PART}")
if [[ ${got} -lt ${MIN_SIZE} ]]; then
  echo "[错误] 下载不完整: ${got} 字节 (期望 ≥ ${MIN_SIZE})" >&2
  echo "[提示] 重跑脚本可断点续传；或 HF_MIRROR=huggingface $0" >&2
  exit 1
fi
mv "${PART}" "${DEST}"

# Hermes func-calling.json 是 JSON 数组（单行），用 python 统计
n=$(python3 -c "import json; print(len(json.load(open('${DEST}'))))" 2>/dev/null || echo "未知")
echo "[完成] ${DEST}"
echo "       大小 $(wc -c < "${DEST}") 字节 (~$(awk "BEGIN{printf \"%.1f\", $(wc -c < "${DEST}")/1024/1024}") MiB)"
echo "       ${n} 条对话（python json 解析）"