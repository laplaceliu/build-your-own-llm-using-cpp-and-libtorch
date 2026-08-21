#!/usr/bin/env bash
# gpt-toolcall/scripts/download-data.sh
#
# 下载 NousResearch/hermes-function-calling-v1 的 3 个 JSON 数据集到 data/raw/：
#   func-calling.json         20.6 MB   多轮 + 单轮混合主数据集
#   func-calling-singleturn.json 17.8 MB   纯单轮补充
#   glaive-function-calling-5k.json 20.4 MB   Glaive 语料 5k 条
# 总 ~58 MB / ~7 万条训练样本
#
# 镜像（与 gpt-bash 同约定）：
#   hf-mirror (默认, 国内) / huggingface (官方)
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

# (文件名, 最小字节数)
# 大小阈值取实测值 - 20% 容差
FILES=(
  "func-calling.json:16777216"               # 16 MiB (实测 20.6 MB)
  "func-calling-singleturn.json:14680064"    # 14 MiB (实测 17.8 MB)
  "glaive-function-calling-5k.json:16777216" # 16 MiB (实测 20.4 MB)
)

echo "[镜像] ${MIRROR} → ${BASE}"
echo "[目录] ${RAW}"

for entry in "${FILES[@]}"; do
  fname="${entry%%:*}"
  min_size="${entry##*:}"
  url="${BASE}/datasets/NousResearch/hermes-function-calling-v1/resolve/main/${fname}"
  dest="${RAW}/${fname}"

  if [[ -s "${dest}" ]] && [[ $(wc -c < "${dest}") -ge ${min_size} ]]; then
    sz=$(wc -c < "${dest}")
    echo "[跳过] ${fname} ($(awk "BEGIN{printf \"%.1f\", ${sz}/1024/1024}") MiB)"
    continue
  fi

  echo "[下载] ${fname}  ←  ${url}"
  part="${dest}.part"
  curl -fSL -C - --connect-timeout 30 --max-time 1800 \
       --retry 5 --retry-delay 5 --retry-all-errors \
       "${url}" -o "${part}"

  got=$(wc -c < "${part}")
  if [[ ${got} -lt ${min_size} ]]; then
    echo "[错误] ${fname} 下载不完整: ${got} 字节 (期望 ≥ ${min_size})" >&2
    echo "[提示] 重跑脚本可断点续传；或 HF_MIRROR=huggingface $0" >&2
    exit 1
  fi
  mv "${part}" "${dest}"
  sz=$(wc -c < "${dest}")
  echo "       完成: $(awk "BEGIN{printf \"%.1f\", ${sz}/1024/1024}") MiB"
done

echo "[全部完成] $(ls -1 "${RAW}"/*.json 2>/dev/null | wc -l) 个文件"