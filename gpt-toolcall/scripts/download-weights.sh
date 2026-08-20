#!/usr/bin/env bash
# gpt-toolcall/scripts/download-weights.sh
#
# 下载 gpt2-medium.safetensors（HF 官方仓库 openai-community/gpt2-medium）
# 大小 ~1.43 GiB（FP32 全精度），保存到
#   chapters/chapter07_instruction_tuning/data/gpt2-medium.safetensors
# 该路径与 train-background.sh 默认期望一致。
#
# 镜像选择（按顺序优先级）：
#   1) $HF_MIRROR  环境变量覆盖一切
#   2) 默认: hf-mirror   （中国大陆友好，免代理，平均 10 MB/s+）
#   3) huggingface       （HF 官方，需要海外网络）
#
# 用法:
#   ./scripts/download-weights.sh                 # 默认 hf-mirror
#   HF_MIRROR=huggingface ./scripts/download-weights.sh
#   HF_MIRROR=hf-mirror  ./scripts/download-weights.sh
#
# 网络: 必要时设置 ALL_PROXY / HTTPS_PROXY / HF_HUB_DOWNLOAD_TIMEOUT 等。
#
# 跳过条件: 目标文件已存在且 ≥ MIN_SIZE 字节（避免重下 1.4 GB）。
# 失败处理: 中途保留 .part，可重跑脚本续传。
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
DEST_DIR="${ROOT}/../chapters/chapter07_instruction_tuning/data"
DEST="${DEST_DIR}/gpt2-medium.safetensors"

# HF 上 gpt2-medium FP32 ~1.43 GiB (1535 MB)；设 1.3 GiB 为最小阈值，
# 防止下载到残缺文件就提前 mv 过去。
MIN_SIZE=$((1300 * 1024 * 1024))

# 镜像解析
MIRROR="${HF_MIRROR:-hf-mirror}"
case "${MIRROR}" in
  hf-mirror)   BASE="https://hf-mirror.com" ;;
  huggingface) BASE="https://huggingface.co" ;;
  *)
    echo "[警告] 未知镜像 '${MIRROR}'，回退到 hf-mirror" >&2
    BASE="https://hf-mirror.com" ;;
esac
URL="${BASE}/openai-community/gpt2-medium/resolve/main/model.safetensors"

mkdir -p "${DEST_DIR}"

# 跳过：已存在且够大
if [[ -s "${DEST}" ]]; then
  size=$(wc -c < "${DEST}")
  if [[ ${size} -ge ${MIN_SIZE} ]]; then
    echo "[跳过] ${DEST}"
    echo "       大小 ${size} 字节 (≥ ${MIN_SIZE})"
    exit 0
  fi
  echo "[注意] ${DEST} 仅 ${size} 字节，视为残缺文件，重新下载"
  rm -f "${DEST}"
fi

echo "[镜像] ${MIRROR} → ${BASE}"
echo "[下载] ${URL}"
echo "[保存] ${DEST}  (最小 ${MIN_SIZE} 字节 ≈ 1.3 GiB)"

PART="${DEST}.part"
# -C - 断点续传；-f 失败非 200 报错退出；-L 跟随重定向；--max-time 防卡死
curl -fSL -C - --connect-timeout 30 --max-time 1800 \
     --retry 5 --retry-delay 5 --retry-all-errors \
     "${URL}" -o "${PART}"

got=$(wc -c < "${PART}")
if [[ ${got} -lt ${MIN_SIZE} ]]; then
  echo "[错误] 下载不完整: ${got} 字节 (期望 ≥ ${MIN_SIZE})" >&2
  echo "[提示] 重跑脚本可断点续传；或换镜像:" >&2
  echo "       HF_MIRROR=huggingface $0  (HF 官方)" >&2
  echo "       # 或设置 ALL_PROXY=socks5h://host:port 走代理" >&2
  exit 1
fi

mv "${PART}" "${DEST}"
final=$(wc -c < "${DEST}")
echo "[完成] ${DEST}"
echo "       大小 ${final} 字节 (~$(awk "BEGIN{printf \"%.2f\", ${final}/1024/1024/1024}") GiB)"