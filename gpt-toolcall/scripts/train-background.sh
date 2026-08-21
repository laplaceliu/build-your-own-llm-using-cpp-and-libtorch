#!/usr/bin/env bash
# 后台训练脚本：medium + 3 epochs，写日志到 logs/
#  默认 medium 是因为唯一可用的预训练权重是 gpt2-medium.safetensors（1024/16/24）
#
# 显存预算（gpt2-medium 354M, RTX 4080 16GB, seq_len=1024, vocab=50257）：
#   模型 fp32 + Adam(m+v) + grad       ≈ 5.6 GiB
#   logits [batch, 1024, 50257] fp32   ≈ batch * 0.2 GiB
#   attention activations              ≈ 0.5 GiB
#   batch=4  → 总  ~6.8 GiB（安全）
#   batch=8  → 总  ~7.6 GiB（边缘）
#   batch=16 → 总 ~9.4 GiB + 碎片化 → 撞 14.7 GiB OOM（实测）
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减少 caching allocator 碎片。
#
# 与 gpt-bash 同源，唯一区别：data/out 是 toolcall-* 而非 bash-*
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
SIZE="${SIZE:-medium}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-1}"          # gpt-toolcall: 默认 batch=1（tool descriptions 让 prompt 很长 + medium 24 layers 容易 OOM）
LR="${LR:-5e-5}"
OUT_DIR="${ROOT}/data"
WEIGHTS_DIR="${ROOT}/../chapters/chapter07_instruction_tuning/data"
WEIGHTS="${WEIGHTS_DIR}/gpt2-medium.safetensors"
OUT_NAME="toolcall-sft-v2-${SIZE}.pth"
LOG="${ROOT}/logs/train-${SIZE}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${ROOT}/logs" "${WEIGHTS_DIR}"

# 减少 PyTorch CUDA caching allocator 碎片（OOM 时官方建议）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 预训练权重预检：缺失则提示用户跑下载脚本（避免 1.4 GB 下载后才发现路径不对）
if [[ ! -s "${WEIGHTS}" ]]; then
  echo "[错误] 预训练权重不存在: ${WEIGHTS}" >&2
  echo "[提示] 第一次训练？请先运行:" >&2
  echo "       ${SELF_DIR}/download-weights.sh" >&2
  echo "       （默认走 hf-mirror 国内加速；可 HF_MIRROR=huggingface 切回 HF 官方）" >&2
  exit 1
fi

cd "${ROOT}/../gpt-sft"
echo "[$(date)] 训练开始: size=${SIZE} epochs=${EPOCHS} batch=${BATCH} lr=${LR}"
echo "[$(date)] 权重: ${WEIGHTS} (大小 $(stat -c%s "${WEIGHTS}") 字节)"
echo "[$(date)] 输出到: ${OUT_DIR}/${OUT_NAME}"
echo "[$(date)] 日志: ${LOG}"
echo "[$(date)] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
./scripts/run.sh train \
    --data "${ROOT}/data/toolcall-data-v2.json" \
    --weights "${WEIGHTS}" \
    --size "${SIZE}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --out "${OUT_DIR}/${OUT_NAME}" 2>&1 | tee "${LOG}"
echo "[$(date)] 训练结束"