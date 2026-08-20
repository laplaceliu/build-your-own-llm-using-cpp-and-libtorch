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
set -euo pipefail
ROOT="/home/maigi/Source/github.com/laplaceliu/build-your-own-llm-using-cpp-and-libtorch"
SIZE="${SIZE:-medium}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-4}"          # 默认 batch=4，避免 OOM
LR="${LR:-5e-5}"
OUT_DIR="${ROOT}/gpt-bash/data"
OUT_NAME="bash-sft-${SIZE}.pth"
LOG="${ROOT}/gpt-bash/logs/train-${SIZE}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${ROOT}/gpt-bash/logs"

# 减少 PyTorch CUDA caching allocator 碎片（OOM 时官方建议）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}/gpt-sft"
echo "[$(date)] 训练开始: size=${SIZE} epochs=${EPOCHS} batch=${BATCH} lr=${LR}"
echo "[$(date)] 输出到: ${OUT_DIR}/${OUT_NAME}"
echo "[$(date)] 日志: ${LOG}"
echo "[$(date)] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
./scripts/run.sh train \
    --data "${ROOT}/gpt-bash/data/bash-instruction-data.json" \
    --weights "${ROOT}/chapters/chapter07_instruction_tuning/data/gpt2-medium.safetensors" \
    --size "${SIZE}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --out "${OUT_DIR}/${OUT_NAME}" 2>&1 | tee "${LOG}"
echo "[$(date)] 训练结束"
