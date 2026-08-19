#!/usr/bin/env bash
# scripts/start-ollama.sh
# 启动 Ollama 服务并确保 llama3 模型就绪（第 7 章 7.8 自动评分依赖）
#
# 用法: ./scripts/start-ollama.sh [--pull]
#   --pull   同时拉取 llama3 模型（首次约 4.7 GB）
set -euo pipefail

OLLAMA_URL="http://localhost:11434"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"

# 1. 检查 ollama 是否安装
if ! command -v ollama >/dev/null 2>&1; then
  echo "[错误] 未找到 ollama，请先安装："
  echo "  curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

# 2. 若服务未运行则后台启动
if curl --noproxy '*' -fsSL --connect-timeout 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "[就绪] Ollama 服务已在运行 (${OLLAMA_URL})"
else
  echo "[启动] 后台启动 ollama serve ..."
  nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
  sleep 4
  if curl --noproxy '*' -fsSL --connect-timeout 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "[就绪] Ollama 服务已启动 (${OLLAMA_URL})"
  else
    echo "[错误] Ollama 启动失败，请查看 /tmp/ollama_serve.log"
    exit 1
  fi
fi

# 3. 检查模型是否存在；--pull 时拉取
if curl --noproxy '*' -fsSL "${OLLAMA_URL}/api/tags" 2>/dev/null | grep -q "\"name\":\"${OLLAMA_MODEL}"; then
  echo "[就绪] 模型 ${OLLAMA_MODEL} 已存在"
elif [ "${1:-}" = "--pull" ]; then
  echo "[拉取] ollama pull ${OLLAMA_MODEL}（首次约 4.7 GB，请耐心等待）..."
  ollama pull "${OLLAMA_MODEL}"
  echo "[完成] 模型 ${OLLAMA_MODEL} 已就绪"
else
  echo "[提示] 模型 ${OLLAMA_MODEL} 未安装，运行第 7 章 7.8 前请执行:"
  echo "  ./scripts/start-ollama.sh --pull"
fi
