#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/qwen3-8b-pharma-sft}"
BIND_HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"

# On this host, physical GPU 3 has shown NCCL P2P hangs in training tests.
# Use the verified 0,1,2,4 set unless the caller explicitly overrides it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed in the current Python environment." >&2
  echo "Install it first, for example:" >&2
  echo "  python -m pip install -U vllm" >&2
  exit 1
fi

PREFIX_FLAG=()
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  PREFIX_FLAG=(--enable-prefix-caching)
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${BIND_HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${PREFIX_FLAG[@]}"
