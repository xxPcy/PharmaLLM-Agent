#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"

# On this host, physical GPU 3 currently hangs NCCL P2P collectives.
# Keep P2P enabled by default, but use the verified 0,1,2,4 set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m pharma_llm.training.sft \
  --model-config configs/model.yaml \
  --train-config configs/train_sft.yaml \
  "$@"
