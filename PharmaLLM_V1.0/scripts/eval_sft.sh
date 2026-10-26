#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/qwen3-8b-base-pharma-sft}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  MODEL_PATH="$1"
  shift
fi

OUTPUT_PATH="${OUTPUT_PATH:-reports/sft_eval.json}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  OUTPUT_PATH="$1"
  shift
fi

python -m pharma_llm.evaluation.runner \
  --model-path "${MODEL_PATH}" \
  --eval-config configs/eval.yaml \
  --output "${OUTPUT_PATH}" \
  "$@"
