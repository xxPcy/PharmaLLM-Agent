#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/data.yaml}"
INPUT_JSONL="${INPUT_JSONL:-data/interim/deduplicated.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-data/processed/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/processed/val.jsonl}"
SPLIT_REPORT="${SPLIT_REPORT:-reports/split_report.json}"

python -m pharma_llm.data.split \
  --config "${CONFIG_PATH}" \
  --input "${INPUT_JSONL}" \
  --train-output "${TRAIN_JSONL}" \
  --validation-output "${VAL_JSONL}" \
  --report "${SPLIT_REPORT}"
