#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/data.yaml}"
INPUT_JSONL="${INPUT_JSONL:-data/interim/merged_normalized.jsonl}"
VALID_JSONL="${VALID_JSONL:-data/interim/merged_valid.jsonl}"
INVALID_JSONL="${INVALID_JSONL:-data/interim/merged_invalid.jsonl}"
DEDUP_JSONL="${DEDUP_JSONL:-data/interim/deduplicated.jsonl}"
NORMALIZATION_REPORT="${NORMALIZATION_REPORT:-reports/normalization_report.json}"
RUN_NORMALIZE="${RUN_NORMALIZE:-true}"
INCLUDE_TCM_TRANSLATION="${INCLUDE_TCM_TRANSLATION:-false}"
TCM_INPUT_FILE="${TCM_INPUT_FILE:-data/interim/tcm_20k.json}"

if [[ "${RUN_NORMALIZE}" == "true" ]]; then
  NORMALIZE_ARGS=(
    --config "${CONFIG_PATH}"
    --merged-output "${INPUT_JSONL}"
    --report "${NORMALIZATION_REPORT}"
  )

  if [[ -f "${TCM_INPUT_FILE}" ]]; then
    NORMALIZE_ARGS+=(--tcm-input-file "${TCM_INPUT_FILE}")
  fi

  if [[ "${INCLUDE_TCM_TRANSLATION}" == "true" ]]; then
    NORMALIZE_ARGS+=(--include-tcm-translation)
  fi

  python -m pharma_llm.data.normalize "${NORMALIZE_ARGS[@]}"
fi

python -m pharma_llm.data.validate \
  --input "${INPUT_JSONL}" \
  --output "${VALID_JSONL}" \
  --invalid-output "${INVALID_JSONL}" \
  --config "${CONFIG_PATH}" \
  --report reports/schema_validation.json

python -m pharma_llm.data.dedup \
  --input "${VALID_JSONL}" \
  --output "${DEDUP_JSONL}" \
  --duplicates-output data/interim/exact_duplicates.jsonl \
  --report reports/exact_dedup.json
