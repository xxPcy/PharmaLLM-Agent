#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

python -m pharma_llm.inference.benchmark \
  --config "${CONFIG_PATH:-configs/serving.yaml}" \
  "$@"
