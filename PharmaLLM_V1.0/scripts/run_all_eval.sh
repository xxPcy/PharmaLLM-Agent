#!/usr/bin/env bash
set -euo pipefail

bash scripts/eval_base.sh
bash scripts/eval_sft.sh

