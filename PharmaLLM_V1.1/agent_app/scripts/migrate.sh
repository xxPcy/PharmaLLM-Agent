#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m alembic upgrade head
python -m pharma_agent.cli checkpoint-setup
