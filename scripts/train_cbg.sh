#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m src.cbg.train_v2 --config config/cbg_paper.yaml "$@"
