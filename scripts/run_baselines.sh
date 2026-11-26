#!/usr/bin/env bash
# Train + evaluate Base/SEAM/SDD/CTRAP (default: Qwen2.5-7B and Mistral-7B, seeds 42 43 44) and print the table.
# Arguments are passed to baselines/run_all.py (see --help). Rerun the same command to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" baselines/run_all.py "$@"
