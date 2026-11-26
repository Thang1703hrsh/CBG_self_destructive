#!/usr/bin/env bash
# Creates .venv (training + evaluation, Python 3.11) and .venv-ctrap (CTRAP's torch 2.1 stack, Python 3.10).
# uv downloads the Python versions if the server does not have them.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r baselines/requirements.txt
[ -x .venv-ctrap/bin/python ] || uv venv --python 3.10 .venv-ctrap
uv pip install --python .venv-ctrap/bin/python -r baselines/requirements-ctrap.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"
.venv-ctrap/bin/python -c "import torch, transformers; print('ctrap env: torch', torch.__version__, transformers.__version__)"
echo "Setup done. Next: export HF_TOKEN=hf_...; bash scripts/run_baselines.sh"
