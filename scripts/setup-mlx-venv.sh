#!/usr/bin/env bash
# Create the pinned, isolated Python environment llmario-beta uses for the MLX-LM backend.
# llmario-beta finds it automatically at $LLMARIO_BETA_HOME/venvs/mlx (default ~/.llmario-beta/venvs/mlx).
set -euo pipefail

MLX_LM_VERSION="${MLX_LM_VERSION:-0.31.3}"   # tested version; see docs/support-matrix.toml
MLX_VERSION="${MLX_VERSION:-0.32.2}"
HOME_DIR="${LLMARIO_BETA_HOME:-$HOME/.llmario-beta}"
VENV="$HOME_DIR/venvs/mlx"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX requires Apple Silicon macOS." >&2
  exit 1
fi

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

echo "creating $VENV with mlx-lm==$MLX_LM_VERSION mlx==$MLX_VERSION"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install "mlx-lm==$MLX_LM_VERSION" "mlx==$MLX_VERSION"
"$VENV/bin/python" -c 'import mlx_lm, mlx.core as mx; print("ok: mlx-lm", mlx_lm.__version__, "mlx", mx.__version__)'
echo "run \`llmario-beta doctor\` to confirm the backend is detected (via managed venv)."
