#!/usr/bin/env bash
#
# Set up a pinned Python 3.11 virtualenv for NeuralGCM and register it as a
# Jupyter kernel. NeuralGCM / dinosaur / JAX are not reliably tested on the
# devcontainer's Python 3.13, so we use uv to fetch a standalone 3.11.
#
# Usage:  bash setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. Ensure uv is available (no root required; installs to ~/.local/bin).
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Create the pinned 3.11 venv.
echo ">> creating .venv on Python 3.11 ..."
uv venv --python 3.11 .venv

# 3. Install dependencies. If the single-resolve install fails on a JAX version
#    conflict, fall back to: install neuralgcm first, then pin jax[cuda12].
echo ">> installing dependencies ..."
if ! uv pip install --python .venv/bin/python -r requirements.txt; then
  echo ">> single-resolve failed; retrying with neuralgcm-first JAX pin ..."
  uv pip install --python .venv/bin/python neuralgcm
  JAX_VER="$(.venv/bin/python -c 'import importlib.metadata as m; print(m.version("jax"))')"
  echo ">> neuralgcm pinned jax==${JAX_VER}; installing matching CUDA build ..."
  uv pip install --python .venv/bin/python "jax[cuda12]==${JAX_VER}"
  uv pip install --python .venv/bin/python \
    gcsfs xarray zarr matplotlib ipykernel jupyter nbconvert
fi

# 4. Register the venv as a Jupyter kernel so the notebook can select it.
echo ">> registering Jupyter kernel 'neuralgcm' ..."
.venv/bin/python -m ipykernel install --user \
  --name neuralgcm --display-name "Python (NeuralGCM 3.11)"

echo ">> done. Verifying JAX sees the GPU:"
.venv/bin/python -c "import jax; print('JAX devices:', jax.devices())"
