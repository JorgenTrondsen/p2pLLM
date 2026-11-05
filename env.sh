# source env.sh: the project's Python, CUDA for FlashInfer's kernels, and nothing written to nas2.
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export HF_HOME="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.cache/hf"
