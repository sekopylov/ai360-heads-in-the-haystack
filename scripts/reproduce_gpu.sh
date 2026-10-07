#!/usr/bin/env bash
# Paper-scale run, intended for a GPU box.  The only difference from
# reproduce_laptop.sh is the detection grid and the context lengths, which is
# the whole point of keeping the grid in DetectionConfig.
#
#   ./scripts/reproduce_gpu.sh qwen3.5-0.8b
#
# The paper's full recipe is 3 needle sets x 20 lengths in 1K-50K x 10 depths
# (~600 instances per model).  `--profile paper` uses 3 x 7 x 10; raise
# --lengths to hit the exact grid.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(qwen3.5-0.8b qwen3-0.6b); fi

# On a GPU, `flash-linear-attention` + `causal-conv1d` remove the pure-PyTorch
# fallback that dominates Qwen3.5's runtime; the code path is identical.
#   uv pip install flash-linear-attention causal-conv1d

for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli detect --model "$m" --profile paper \
      --lengths 1024 2048 4096 8192 16384 24576 32768 40960 49152
done

for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli mask --model "$m" --k 1 2 4 8 16 32 64 128 \
      --lengths 4096 8192 16384 --random-trials 5
  "$PY" -m retrieval_heads.cli qa  --model "$m" --k 8 32 64 --random-trials 5
  "$PY" -m retrieval_heads.cli cot --model "$m" --k 32 --random-trials 3
done

"$PY" -m retrieval_heads.cli compare --runs results/qwen3.5-0.8b results/qwen3-0.6b --out results
"$PY" -m retrieval_heads.cli figures --runs results/qwen3.5-0.8b results/qwen3-0.6b --out results/figures
