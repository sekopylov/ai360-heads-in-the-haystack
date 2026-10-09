#!/usr/bin/env bash
# Paper-scale run, intended for a GPU box.  The only difference from
# reproduce_laptop.sh is the detection grid and the context lengths, which is
# the whole point of keeping the grid in DetectionConfig.
#
#   ./scripts/reproduce_gpu.sh qwen3.5-0.8b
#
# The paper's full recipe is 3 needle sets x 20 lengths in 1K-50K x 10 depths
# (~600 instances per model).  `--profile paper` uses 3 x 7 x 10; the explicit
# --lengths below widens that to 9 geometric lengths over the same span.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(qwen3.5-0.8b qwen3-0.6b); fi
RUNS=()
for m in "${MODELS[@]}"; do RUNS+=("results/$m"); done

# On a GPU, `flash-linear-attention` + `causal-conv1d` remove the pure-PyTorch
# fallback that dominates Qwen3.5's runtime; the code path is identical.
#   uv pip install flash-linear-attention causal-conv1d

# bfloat16 matches what the DataSphere job configs (t4.yaml, t4-cached.yaml,
# paper.yaml) use, so a local GPU run and a job run are the same experiment; the
# artifact records the dtype, so a mixed comparison is visible.
DTYPE=(--dtype bfloat16)

# Census first, as the laptop script and the job driver do: without --out no
# model_info.json is written and the run tree loses that artifact.
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli describe --model "$m" --out "results/$m" "${DTYPE[@]}"
done

for m in "${MODELS[@]}"; do
  # `--argmax-domain haystack` is the paper's `a in R^{|x|}` (the context span, no
  # question or template) and the code default; pinned so SCALES["paper"] and this
  # script cannot drift (tests/test_cli_argv.py compares them).
  "$PY" -m retrieval_heads.cli detect --model "$m" --profile paper "${DTYPE[@]}" \
      --argmax-domain haystack \
      --lengths 1024 2048 4096 8192 16384 24576 32768 40960 49152
done

for m in "${MODELS[@]}"; do
  # Fractions, not absolute K: K=32 is 67% of Qwen3.5-0.8B's 48 scoreable heads
  # but 7% of Qwen3-0.6B's 448, and on the hybrid every K above the non-retrieval
  # pool collapses to the same point.  Mirrors SCALES["paper"] in the job driver.
  "$PY" -m retrieval_heads.cli mask --model "$m" "${DTYPE[@]}" \
      --k-frac 0.01 0.02 0.04 0.08 0.17 0.33 \
      --lengths 4096 8192 16384 --random-trials 5
  "$PY" -m retrieval_heads.cli qa  --model "$m" "${DTYPE[@]}" \
      --k-frac 0.04 0.08 0.17 --random-trials 5
  "$PY" -m retrieval_heads.cli cot --model "$m" "${DTYPE[@]}" \
      --k-frac 0.08 --random-trials 3 --max-new-tokens 256
done

if [[ ${#MODELS[@]} -ge 2 ]]; then
  "$PY" -m retrieval_heads.cli compare --runs "${RUNS[@]}" --out results
else
  echo "(skipped: compare needs at least two models)"
fi
"$PY" -m retrieval_heads.cli figures --runs "${RUNS[@]}" --out results/figures
