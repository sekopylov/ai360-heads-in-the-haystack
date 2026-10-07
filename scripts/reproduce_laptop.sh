#!/usr/bin/env bash
# End-to-end reproduction on a CPU-only laptop: every stage of the paper, at a
# scale that finishes in well under an hour.
#
#   ./scripts/reproduce_laptop.sh
#
# Stages
#   1. architecture census        (what is even scoreable?)
#   2. retrieval-head detection   (paper Sec. 3)
#   3. masking curve + mixer abl. (paper Sec. 4.1 and 5)
#   4. extractive QA + CoT        (paper Sec. 5)
#   5. cross-model correlation    (paper Sec. 4.3)
#   6. figures                    (paper figures -> results/figures)
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(qwen3.5-0.8b qwen3-0.6b); fi

echo "### 1/6 architecture census"
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli describe --model "$m"
done

echo "### 2/6 retrieval-head detection (laptop grid)"
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli detect --model "$m" --profile laptop
done

echo "### 3/6 masking + token-mixer ablation"
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli mask --model "$m" \
      --k 1 2 4 8 16 --lengths 1024 --random-trials 2
done

echo "### 4/6 downstream tasks"
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli qa  --model "$m" --k 8 --random-trials 2
  "$PY" -m retrieval_heads.cli cot --model "$m" --k 8 --random-trials 1 --max-new-tokens 192
done

echo "### 5/6 cross-model comparison"
"$PY" -m retrieval_heads.cli compare \
    --runs results/qwen3.5-0.8b results/qwen3-0.6b --out results

echo "### 6/6 figures"
"$PY" -m retrieval_heads.cli figures \
    --runs results/qwen3.5-0.8b results/qwen3-0.6b --out results/figures

echo
echo "done -> results/ and results/figures/"
