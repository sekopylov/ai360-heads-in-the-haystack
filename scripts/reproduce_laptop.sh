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
RUNS=()
for m in "${MODELS[@]}"; do RUNS+=("results/$m"); done

echo "### 1/6 architecture census"
for m in "${MODELS[@]}"; do
  # --out: without it no model_info.json is written and the census artifact is
  # missing from the run tree (the driver passes it; this script did not).
  "$PY" -m retrieval_heads.cli describe --model "$m" --out "results/$m"
done

echo "### 2/6 retrieval-head detection (laptop grid)"
for m in "${MODELS[@]}"; do
  # `--argmax-domain haystack` is the code default; pinned here so the script and
  # SCALES["laptop"] cannot drift apart (tests/test_cli_argv.py compares them).
  "$PY" -m retrieval_heads.cli detect --model "$m" --profile laptop \
      --argmax-domain haystack
done

echo "### 3/6 masking + token-mixer ablation"
# Fractions, not absolute K: absolute K is not comparable across a 48-head and a
# 448-head model.  Mirrors SCALES["laptop"] in scripts/datasphere_job.py.
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli mask --model "$m" \
      --k-frac 0.02 0.04 0.08 0.17 0.33 --lengths 1024 --random-trials 2
done

echo "### 4/6 downstream tasks"
for m in "${MODELS[@]}"; do
  "$PY" -m retrieval_heads.cli qa  --model "$m" --k-frac 0.04 0.08 0.17 --random-trials 2
  "$PY" -m retrieval_heads.cli cot --model "$m" --k-frac 0.08 --random-trials 1 --max-new-tokens 192
done

echo "### 5/6 cross-model comparison"
if [[ ${#MODELS[@]} -ge 2 ]]; then
  "$PY" -m retrieval_heads.cli compare --runs "${RUNS[@]}" --out results
else
  echo "(skipped: compare needs at least two models)"
fi

echo "### 6/6 figures"
"$PY" -m retrieval_heads.cli figures --runs "${RUNS[@]}" --out results/figures

echo
echo "done -> results/ and results/figures/"
