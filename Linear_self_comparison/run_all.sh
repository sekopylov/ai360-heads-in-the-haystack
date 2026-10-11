#!/usr/bin/env bash
# Full pipeline for Qwen3.5-0.8B. Usage: bash run_all.sh [model_path] [max_len]
set -euo pipefail
MODEL=${1:-Qwen/Qwen3.5-0.8B}
E=${2:-16000}
NAME=$(basename "$MODEL")
HAY=${HAYSTACK_ROOT:-../Retrieval_Head}
[ -d "$HAY" ] || git clone https://github.com/nightdessert/Retrieval_Head "$HAY"

python selftest.py --model_path "$MODEL" --haystack_root "$HAY" | tee logs_selftest.txt
python detect.py   --model_path "$MODEL" --haystack_root "$HAY" -s 1000 -e "$E" | tee logs_detect.txt
python ablate.py   --model_path "$MODEL" --haystack_root "$HAY" -s 1000 -e "$E" \
    --head_score "head_score/$NAME.json" --kinds linear,full --ks 1,2,4,8,16 --seeds 0,1,2 | tee logs_ablate.txt
python plots.py --head_score "head_score/$NAME.json" --niah_dir results/niah --out figs
python plots.py --head_score "head_score/$NAME.contrib.json" --niah_dir results/niah --out figs_contrib
