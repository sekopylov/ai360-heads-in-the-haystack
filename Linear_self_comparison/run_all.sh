#!/usr/bin/env bash
# Full pipeline for Qwen3.5-0.8B. Usage: bash run_all.sh [model_path] [max_len]
set -euo pipefail
MODEL=${1:-Qwen/Qwen3.5-0.8B}
E=${2:-16000}
NAME=$(basename "$MODEL")
HAY=${HAYSTACK_ROOT:-../Retrieval_Head}
[ -d "$HAY" ] || git clone https://github.com/nightdessert/Retrieval_Head "$HAY"

# fail fast if there is no GPU (the python scripts also check and abort without --allow_cpu)
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv || { echo "nvidia-smi failed: no GPU?"; exit 1; }
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda available:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"

# every script writes its own timestamped log to logs/ (progress %, ETA, GPU memory)
python selftest.py --model_path "$MODEL" --haystack_root "$HAY"
python detect.py   --model_path "$MODEL" --haystack_root "$HAY" -s 1000 -e "$E"
python ablate.py   --model_path "$MODEL" --haystack_root "$HAY" -s 1000 -e "$E" \
    --head_score "head_score/$NAME.json" --kinds linear,full --ks 1,2,4,8,16 --seeds 0,1,2
python plots.py --head_score "head_score/$NAME.json" --niah_dir results/niah --out figs
python plots.py --head_score "head_score/$NAME.contrib.json" --niah_dir results/niah --out figs_contrib
