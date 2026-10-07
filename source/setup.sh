#!/usr/bin/env bash
# Подготовка удалённой GPU-машины. Лежит в source/ (клон Retrieval_Head) рядом с *_qwen35.py.
# Запуск из любой папки:  bash source/setup.sh
# Переменные: MODEL_ID (по умолчанию Qwen/Qwen3.5-0.8B), MODEL_DIR (по умолчанию source/models/<имя модели>)
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
MODEL_DIR="${MODEL_DIR:-$SRC/models/$(basename "$MODEL_ID")}"

# 1. Данные репозитория (haystack_for_detect/, PaulGrahamEssays/) — докачать, если их нет
if [ ! -f "$SRC/haystack_for_detect/needles.jsonl" ] || [ ! -d "$SRC/PaulGrahamEssays" ]; then
  echo "Нет данных репозитория в $SRC — докачиваю"
  TMP="$(mktemp -d)"
  git clone --depth 1 https://github.com/nightdessert/Retrieval_Head.git "$TMP/rh"
  cp -rn "$TMP/rh/haystack_for_detect" "$TMP/rh/PaulGrahamEssays" "$TMP/rh/viz" "$SRC/" 2>/dev/null || true
  rm -rf "$TMP"
fi
mkdir -p "$SRC/head_score" "$(dirname "$MODEL_DIR")"

# 2. Зависимости (нужен transformers с поддержкой qwen3_5)
python -m pip install -U pip
python -m pip install -U "transformers>=5.0" accelerate "huggingface_hub[cli]" rouge_score numpy sentencepiece
if ! python -c "from transformers.models import qwen3_5" 2>/dev/null; then
  echo "В релизной версии нет qwen3_5 — ставлю transformers из main"
  python -m pip install -U "git+https://github.com/huggingface/transformers.git@main"
fi
python -c "import transformers; print('transformers', transformers.__version__)"
# Быстрые ядра для Gated DeltaNet (без них работает torch-fallback, но prefill медленнее):
python -m pip install -U flash-linear-attention || echo "WARN: flash-linear-attention не установился"
python -m pip install causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d не установился (необязательно)"

# 3. Модель
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

echo
echo "Готово. Запуск (из любой папки):"
echo "  python $SRC/retrieval_head_detection_qwen35.py   --model_path $MODEL_DIR --s_len 0 --e_len 50000"
echo "  python $SRC/needle_in_haystack_with_mask_qwen35.py --model_path $MODEL_DIR --mask_topk 10  --e_len 100000"
echo "  python $SRC/needle_in_haystack_with_mask_qwen35.py --model_path $MODEL_DIR --mask_topk -10 --e_len 100000"