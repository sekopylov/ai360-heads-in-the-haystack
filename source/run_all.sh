#!/usr/bin/env bash
# Полный пайплайн retrieval heads для Qwen2.5-0.5B-Instruct:
#   0) проверка окружения   1) детекция retrieval heads   2) pie chart
#   3) NIAH с выкинутыми 20 головами (top / random / bottom)   4) графики
#
# Запускать из корня репозитория Retrieval_Head (там лежат haystack_for_detect/ и PaulGrahamEssays/),
# положив рядом все .py из этого набора. Параметры можно переопределить переменными окружения:
#   MODEL=... E_LEN=8000 K=20 ./run_all.sh
# Пропуск этапов: SKIP_DETECT=1 (если head_score уже посчитан), SKIP_MASK=1.
# APPEND=1 -- дописать новые пробы к существующему head_score (как в оригинальном скрипте).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
# детекция
S_LEN="${S_LEN:-1000}"
E_LEN="${E_LEN:-16000}"
NUM_LENGTHS="${NUM_LENGTHS:-8}"
NUM_DEPTHS="${NUM_DEPTHS:-10}"
# маскирование
K="${K:-20}"
RANDOM_SEEDS="${RANDOM_SEEDS:-3}"
EVAL_S_LEN="${EVAL_S_LEN:-1000}"
EVAL_E_LEN="${EVAL_E_LEN:-16000}"
EVAL_NUM_LENGTHS="${EVAL_NUM_LENGTHS:-8}"
EVAL_NUM_DEPTHS="${EVAL_NUM_DEPTHS:-10}"
EVAL_NEEDLES="${EVAL_NEEDLES:-sf}"

mkdir -p logs
RUN_LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | run_all    | $*" | tee -a "$RUN_LOG"; }

run_stage() {
  local name="$1"; shift
  log ">>> ЭТАП: $name"
  log "    команда: $*"
  local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$RUN_LOG"; then
    log "<<< ЭТАП '$name' завершён за $((SECONDS - t0)) c"
  else
    local rc=${PIPESTATUS[0]}
    log "!!! ЭТАП '$name' упал (код $rc), см. $RUN_LOG"
    exit 1
  fi
}

log "Старт пайплайна. Модель: $MODEL; лог: $RUN_LOG"
log "детекция: $S_LEN..$E_LEN, $NUM_LENGTHS длин x $NUM_DEPTHS глубин"
log "маскирование: K=$K, случайных наборов=$RANDOM_SEEDS, $EVAL_S_LEN..$EVAL_E_LEN, иглы=$EVAL_NEEDLES"

run_stage "0. Проверка окружения" \
  python -u check_setup.py --model_path "$MODEL"

if [[ "${SKIP_DETECT:-0}" != "1" ]]; then
  run_stage "1. Детекция retrieval heads" \
    python -u detect_retrieval_heads.py --model_path "$MODEL" --dtype "$DTYPE" \
      -s "$S_LEN" -e "$E_LEN" --num_lengths "$NUM_LENGTHS" --num_depths "$NUM_DEPTHS" \
      $([[ "${APPEND:-0}" == "1" ]] && echo --append)
else
  log "    этап 1 пропущен (SKIP_DETECT=1)"
fi

run_stage "2. Pie chart распределения голов" \
  python -u plot_head_distribution.py --model_path "$MODEL" --lo 0.1 --hi 0.4

if [[ "${SKIP_MASK:-0}" != "1" ]]; then
  run_stage "3. NIAH с выкинутыми $K головами (baseline/top/random/bottom)" \
    python -u eval_head_masking.py --model_path "$MODEL" --dtype "$DTYPE" --k "$K" \
      --random_seeds "$RANDOM_SEEDS" --eval_needles "$EVAL_NEEDLES" \
      -s "$EVAL_S_LEN" -e "$EVAL_E_LEN" --num_lengths "$EVAL_NUM_LENGTHS" --num_depths "$EVAL_NUM_DEPTHS"
else
  log "    этап 3 пропущен (SKIP_MASK=1)"
fi

run_stage "4. Графики маскирования" \
  python -u plot_masking_results.py --model_path "$MODEL"

log "Готово. Графики: figures/, скоры: head_score/, сырые результаты: results/, логи этапов: logs/"