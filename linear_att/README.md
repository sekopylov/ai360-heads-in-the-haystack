# Retrieval heads для линейного внимания (Mamba-2)

Обобщение работы Wu et al. 2024, *Retrieval Head Mechanistically Explains Long-Context Factuality*
(arXiv:2404.15574) с softmax-внимания на линейное внимание / SSM.

## 1. Идея обобщения

**Исходное определение (софтмакс).** На каждом шаге декодирования голова *h* «копирует»,
если argmax её весов внимания (запрос — текущий токен) попадает в позицию *s* внутри иголки
и токен `x_s` совпадает с только что сгенерированным `w`. Счёт головы равен доле токенов
иголки, скопированных этой головой (`|g_h ∩ k| / |k|`). Затем он усредняется по успешным
прогонам NIAH, где ROUGE-1 recall > 50.

**Линейное внимание как маскированное внимание.** Почти любой вариант линейного внимания
можно развернуть в явную причинную матрицу (Structured Masked Attention, Dao & Gu 2024):

```
y_t = Σ_{s≤t} α_{t,s} v_s ,      α_{t,s} = q_tᵀ M_{t,s} k_s
```

| семейство | M_{t,s} (вклад s в t) |
|---|---|
| vanilla linear attn | `1` (+ нормировка `/ Σ_j φ(q_t)·φ(k_j)`), q,k → φ(q), φ(k) |
| RetNet | `γ^{t-s}` |
| Mamba-2 / SSD | `exp(Σ_{j=s+1..t} Δ_j A_h)·Δ_s` (скаляр на голову) |
| GLA / RWKV-6 / HGRN-2 | `diag(Π_{j=s+1..t} a_j)` (по каналам) |
| DeltaNet / Gated DeltaNet / RWKV-7 | `Π_{j=s+1..t}(α_j (I − β_j k_j k_jᵀ))·β_s` (матрицы) |

Получается «неявное» (hidden) внимание `α_{t,s}`, к которому применимо определение из статьи.
В отличие от софтмакса здесь есть две особенности:

1. `α` не нормировано и может быть отрицательным. При этом нормы значений `v_s` сильно
   различаются между токенами, так что argmax по `α` не означает «откуда пришла
   информация».
2. Поэтому argmax берётся по **норме вклада** токена `‖α_{t,s} v_s‖ = |α_{t,s}|·‖v_s‖`
   (norm-based анализ, Kobayashi et al. 2020). Флаг `--weighting alpha` даёт обычный argmax |α|.

**Обобщённый retrieval score.** Для шага *t* со сгенерированным токеном *w*:

```
s* = argmax_s |α^h_{t,s}|·‖v^h_s‖ ;   hit_h(t) = [s* ∈ needle] · [x_{s*} = w]
score_h = Σ_t hit_h(t) / |needle|        (среднее по успешным прогонам)
```

Дополнительно считается **soft-score**: вместо индикатора top-1 берётся доля массы
`|α|·‖v‖`, которая приходится на позиции-источники копирования. Для линейного внимания это
полезно: «копирование» тут часто распределено, а не сосредоточено в одном пике.

**Маскирование головы.** Авторы зануляют query, и софтмакс-голова теряет способность
селективно смотреть в контекст. В Mamba-2 матрицы C (query) и B (key) общие для голов
одной группы, поэтому аналог делается через Δ: для маскируемой головы ставится `Δ_h ≡ 0`.
Голова тогда никогда не пишет в своё состояние, её смешивающая токены часть равна нулю
(точно, а не приближённо), а локальный skip `D_h x_t` остаётся. Вариант `--mask_mode out`
зануляет весь выход головы (столбцы `out_proj`).

**Особенность Mamba.** Перед SSM стоит causal conv1d с шириной 4, поэтому `x_s` и `B_s`
содержат примесь токенов `s-3..s`. Флаг `--pos_tolerance k` засчитывает попадание, если
копируемый токен стоит до *k* позиций левее. По умолчанию 0, как в статье.

## 2. Модель

Основная модель — **Mamba-2 2.7B** (`AntonV/mamba2-2.7b-hf`, HF-конвертация
`state-spaces/mamba2-2.7b`): чистое SSD/линейное внимание без единого софтмакс-слоя,
64 слоя × 80 голов = **5120 голов**, head_dim 64, state 128. Обучена на Pile с контекстом 2k,
поэтому длины контекста по умолчанию 500–2000.

Альтернатива: `mistralai/Mamba-Codestral-7B-v0.1` (7.3B, 64 × 128 = 8192 головы, n_groups = 8,
длинный контекст). Код поддерживает её без изменений, только поменяйте `--model` и `--lengths`.

## 3. Запуск

```bash
pip install "torch>=2.1" "transformers>=4.44" accelerate rouge_score matplotlib numpy
# опционально, для скорости:  pip install mamba-ssm causal-conv1d
git clone https://github.com/nightdessert/Retrieval_Head     # хейстек и иголки авторов

# 0) проверка: неявное внимание точно восстанавливает выход SSM, маска работает
python verify_implicit_attention.py --model AntonV/mamba2-2.7b-hf --dtype float32

# 1) подсчёт retrieval score (3 иголки × 4 длины × 10 глубин = 120 прогонов)
python detect_retrieval_heads.py --model AntonV/mamba2-2.7b-hf \
    --haystack_dir Retrieval_Head/haystack_for_detect \
    --lengths 500 1000 1500 2000 --depth_intervals 10
#   -> head_score/mamba2-2.7b-hf.json  (+ _soft.json, _runs.jsonl)

# 2) маскирование top-K и K случайных голов (иголки отложенные, не из детекции)
python mask_heads_niah.py --model AntonV/mamba2-2.7b-hf \
    --head_score head_score/mamba2-2.7b-hf.json \
    --haystack_dir Retrieval_Head/haystack_for_detect --needles_file needles_eval.jsonl \
    --ks 0 5 10 20 50 100 200 --random_seeds 3 --mask_mode dt

# 3) графики
python plot_results.py --head_score head_score/mamba2-2.7b-hf.json \
    --mask_results results/mask_mamba2-2.7b-hf_dt.json --out_dir figs
```

Графики:
* `figs/mamba2-2.7b-hf_counts.png`: число голов в корзинах `score<0.1`, `0.1≤score<0.4`,
  `score≥0.4` (лог-шкала, с процентами), отсортированные скоры и тепловая карта слой×голова.
  То же для soft-score: `..._soft_counts.png`.
* `figs/mask_mamba2-2.7b-hf_dt.png`: NIAH-score в зависимости от K для top-K и для K
  случайных голов (среднее ± std по сидам).
* `figs/mask_..._heatmaps.png`: NIAH-тепловые карты (длина × глубина), как в статье: без
  маски, top-K и random-K.

Если модель редко решает NIAH (меньше ~10 успешных прогонов), уменьшите `--lengths` или
добавьте `--accumulate_all`.

## Файлы
* `mamba2_implicit_attention.py`: неявное внимание Mamba-2, подсчёт hard/soft retrieval
  score через hooks (работает и с torch-путём, и с CUDA-ядрами), маскирование голов,
  жадное декодирование.
* `niah_utils.py`: построение NIAH-промптов (портировано из кода авторов), ROUGE.
* `detect_retrieval_heads.py`, `mask_heads_niah.py`, `plot_results.py`,
  `verify_implicit_attention.py`: скрипты.
* `needles_eval.jsonl`: отложенные иголки для эксперимента с маскированием.
