# Запуски старого кода: Qwen1.5-14B-Chat, A100 80 ГБ

Ячейки для Jupyter, по одной команде на ячейку. Рабочая папка ноутбука — `source/`. Порядок: подготовка → тестовый прогон → очистка → детекция → маскирование → сохранение.

Что означают этапы и зачем нужна выгрузка — в [report.md](report.md) и [PLAN.md](PLAN.md).

## 0. Что нельзя затереть

| Что | Чем затирается | Как защищено |
|---|---|---|
| `head_score/*.json` — скоры авторов | Детекция **дописывает** в файл с именем модели | Всегда запускать детекцию с `--head_score_dir head_score_ours` |
| `head_score_ours/Qwen1.5-14B-Chat.json` — наши скоры | Повторный запуск детекции дописывает в него второй раз | Перед полной детекцией файла быть не должно (ячейка 3.1 проверяет) |
| `results/graph/Qwen1.5-14B-Chat/` — результаты детекции | Маскирование с `--mask_topk 0` пишет в **ту же папку** | После детекции папка переименовывается (ячейка 3.3) |
| Результаты тестового прогона | Смешиваются с полными в одной папке | Очистка после теста (ячейка 2.8) |
| `results/graph/llama-2-7b-80k.zip` — файл авторов | Удалением всей папки `results/` | Удалять только `results/graph/Qwen1.5-14B-Chat*` |

Удалять вручную перед прогоном ничего из файлов авторов не нужно: достаточно не запускать детекцию без `--head_score_dir`.

Папки `results/` и `contexts/` в `.gitignore`, в git они не попадут. Выгрузки нужно сохранять отдельно (раздел 5).

## 1. Подготовка

```python
!mkdir -p logs
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
!../.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
!../.venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"
!git status --short head_score
```

Последняя команда должна ничего не напечатать: файлы авторов не изменены. Веса модели — около 28 ГБ на диске.

## 2. Тестовый прогон

Цель — проверить, что ничего не падает, хватает памяти на верхней длине, и измерить время на пример. Длины 1000 и 30000.

### 2.1. Детекция: 3 иглы × 2 длины × 3 глубины = 18 примеров

```python
!../.venv/bin/python -u retrieval_head_detection.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 2 \
    --depths 0,50,100 \
    --dump_dir results/dump_test \
    --head_score_dir head_score_test \
    2>&1 | tee logs/test_detect.log
```

### 2.2. Проверка выгрузки детекции

```python
!../.venv/bin/python check_dump.py results/dump_test/Qwen1.5-14B-Chat/detect \
    --ref head_score/Qwen1.5-14B-Chat.json
```

Смотреть: `recomputed score mismatch: 0`, `needle not found: 0`, и что на длине 30000 есть успешные примеры.

### 2.3. Маскирование, без маски: 2 длины × 1 глубина = 2 примера

Головы для теста берутся из файла авторов (скрипт маскирования его только читает).

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 2 \
    --depths 50 \
    --head_score_dir head_score \
    --dump_dir results/dump_test_mask \
    --mask_topk 0 \
    2>&1 | tee logs/test_mask_0.log
```

### 2.4. Маскирование, top-30

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 2 \
    --depths 50 \
    --head_score_dir head_score \
    --dump_dir results/dump_test_mask \
    --mask_topk 30 \
    2>&1 | tee logs/test_mask_top30.log
```

### 2.5. Маскирование, 30 случайных голов

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 2 \
    --depths 50 \
    --head_score_dir head_score \
    --dump_dir results/dump_test_mask \
    --mask_topk -30 \
    2>&1 | tee logs/test_mask_random30.log
```

### 2.6. Проверка, что маскирование действует

```python
import glob, numpy as np

def outputs(folder):
    return {f.split("/")[-1]: np.load(f) for f in sorted(glob.glob(f"results/dump_test_mask/{folder}/*.npz"))}

base = outputs("Qwen1.5-14B-Chat")
top = outputs("Qwen1.5-14B-Chat_block_top30")
rnd = outputs("Qwen1.5-14B-Chat_block_random30")
for f in base:
    print(f, "| без маски:", float(base[f]["rouge"]), "| top30:", float(top[f]["rouge"]), "| random30:", float(rnd[f]["rouge"]))
    print("   ответ top30 отличается от ответа без маски:", not np.array_equal(base[f]["output_ids"], top[f]["output_ids"]))
    print("   замаскировано голов:", len(top[f]["block_list"]), len(rnd[f]["block_list"]))
```

Если ответ с top-30 ни на одном примере не отличается от ответа без маски — починка маскирования не сработала, полный прогон маскирования запускать нет смысла.

### 2.7. Память и время

```python
!grep -i -E "out of memory|error|traceback" logs/test_*.log
!grep "Duration" logs/test_detect.log
!grep "Duration" logs/test_mask_*.log
```

- Если на 30000 не хватает памяти — в полном прогоне уменьшить `--e` (например, до 26000) и использовать то же значение везде.
- По строкам `Duration` оценить время полного прогона: среднее между примером на 1000 и на 30000, умноженное на число примеров (600 для детекции, 200 на каждый запуск маскирования).

### 2.8. Очистка после теста

```python
!rm -rf head_score_test results/dump_test results/dump_test_mask
!rm -rf results/graph/Qwen1.5-14B-Chat results/graph/Qwen1.5-14B-Chat_block_* contexts/Qwen1.5-14B-Chat
!git status --short head_score
```

## 3. Полный прогон: детекция

### 3.1. Проверка перед запуском

```python
import glob, os, subprocess
assert not os.path.exists("head_score_ours/Qwen1.5-14B-Chat.json"), "наши скоры уже есть: повторный запуск допишет в них второй раз"
assert not os.path.exists("results/dump/Qwen1.5-14B-Chat/detect"), "выгрузка детекции уже есть"
assert not glob.glob("results/graph/Qwen1.5-14B-Chat*"), "остались результаты прошлых прогонов"
assert subprocess.run(["git", "status", "--short", "head_score"], capture_output=True, text=True).stdout == "", "файлы авторов изменены"
print("можно запускать")
```

### 3.2. Детекция: 3 иглы × 20 длин × 10 глубин = 600 примеров

```python
!../.venv/bin/python -u retrieval_head_detection.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --dump_dir results/dump \
    --head_score_dir head_score_ours \
    2>&1 | tee logs/detect.log
```

`--depths` не задаётся: по умолчанию 10 глубин от 0 до 100, как у авторов.

Если прогон упал на середине, `head_score_ours/` не появится (скрипт пишет его в самом конце). Скоры по уже пройденным примерам восстанавливаются из выгрузки ячейкой 3.4.

### 3.3. Убрать результаты детекции из-под маскирования

```python
!mv results/graph/Qwen1.5-14B-Chat results/graph/Qwen1.5-14B-Chat_detect
```

### 3.4. Проверка выгрузки и сравнение с авторами

```python
!../.venv/bin/python check_dump.py results/dump/Qwen1.5-14B-Chat/detect \
    --ref head_score/Qwen1.5-14B-Chat.json \
    --out head_score_ours/Qwen1.5-14B-Chat_from_dump.json \
    2>&1 | tee logs/check_dump.log
```

## 4. Полный прогон: маскирование

20 длин × 10 глубин = 200 примеров на запуск. Головы берутся из нашей детекции. Seed не задаётся, как в оригинале; какие случайные головы были выбраны на каждом примере, записано в выгрузке.

### 4.0. Какие запуски нужны

В статье число маскируемых голов «постепенно увеличивали», но явно названо только K = 50; в README авторов пример — 30 лучших против 30 случайных.

| Набор | Запуски | Число запусков |
|---|---|---|
| Обязательный (как в README) | без маски, top-30, random-30 | 3 |
| Точка из статьи | top-50, random-50 | 2 |
| Кривая | top-10, top-20, top-100 и random-10, random-20, random-100 | 6 |

Оценка времени — **не измерена**, по расчёту: 30–45 минут на запуск маскирования, около 2–3 часов на детекцию. Тогда обязательный набор — около 2 часов, все 11 запусков — 6–8 часов. Точные цифры даст ячейка 2.7.

### 4.1. Без маски

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 0 \
    2>&1 | tee logs/mask_0.log
```

### 4.2. Top-30

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 30 \
    2>&1 | tee logs/mask_top30.log
```

### 4.3. Random-30

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk -30 \
    2>&1 | tee logs/mask_random30.log
```

### 4.4. Top-50 (точка из статьи)

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 50 \
    2>&1 | tee logs/mask_top50.log
```

### 4.5. Random-50

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk -50 \
    2>&1 | tee logs/mask_random50.log
```

### 4.6. Кривая, если хватает времени

Top-10:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 10 \
    2>&1 | tee logs/mask_top10.log
```

Top-20:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 20 \
    2>&1 | tee logs/mask_top20.log
```

Top-100:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk 100 \
    2>&1 | tee logs/mask_top100.log
```

Random-10:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk -10 \
    2>&1 | tee logs/mask_random10.log
```

Random-20:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk -20 \
    2>&1 | tee logs/mask_random20.log
```

Random-100:

```python
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --head_score_dir head_score_ours \
    --dump_dir results/dump_mask \
    --mask_topk -100 \
    2>&1 | tee logs/mask_random100.log
```

### 4.7. Сводка

```python
import glob, json, numpy as np

for folder in sorted(glob.glob("results/graph/Qwen1.5-14B-Chat*")):
    scores = [json.load(open(f))["score"] for f in glob.glob(f"{folder}/*_results.json")]
    print(f"{folder.split('/')[-1]:45s} примеров {len(scores):4d}  средний ROUGE {np.mean(scores):6.2f}")
```

Папки: `Qwen1.5-14B-Chat` (без маски), `Qwen1.5-14B-Chat_block_top30`, `Qwen1.5-14B-Chat_block_random30` и так далее, плюс `Qwen1.5-14B-Chat_detect`. В `_detect` файлов 200, а не 600: три иглы пишут результаты под одинаковыми именами, остаётся последняя. Полные данные детекции — в выгрузке.

## 5. Сохранение

```python
!git status --short head_score
!tar -czf baseline_Qwen1.5-14B-Chat.tar.gz head_score_ours results/dump results/dump_mask results/graph/Qwen1.5-14B-Chat* logs
!ls -lh baseline_Qwen1.5-14B-Chat.tar.gz
```

Первая команда должна ничего не напечатать. Архив скачать или переложить в постоянное хранилище: в git эти папки не попадают.

## 6. Известные особенности

- Случайные головы скрипт авторов выбирает заново на каждом примере, из диапазона 32 × 32 при модели 40 × 40, и они могут совпасть с top-головами.
- Повторный запуск с тем же `--mask_topk` перезаписывает результаты и выгрузку в той же папке. Чтобы сделать второй прогон случайных голов, добавить `--random_seed 1`: seed попадёт в имя папки.
- Скрипт маскирования берёт не больше 100 лучших голов, поэтому `--mask_topk` больше 100 смысла не имеет.
- Каждый запуск загружает модель заново.
