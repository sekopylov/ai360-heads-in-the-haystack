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
!mkdir -p logs
```

```python
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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

Прямая проверка, если по ответам неясно: один шаг генерации с маской и без, в одном процессе и на одном входе. Читает файл авторов, ничего не пишет.

```python
!../.venv/bin/python check_mask.py --model_path Qwen/Qwen1.5-14B-Chat
```

Последняя строка — `MASKING WORKS` или `MASKING HAS NO EFFECT`.

### 2.7. Память и время

```python
!grep -i -E "out of memory|error|traceback" logs/test_*.log || true
!grep -E "Duration|Context|insertion at" logs/test_detect.log || true
!grep "Duration" logs/test_mask_*.log || true
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
os.makedirs("logs", exist_ok=True)
assert not os.path.exists("head_score_ours/Qwen1.5-14B-Chat.json"), "наши скоры уже есть: повторный запуск допишет в них второй раз"
assert not os.path.exists("results/dump/Qwen1.5-14B-Chat/detect"), "выгрузка детекции уже есть"
assert not glob.glob("results/graph/Qwen1.5-14B-Chat*"), "остались результаты прошлых прогонов"
assert subprocess.run(["git", "status", "--short", "head_score"], capture_output=True, text=True).stdout == "", "файлы авторов изменены"
print("можно запускать")
```

### 3.2. Детекция: 3 иглы × 20 длин × 10 глубин = 600 примеров

```python
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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
!mkdir -p logs
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

## 6. Второй прогон: правильная вставка иглы

Основной прогон (разделы 3–4) повторяет запуск авторов, где игла на Qwen вставляется не на заданной глубине. Этот прогон — те же данные с флагом `--correct_insertion`: граница предложения ищется по словарю самой модели. Всё пишется в отдельные папки, основной прогон не затрагивается.

Детекция с этим флагом сама пишет результаты в `results/graph/Qwen1.5-14B-Chat_detect_insfix`, переименовывать ничего не нужно. Запускать можно в любой момент после основной детекции, в том числе после маскирования.

### 6.1. Детекция

```python
!mkdir -p logs
!../.venv/bin/python -u retrieval_head_detection.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --correct_insertion \
    --dump_dir results/dump_insfix \
    --head_score_dir head_score_ours_insfix \
    2>&1 | tee logs/detect_insfix.log
```

### 6.2. Сравнение двух вставок

```python
!../.venv/bin/python check_dump.py results/dump_insfix/Qwen1.5-14B-Chat/detect \
    --ref head_score_ours/Qwen1.5-14B-Chat.json \
    2>&1 | tee logs/check_dump_insfix.log
```

Строка `ours vs ref` здесь — сходство скоров при правильной вставке со скорами основного прогона.

### 6.3. Маскирование

Любая ячейка из раздела 4 с тремя заменами: добавить `--correct_insertion`, взять головы из `head_score_ours_insfix`, писать выгрузку в `results/dump_mask_insfix`. Папки результатов получают окончание `_insfix`. Пример для top-30:

```python
!mkdir -p logs
!../.venv/bin/python -u needle_in_haystack_with_mask.py \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --s 1000 \
    --e 30000 \
    --context-intervals 20 \
    --correct_insertion \
    --head_score_dir head_score_ours_insfix \
    --dump_dir results/dump_mask_insfix \
    --mask_topk 30 \
    2>&1 | tee logs/mask_top30_insfix.log
```

В архив из раздела 5 добавить `head_score_ours_insfix results/dump_insfix results/dump_mask_insfix`.

## 7. Новый код: сверка со старым на первой игле

Новый код лежит в `rh/` и требует свежий `transformers`, поэтому ему нужно отдельное окружение; старое (`.venv`) не трогаем. Сверка идёт на уже посчитанной выгрузке старого кода, заново старый код запускать не нужно.

### 7.1. Окружение

```python
!cd .. && python3 -m venv .venv_new && .venv_new/bin/pip install -q -r requirements-new.txt
!cd .. && .venv_new/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

Порядок: сначала малый тест на Qwen3-8B (7.2) — что новый код вообще работает на новой модели; затем сверка со старым кодом на части уже посчитанных данных Qwen1.5-14B-Chat (7.3–7.5). Если сверка сходится, новый код считаем валидным и дальше работаем на нём.

### 7.2. Малый тест на Qwen3-8B

3 иглы × 2 длины × 3 глубины = 18 примеров. Без `--legacy`: игла ставится на границе предложения, её положение ищется точно, генерация останавливается на конце ответа.

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.verify.detect \
    --model_path Qwen/Qwen3-8B \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 2 \
    --depths 0,50,100 \
    --out results/new/qwen3_test \
    2>&1 | tee source/logs/new_qwen3_test.log
```

```python
!cd .. && .venv_new/bin/python source/check_dump.py results/new/qwen3_test
```

Что смотреть: прогон не падает на 30000 токенов; `needle not found: 0`; есть успешные примеры; ответы в логе осмысленные и без `<think>`; в сводке есть головы со скором выше 0,1.

### 7.3. Входы Qwen1.5-14B-Chat: модель не загружается, GPU не нужен

Новый код заново собирает контексты и промпты и сравнивает их с тем, что подал в модель старый код.

```python
!cd .. && .venv_new/bin/python -m rh.verify.detect \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --legacy \
    --inputs_only \
    --ref_dump source/results/dump/Qwen1.5-14B-Chat/detect \
    --needle 0
```

Ожидается `inputs identical to the old code: 200/200`.

### 7.4. Модель: повтор входов старого кода, первая игла

Входы берутся из выгрузки старого кода, так что проверяется только модельная часть: перехват внимания, генерация, подсчёт скора. Сначала малая часть: каждый пятый пример первой иглы, 40 штук на всех длинах от 1000 до 30000:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.verify.detect \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --legacy \
    --replay source/results/dump/Qwen1.5-14B-Chat/detect \
    --needle 0 \
    --every 5 \
    --out results/new/replay_needle0 \
    2>&1 | tee source/logs/new_replay_test.log
```

Сравнить (ячейка 7.5). Если сошлось, этого достаточно; при желании — все 200 примеров первой иглы:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.verify.detect \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --legacy \
    --replay source/results/dump/Qwen1.5-14B-Chat/detect \
    --needle 0 \
    --out results/new/replay_needle0 \
    2>&1 | tee source/logs/new_replay.log
```

### 7.5. Сравнение

```python
!cd .. && .venv_new/bin/python -m rh.verify.compare_dumps \
    source/results/dump/Qwen1.5-14B-Chat/detect \
    results/new/replay_needle0
```

Что смотреть:

- `inputs identical` — должны совпасть все (входы взяты из выгрузки);
- `generated tokens identical` — ожидается почти все; отдельные расхождения возможны из-за разных реализаций быстрого внимания в старом и новом коде;
- `top-1 attention position identical` — доля совпавших позиций, ожидается выше 0,99;
- `head scores ... spearman` и `top20 overlap` — пороги из PLAN.md, раздел 5: корреляция не ниже 0,95, пересечение top-20 не ниже 18.

Выгрузка нового кода в том же формате, что у старого, поэтому `check_dump.py` работает и на ней.

### 7.6. Уровень численного шума (по желанию)

Тот же повтор, но контекст считается другим ядром внимания PyTorch. Код и библиотеки те же, меняется только порядок вычислений в половинной точности. Сравнение двух прогонов нового кода между собой показывает, какие расхождения даёт один лишь шум.

```python
!mkdir -p logs
!cd .. && HF_HUB_CACHE="$TRANSFORMERS_CACHE" .venv_new/bin/python -u -m rh.verify.detect \
    --model_path Qwen/Qwen1.5-14B-Chat \
    --legacy \
    --replay source/results/dump/Qwen1.5-14B-Chat/detect \
    --needle 0 \
    --every 5 \
    --no_flash_sdp \
    --out results/new/replay_needle0_noflash \
    2>&1 | tee source/logs/new_replay_noflash.log
```

```python
!cd .. && .venv_new/bin/python -m rh.verify.compare_dumps \
    results/new/replay_needle0 \
    results/new/replay_needle0_noflash
```

Если цифры близки к сравнению со старым кодом (совпадение позиций около 0,97, ответы идентичны в 60–70% примеров), расхождение старого и нового кода — шум половинной точности. Если новый код сам с собой совпадает почти полностью, расхождение со старым кодом имеет другую причину.

Если прогон упадёт по памяти на длинных контекстах, добавить `--limit 20`: другое ядро может требовать больше памяти.

## 8. Новый конвейер: прогон и метрики

Основной код для экспериментов. Прогон (`rh.run`) и метрики (`rh.metrics`) — отдельные процессы, общаются только через файлы в папке `spool`. Решения авторов, признанные ошибочными, здесь не используются: игла ставится на границе предложения, её положение точное, генерация останавливается на конце ответа.

Все команды — из ноутбука в `source/`, запуск идёт из корня репозитория (`cd ..`). Для Qwen1.5-14B-Chat перед `.venv_new/bin/python` добавить `HF_HUB_CACHE="$TRANSFORMERS_CACHE"`.

### 8.1. Малый тест

3 иглы × 2 длины × 3 глубины = 18 примеров.

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task niah \
    --needles data/needles_detect.jsonl \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 2 \
    --depths 0,50,100 \
    --out results/new/qwen3_small \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_small.log
```

В конце печатается сводка; она же лежит в `results/new/qwen3_small/summary.json`.

### 8.2. Детекция

3 иглы × 20 длин × 10 глубин = 600 примеров.

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task niah \
    --needles data/needles_detect.jsonl \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 20 \
    --out results/new/qwen3_detect \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_detect.log
```

Если прогон прервался, та же команда продолжит с места остановки: примеры, которые уже есть в `samples.jsonl`, пропускаются.

### 8.3. Маскирование

Оценочная игла (`data/needles_eval.jsonl`), 20 длин × 10 глубин = 200 примеров на запуск. Внимание не сохраняется (`--save none`), только ответы. Головы берутся из детекции.

Без маски:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task niah \
    --needles data/needles_eval.jsonl \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 20 \
    --save none \
    --out results/new/qwen3_mask_none \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_none.log
```

Лучшие головы:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task niah \
    --needles data/needles_eval.jsonl \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 20 \
    --save none \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json \
    --mask_top 60 \
    --out results/new/qwen3_mask_top60 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top60.log
```

Случайные головы (вне 100 лучших, одни и те же на весь прогон):

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task niah \
    --needles data/needles_eval.jsonl \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 20 \
    --save none \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json \
    --mask_random 60 \
    --seed 0 \
    --out results/new/qwen3_mask_random60 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random60.log
```

Число голов задаётся в штуках, а сравнивать модели нужно в долях: у Qwen3-8B 1152 головы, 60 — это около 5%; у Qwen1.5-14B-Chat 1600 голов, 5% — это 80. Для каждого значения и каждого seed нужна своя папка `--out`.

### 8.4. Что сохраняется и сколько это занимает

`--save` задаёт, что прогон пишет на каждый шаг генерации:

| Режим | Что пишется | Объём на шаг, Qwen3-8B на 30000 токенов |
|---|---|---|
| `compact` (по умолчанию) | 5 лучших позиций внимания каждой головы и доля внимания на каждый размеченный фрагмент | около 50 КБ |
| `rows` | то же плюс вся строка внимания каждой головы | около 70 МБ |
| `none` | только сгенерированные токены | байты |

`--buffer_gb` (по умолчанию 2) — сколько непрочитанных данных может лежать в `spool`. Когда лимит достигнут, прогон ждёт, пока метрики прочитают и удалят файлы. `--buffer_gb 0` снимает лимит: всё остаётся на диске, метрики можно посчитать потом.

В режиме `rows` ответ в 50 токенов на 30000 токенов контекста — это около 3,5 ГБ на пример; ответ в 2000 токенов — около 140 ГБ. С окном это помещается на диск, но время прогона определяется скоростью записи. Для длинных ответов режим по умолчанию — `compact`.

### 8.5. Метрики отдельно от прогона

`--with_metrics` только запускает второй процесс. То же вручную, в любой момент и сколько угодно раз:

```python
!cd .. && .venv_new/bin/python -m rh.metrics results/new/qwen3_detect/spool --out results/new/qwen3_detect
```

- `--follow` — читать, пока прогон не закончится;
- `--consume` — удалять прочитанное из `spool`. Без этого флага файлы остаются, и прогон с лимитом буфера остановится в ожидании.

Результаты в папке `--out`:

| Файл | Что в нём |
|---|---|
| `samples.jsonl` | по строке на пример: длина, глубина, ответ, ROUGE, успех |
| `heads/<пример>.npz` | метрики голов на этом примере |
| `summary.json` | сводка: доля успешных, лучшие головы, сходство метрик |
| `head_score_<метрика>.json` | скоры голов по успешным примерам, в формате авторов |

Метрики голов: `copy_count` — скор как в коде авторов, `copy_recall` — как в статье, `needle_mass` — средняя доля внимания на ответ в игле.

### 8.6. Если прогон прервался

Достаточно запустить ту же команду ещё раз, с тем же `--out`. Единица возобновления — пример:

- примеры, результат которых уже есть в `samples.jsonl`, пропускаются;
- примеры, которые сгенерированы, но ещё не прочитаны метриками, не генерируются заново — метрики дочитают их из `spool`;
- пример, на котором прогон оборвался, удаляется из `spool` и генерируется с начала: продолжить генерацию с середины нельзя, кэш модели не сохраняется;
- пример, на котором оборвались метрики, тоже генерируется заново: прочитанные части уже удалены, а накопленное состояние метрик было только в памяти.

Одну папку `--out` должен писать один прогон. Второй одновременный прогон в ту же папку примет файлы первого за остатки прерванного и удалит их.

Метрики с `--follow` останавливаются сами, если прогон завершился с ошибкой, а при жёстком обрыве — через час без новых данных (`--idle_timeout`).

### 8.7. Сверка конвейера с кодом, прошедшим сверку, на малом тесте Qwen3-8B

Один и тот же малый тест (18 примеров) двумя способами: проверенным кодом `rh.verify.detect` и конвейером. Оба строят одинаковые входы и считают одним вычислительным путём.

Проверенный код:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.verify.detect \
    --model_path Qwen/Qwen3-8B \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 2 \
    --depths 0,50,100 \
    --out results/new/qwen3_test_old \
    2>&1 | tee source/logs/new_qwen3_test_old.log
```

Конвейер — ячейка 8.1 (`--out results/new/qwen3_small`).

Сравнение:

```python
!cd .. && .venv_new/bin/python -m rh.verify.compare_pipeline \
    results/new/qwen3_test_old \
    results/new/qwen3_small
```

Ожидается: длины промптов, ответы и ROUGE совпадают в 18 из 18; разница метрик голов 0,000000; лучшие головы совпадают полностью.

Прежний прогон `results/new/qwen3_test` для этого сравнения не годится: он сделан до того, как порядок чтения файлов текста был закреплён сортировкой, и его контексты могут отличаться.

## 9. Известные особенности

- Перед первым запуском нужна папка `logs/`: без неё `tee` завершается с ошибкой, и ячейка падает уже после прогона.
- Игла вставляется не на заданной глубине. При `--model_provider` по умолчанию конец предложения ищется по id точек из словаря Llama; в тесте на 1000 токенов игла на глубинах 0 и 50 оказалась в позиции 0 (`insertion at 0` в логе). Фактическая позиция пишется в выгрузку (`needle_start`).
- Две наши правки ради памяти: контекст прогоняется без расчёта логитов, а кэш при генерации дополняется на месте, без копий. Без них на 30000 токенов не хватает 80 ГБ. На результат не влияют.

- Случайные головы скрипт авторов выбирает заново на каждом примере, из диапазона 32 × 32 при модели 40 × 40, и они могут совпасть с top-головами.
- Повторный запуск с тем же `--mask_topk` перезаписывает результаты и выгрузку в той же папке. Чтобы сделать второй прогон случайных голов, добавить `--random_seed 1`: seed попадёт в имя папки.
- Скрипт маскирования берёт не больше 100 лучших голов, поэтому `--mask_topk` больше 100 смысла не имеет.
- Каждый запуск загружает модель заново.
