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

### 8.3. Что сохраняется и сколько это занимает

`--save` задаёт, что прогон пишет на каждый шаг генерации:

| Режим | Что пишется | Объём на шаг, Qwen3-8B на 30000 токенов |
|---|---|---|
| `compact` (по умолчанию) | 5 лучших позиций внимания каждой головы и доля внимания на каждый размеченный фрагмент | около 50 КБ |
| `rows` | то же плюс вся строка внимания каждой головы | около 70 МБ |
| `none` | только сгенерированные токены | байты |

Ответы модели от `--save` не зависят: при любом режиме каждый ответ пишется в `<out>/outputs/<пример>.json` и никем не удаляется (см. 8.4).

`--buffer_gb` (по умолчанию 2) — сколько непрочитанных данных может лежать в `spool`. Когда лимит достигнут, прогон ждёт, пока метрики прочитают и удалят файлы. `--buffer_gb 0` снимает лимит: всё остаётся на диске, метрики можно посчитать потом.

В режиме `rows` ответ в 50 токенов на 30000 токенов контекста — это около 3,5 ГБ на пример; ответ в 2000 токенов — около 140 ГБ. С окном это помещается на диск, но время прогона определяется скоростью записи. Для длинных ответов режим по умолчанию — `compact`.

### 8.4. Метрики отдельно от прогона

`--with_metrics` только запускает второй процесс. То же вручную, в любой момент и сколько угодно раз:

```python
!cd .. && .venv_new/bin/python -m rh.metrics results/new/qwen3_detect/spool --out results/new/qwen3_detect
```

- `--follow` — читать, пока прогон не закончится;
- `--consume` — удалять прочитанное из `spool`. Без этого флага файлы остаются, и прогон с лимитом буфера остановится в ожидании.

Результаты в папке `--out`:

| Файл | Что в нём |
|---|---|
| `outputs/<пример>.json` | пишет сам прогон: ответ в исходном виде и всё, что нужно для его оценки |
| `samples.jsonl` | по строке на пример: длина, глубина, ответ, ROUGE, успех |
| `heads/<пример>.npz` | метрики голов на этом примере |
| `summary.json` | сводка: доля успешных, лучшие головы, сходство метрик |
| `head_score_<метрика>.json` | скоры голов по успешным примерам, в формате авторов |

Если прогон шёл с маской и с сохранением внимания, файлы скоров называются `head_score_masked_<метрика>.json`. Они описывают модель с отключёнными головами и для выбора голов не годятся: `rh.run` откажется принять такой файл в `--mask_file`. При `--save none` скоры голов не считаются вовсе.

**Ответы хранятся отдельно и не удаляются.** В `outputs/<пример>.json` лежат: токены ответа (`output_ids`), текст со служебными токенами (`text`) и без них (`response`), текст под каждым лимитом длины (`responses_at`), эталон, параметры примера, признак «ответ закончился сам» (`stopped`), число токенов и время. Метрики ответов (`samples.jsonl`) считаются из этих файлов. При лимите 256 токенов файл занимает несколько килобайт, прогон на 200 примеров — около мегабайта.

Пересчёт метрик ответов после правки `answer_metrics` в `rh/metrics.py` или добавления новой — без модели и без `spool`:

```python
!cd .. && .venv_new/bin/python -m rh.metrics results/new/qwen3_mask/top20/spool --out results/new/qwen3_mask/top20 --rescore
```

Переписываются строки `samples.jsonl` и `summary.json`; метрики голов не трогаются. У прогонов, сделанных до этого изменения, папки `outputs` нет — их строки остаются как были.

Метрики голов: `copy_count` — скор как в коде авторов, `copy_recall` — как в статье, `needle_mass` — средняя доля внимания на ответ в игле по всем шагам ответа, `needle_attention_mass` — то же, но только по шагам, где сгенерированный токен есть в ответе иглы (метрика ветки `c0`, см. `METRIC_ATTENTION_MASS.md`). На каждую метрику пишется свой `head_score_<метрика>.json`; любой из них можно передать в `--mask_file`.

### 8.5. Если прогон прервался

Достаточно запустить ту же команду ещё раз, с тем же `--out`. Единица возобновления — пример:

- примеры, результат которых уже есть в `samples.jsonl`, пропускаются;
- примеры, которые сгенерированы, но ещё не прочитаны метриками, не генерируются заново — метрики дочитают их из `spool`;
- пример, на котором прогон оборвался, удаляется из `spool` и генерируется с начала: продолжить генерацию с середины нельзя, кэш модели не сохраняется;
- пример, на котором оборвались метрики, тоже генерируется заново: прочитанные части уже удалены, а накопленное состояние метрик было только в памяти.

У каждого прогона своя папка `--out`. Имена примеров повторяются между прогонами (`n0_len1000_d0` есть и в детекции, и в маскировании), поэтому запуск с другими настройками в уже занятую папку пропустил бы все примеры как готовые. Теперь такой запуск отклоняется сразу, до загрузки модели, с перечнем отличающихся настроек. Те же настройки — это продолжение прогона, оно разрешено.

Одну папку `--out` должен писать один прогон. Второй одновременный прогон в ту же папку примет файлы первого за остатки прерванного и удалит их.

Метрики с `--follow` останавливаются сами, если прогон завершился с ошибкой, а при жёстком обрыве — через час без новых данных (`--idle_timeout`).

### 8.6. Сверка конвейера с кодом, прошедшим сверку, на малом тесте Qwen3-8B

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

Конвейер — ячейка 8.1 (`--out results/new/qwen3_small`) с добавленным `--max_new_tokens 50`: проверенный код генерирует не больше 50 токенов, а у конвейера предел по умолчанию теперь 256.

Сравнение:

```python
!cd .. && .venv_new/bin/python -m rh.verify.compare_pipeline \
    results/new/qwen3_test_old \
    results/new/qwen3_small
```

Ожидается: длины промптов, ответы и ROUGE совпадают в 18 из 18; разница метрик голов 0,000000; лучшие головы совпадают полностью.

Прежний прогон `results/new/qwen3_test` для этого сравнения не годится: он сделан до того, как порядок чтения файлов текста был закреплён сортировкой, и его контексты могут отличаться.

### 8.7. Лимит длины ответа: один прогон, все лимиты

Прогон генерирует до конца ответа, но не дольше `--max_new_tokens` (по умолчанию 256). Метрики попутно считают всё так, как если бы генерацию обрезали на каждом из лимитов `--limits` (по умолчанию 50, 64, 128, 256): ответ, ROUGE, признак обрезки и метрики голов по первым N токенам. Отдельных прогонов под каждый лимит не нужно.

Ячейки запуска менять не требуется, это поведение по умолчанию.

Таблица по лимитам, без GPU и без модели:

```python
!cd .. && .venv_new/bin/python -m rh.limits results/new/qwen3_detect
```

Столбцы: `truncated` — доля ответов, не закончившихся к этому лимиту; `success` — доля успешных; `rouge` — средний ROUGE; дальше — число голов выше 0,1 и сходство порядка голов с самым большим лимитом (корреляция и пересечение лучших 10, 20, 50). Сравнить с другим лимитом: `--reference 50`.

Выбор лимита — записывает `head_score_<метрика>@<лимит>.json`, этот файл и передаётся в `--mask_file`:

```python
!cd .. && .venv_new/bin/python -m rh.limits results/new/qwen3_detect --select 128
```

Для прогонов с маской (`--save none`) та же команда даёт таблицу по ответам, без столбцов про головы.

На что смотреть при выборе: наименьший лимит, при котором `truncated` близко к нулю, а порядок голов уже не отличается от самого большого. Число ответов, упёршихся в предел 256, есть в сводке прогона (`truncated_at_cap`).

### 8.8. Маскирование

Запускается после детекции (8.2): головы берутся из готового файла `results/new/qwen3_detect/head_score_copy_count.json`.

Это отдельные прогоны на другой игле (`data/needles_eval.jsonl`, про Сан-Франциско), 20 длин × 10 глубин = 200 примеров на запуск. Внимание не сохраняется (`--save none`), считаются только ответы.

**Куда ложатся результаты.** Все прогоны с маской лежат в одной папке, по подпапке на прогон, отдельно от детекции:

```
results/new/qwen3_detect/          детекция: скоры голов (уже посчитана, сюда ничего не пишем)
results/new/qwen3_mask/none/       без маски — точка отсчёта
results/new/qwen3_mask/top20/      20 лучших голов
results/new/qwen3_mask/random20/   20 случайных россыпью
results/new/qwen3_mask/groups20/   20 случайных целыми группами
...
```

Запуск в занятую папку с другими настройками отклоняется сразу, с перечнем отличий; с теми же настройками — продолжает прерванный прогон.

**Число голов:** 20, 40, 60, 80, 100, 120 — от 1,7% до 10,4% из 1152 голов Qwen3-8B. Все значения кратны 4, размеру группы голов с общими ключами и значениями.

| Вид маски | Аргумент | Какие головы |
|---|---|---|
| без маски | — | никакие |
| лучшие | `--mask_top K` | первые K голов по скору детекции |
| случайные россыпью | `--mask_random K` | K случайных голов вне 100 лучших, по всей модели |
| случайные группами | `--mask_random_groups K` | K/4 случайных групп целиком, без групп, где есть лучшие головы |

Всего 19 запусков. Если времени мало, сначала без маски и 20, 60, 120 голов — это 10 запусков. Случайные головы выбираются один раз на запуск, `--seed 0` по умолчанию; для второго набора — другой `--seed` и другая подпапка, например `random20_s1`.

**Сводная таблица по всем запускам**, без GPU:

```python
!cd .. && .venv_new/bin/python -m rh.sweep results/new/qwen3_mask/*
```

Столбцы: вид маски, число голов и затронутых групп, предел ответа, доля успешных, средний ROUGE, доля обрезанных ответов. С `--limit 128` ответы оцениваются обрезанными на этом лимите.

#### Всё одной командой, модель загружается один раз

Основной способ. Проходит все 19 конфигураций подряд и раскладывает результаты по тем же подпапкам, что и отдельные ячейки ниже.

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.mask_sweep \
    --model_path Qwen/Qwen3-8B \
    --needles data/needles_eval.jsonl \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json \
    --s_len 1000 \
    --e_len 30000 \
    --context_intervals 20 \
    --out results/new/qwen3_mask \
    2>&1 | tee source/logs/mask_sweep_qwen3.log
```

Сокращённый набор — добавить `--counts 20,60,120`; только часть видов маски — `--kinds none,top,random` (виды: `none`, `top`, `random`, `groups`). Второй набор случайных голов — `--seed 1 --kinds random,groups`: подпапки получат окончание `_s1`.

Что происходит при сбоях:

- **Неверные аргументы или занятая папка с другими настройками** — отказ сразу, до загрузки модели.
- **Ошибка в одной конфигурации** — она помечается как `failed` с текстом ошибки, остальные продолжают считаться.
- **В конце печатается сводка** по каждой конфигурации: `done`, `failed`, `incomplete` (не все примеры получили результат), `interrupted`. Если что-то не завершено, последняя строка — `NOT FINISHED: …`, и команда завершается с ошибкой.
- **Состояние лежит в файле** `results/new/qwen3_mask/sweep_status.json`: его можно посмотреть и во время прогона.
- **Повторный запуск той же команды** пропускает готовое и продолжает оборванное с того примера, на котором остановилось.

Посмотреть состояние, не дожидаясь конца:

```python
!cat ../results/new/qwen3_mask/sweep_status.json
```

Ячейки ниже делают то же по одной конфигурации; они нужны, если хочется запустить или перезапустить что-то отдельно.

#### Без маски

Запускать первым: с ним сравниваются все остальные.

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
    --out results/new/qwen3_mask/none \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_none.log
```

#### Лучшие головы

20 голов:

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
    --mask_top 20 \
    --out results/new/qwen3_mask/top20 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top20.log
```

40 голов:

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
    --mask_top 40 \
    --out results/new/qwen3_mask/top40 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top40.log
```

60 голов:

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
    --out results/new/qwen3_mask/top60 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top60.log
```

80 голов:

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
    --mask_top 80 \
    --out results/new/qwen3_mask/top80 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top80.log
```

100 голов:

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
    --mask_top 100 \
    --out results/new/qwen3_mask/top100 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top100.log
```

120 голов:

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
    --mask_top 120 \
    --out results/new/qwen3_mask/top120 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_top120.log
```

#### Случайные россыпью

20 голов:

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
    --mask_random 20 \
    --out results/new/qwen3_mask/random20 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random20.log
```

40 голов:

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
    --mask_random 40 \
    --out results/new/qwen3_mask/random40 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random40.log
```

60 голов:

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
    --out results/new/qwen3_mask/random60 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random60.log
```

80 голов:

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
    --mask_random 80 \
    --out results/new/qwen3_mask/random80 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random80.log
```

100 голов:

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
    --mask_random 100 \
    --out results/new/qwen3_mask/random100 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random100.log
```

120 голов:

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
    --mask_random 120 \
    --out results/new/qwen3_mask/random120 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_random120.log
```

#### Случайные группами

20 голов:

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
    --mask_random_groups 20 \
    --out results/new/qwen3_mask/groups20 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups20.log
```

40 голов:

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
    --mask_random_groups 40 \
    --out results/new/qwen3_mask/groups40 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups40.log
```

60 голов:

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
    --mask_random_groups 60 \
    --out results/new/qwen3_mask/groups60 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups60.log
```

80 голов:

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
    --mask_random_groups 80 \
    --out results/new/qwen3_mask/groups80 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups80.log
```

100 голов:

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
    --mask_random_groups 100 \
    --out results/new/qwen3_mask/groups100 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups100.log
```

120 голов:

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
    --mask_random_groups 120 \
    --out results/new/qwen3_mask/groups120 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_groups120.log
```

### 8.9. Маска 20 лучших голов по другой метрике: масса внимания на игле

Рейтинг голов берётся из `data/head_scores/head_scores_needle_attention_mass_v1.json` (файл Михаила: 1152 головы Qwen3-8B, 27 примеров на голову). Задача, длины и лимит ответа те же, что в 8.8, меняется только рейтинг. Результат пишется в отдельную папку, чтобы не смешиваться с `results/new/qwen3_mask/top20`, где 20 лучших по `copy_count`.

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
    --mask_file data/head_scores/head_scores_needle_attention_mass_v1.json \
    --mask_top 20 \
    --out results/new/qwen3_mask_attention_mass/top20 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_attention_mass_top20.log
```

Сравнение с прогоном без маски и с 20 лучшими по `copy_count`:

```python
!cd .. && .venv_new/bin/python -m rh.sweep \
    results/new/qwen3_mask/none \
    results/new/qwen3_mask/top20 \
    results/new/qwen3_mask_attention_mass/top20
```

### 8.10. То же для второго файла метрики

Рейтинг из `data/head_scores/head_scores_needle_attention_mass_v1_2.json` (второй файл Михаила с тем же исходным именем, значения другие). Всё остальное как в 8.9, папка результата своя.

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
    --mask_file data/head_scores/head_scores_needle_attention_mass_v1_2.json \
    --mask_top 20 \
    --out results/new/qwen3_mask_attention_mass_2/top20 \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_mask_attention_mass_2_top20.log
```

Сравнение всех четырёх прогонов:

```python
!cd .. && .venv_new/bin/python -m rh.sweep \
    results/new/qwen3_mask/none \
    results/new/qwen3_mask/top20 \
    results/new/qwen3_mask_attention_mass/top20 \
    results/new/qwen3_mask_attention_mass_2/top20
```

### 8.11. Усложнённые задачи: `hard_niah`

Данные и описание уровней — в `data/needles_hard/README.md`. Задача строит промпт по семейству (`lisbon`, `cairo`, `oslo`) и уровню (`A0`, `B3`, `F1_before_50`, …): ставит иглу и вставки уровня, размечает отрезок на каждую вставку.

- `--levels` — уровни через запятую, целая ось (`B`) или `all`; по умолчанию `A0`;
- `--families` — семейства через запятую; по умолчанию все;
- `--insert_seed` — seed расположения вставок; `--seed` остаётся за случайными головами маски;
- `--lengths` и `--depths` — длины контекста и глубины иглы через запятую. Берём сетку перебора масок на простой задаче, прореженную вдвое по обеим осям, с сохранением последней точки: 11 длин (каждая вторая плюс 30000) и 6 глубин (0, 22, 44, 67, 89 плюс 100). Все точки есть в старой сетке, поэтому уровни сравнимы с простой задачей по отдельным клеткам.

Один прогон может держать несколько уровней и семейств: они записаны в имени примера и в `samples.jsonl` (поля `family`, `level`). Примеры, где вставка не помещается на заданном расстоянии (уровни F1), пропускаются, их число печатается в конце.

Первый прогон — базовый уровень без маски: 3 семейства × 11 длин × 6 глубин = 198 примеров. По нему видно, отвечает ли модель на A0 обоими фактами.

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task hard_niah \
    --levels A0 \
    --lengths 1000,4053,7105,10158,13211,16263,19316,22368,25421,28474,30000 \
    --depths 0,22,44,67,89,100 \
    --out results/new/qwen3_hard/A0/none \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_hard_A0_none.log
```

Целая ось без маски, с сохранением внимания (ось B: 4 уровня × 3 семейства × 66 = 792 примера):

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task hard_niah \
    --levels B \
    --lengths 1000,4053,7105,10158,13211,16263,19316,22368,25421,28474,30000 \
    --depths 0,22,44,67,89,100 \
    --out results/new/qwen3_hard/B/none \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_hard_B_none.log
```

Та же ось с маской 60 лучших голов по старому рейтингу; внимание не сохраняется:

```python
!mkdir -p logs
!cd .. && .venv_new/bin/python -u -m rh.run \
    --model_path Qwen/Qwen3-8B \
    --task hard_niah \
    --levels B \
    --lengths 1000,4053,7105,10158,13211,16263,19316,22368,25421,28474,30000 \
    --depths 0,22,44,67,89,100 \
    --save none \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json \
    --mask_top 60 \
    --out results/new/qwen3_hard/B/top60_copy \
    --with_metrics \
    2>&1 | tee source/logs/run_qwen3_hard_B_top60_copy.log
```

По новому рейтингу — та же ячейка с `--mask_file data/head_scores/head_scores_needle_attention_mass_v1_2.json` и `--out results/new/qwen3_hard/B/top60_mass`.

**Внимание на вставках.** В прогоне `hard_niah` с сохранением внимания (без `--save none`) для каждого примера, кроме обычных метрик голов, считаются:

| Метрика | Что это |
|---|---|
| `needle_sentence_mass` | доля внимания головы на всю иглу (`needle_mass` — только на ответ внутри неё) |
| `competing_mass` | доля внимания на все конкурирующие вставки вместе: близнецы, приманки, шум |
| `competing_max_mass` | доля внимания на самую сильную из них на шаге: одна вставка против одной иглы |
| `supporting_mass` | доля внимания на вставки с частью ответа (уровни D2, D3, E3) |
| `copy_sentence` | копирование из иглы целиком |
| `copy_competing` | копирование из конкурирующих вставок |

Обе метрики копирования делятся на длину ответа в игле, поэтому сравнимы между собой и с `copy_count`. Массы усредняются по всем шагам ответа.

Таблица по уровням для заданного набора голов — без модели, по готовым результатам:

```python
!cd .. && .venv_new/bin/python -m rh.levels results/new/qwen3_hard/B/none \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json --top 20
```

Столбцы: `copy` — копирование ответа из иглы, `needle` и `compet` — внимание на иглу и на конкурентов, `max` — на самого сильного конкурента, `share` = `needle / (needle + compet)`, `copy_c` — копирование из конкурентов, `all` — внимание на иглу в среднем по всем головам. Для второго рейтинга — та же команда с другим `--mask_file`. Без `--mask_file` усреднение идёт по всем головам.

Что пока не готово: `summary.json`, `rh.sweep` и ноутбук визуализации усредняют по всему прогону и уровни не разделяют; успех считается по ROUGE, который на осях B и D непригоден. `rh.levels` поэтому берёт все примеры, а не только успешные. Ответы сохраняются в `outputs/`, класс ответа по фактам можно посчитать после прогонов.

## 9. Известные особенности

- Перед первым запуском нужна папка `logs/`: без неё `tee` завершается с ошибкой, и ячейка падает уже после прогона.
- Игла вставляется не на заданной глубине. При `--model_provider` по умолчанию конец предложения ищется по id точек из словаря Llama; в тесте на 1000 токенов игла на глубинах 0 и 50 оказалась в позиции 0 (`insertion at 0` в логе). Фактическая позиция пишется в выгрузку (`needle_start`).
- Две наши правки ради памяти: контекст прогоняется без расчёта логитов, а кэш при генерации дополняется на месте, без копий. Без них на 30000 токенов не хватает 80 ГБ. На результат не влияют.

- Случайные головы скрипт авторов выбирает заново на каждом примере, из диапазона 32 × 32 при модели 40 × 40, и они могут совпасть с top-головами.
- Повторный запуск с тем же `--mask_topk` перезаписывает результаты и выгрузку в той же папке. Чтобы сделать второй прогон случайных голов, добавить `--random_seed 1`: seed попадёт в имя папки.
- Скрипт маскирования берёт не больше 100 лучших голов, поэтому `--mask_topk` больше 100 смысла не имеет.
- Каждый запуск загружает модель заново.
