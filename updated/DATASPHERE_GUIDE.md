# Запуск эксперимента в Yandex DataSphere

Рабочий корень проекта — папка `updated`. Все пути ниже указаны относительно неё.

В репозитории уже есть:

- `datasphere/job.py` — запускает этапы эксперимента по очереди;
- `datasphere/smoke.yaml` — короткая проверка на CPU;
- `datasphere/full.yaml` — полный прогон на одной GPU;
- `requirements.txt` — обычные зависимости и CPU smoke;
- `datasphere/requirements-gpu.txt` — GPU-зависимости с PyTorch 2.6.0 CUDA 11.8.

Все команды ниже выполняются **из папки `updated`**.

Корпуса находятся в `data/haystack_for_detect/` и
`data/PaulGrahamEssays/`; YAML передаёт их в job через `inputs`.

Драйвер компилируется, оба YAML разбираются локально без ошибок. Удалённый
smoke также проверен в указанном проекте: job `bt1mpuoprch2p2dmfsoh` успешно
загрузила Qwen, выполнила все три случая и вернула scores 100, 100 и 94.44.
Несмотря на это, после изменения зависимостей сначала повторяйте smoke, а не
запускайте платный полный прогон.

## Что произойдёт

Локальный `datasphere` CLI загрузит в job код и тексты. На отдельной VM
DataSphere создаст Python 3.12 environment, установит зависимости, а
Transformers скачает `Qwen/Qwen3.5-0.8B` с Hugging Face. Результаты job будут
скачаны обратно в `datasphere-results/`.

Модель можно один раз сохранить в хранилище проекта и использовать между jobs.
Все три YAML подключают диск проекта через `flags: [attach-project-disk]`.
Каждый YAML явно передаёт `--model-search-dir ${DS_PROJECT_HOME}/models`.
Код ищет выбранную модель в `<model-search-dir>/<model_id>`, например
`${DS_PROJECT_HOME}/models/Qwen/Qwen3.5-0.8B`.
В Python нет встроенного пути поиска и автоматического чтения `DS_PROJECT_HOME`.
При отсутствии папки используется обычная загрузка Hugging Face с кешем этой VM.

### Один раз скачать модель в хранилище проекта

Откройте JupyterLab проекта DataSphere и его Terminal. Из корня хранилища
проекта выполните команды ниже. Не определяйте корень только по названию пути:
CPU-проверка 2026-10-07 показала, что подключённый к Jobs диск содержит
`timon/`, `ai360-heads-in-the-haystack/` и другие каталоги, но не `c0/`,
созданный в `/home/jupyter/datasphere/project`. По ранее полученному выводу
ноутбука этому набору соответствует `/home/jupyter/project`.
Для воспроизводимой проверки содержимого диска используйте
`datasphere/storage-probe.yaml`; отчёт сохраняется в
`datasphere-results/storage-probe/report.json`.

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen/Qwen3.5-0.8B
```

GPU для скачивания не нужна. Скачиваются файлы checkpoint и tokenizer,
модель в память не загружается. При повторной команде Hugging Face использует
метаданные локальной загрузки и не скачивает неизменившиеся файлы повторно.
Не помещайте веса в Git. Диск проекта подключается к Jobs для чтения:
fallback скачивает в кеш job, но не сохраняет модель на диск проекта.

### Выбрать модель и папки поиска

Проверка сохранённой Qwen3.5-0.8B отдельной короткой GPU-job:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 -c datasphere/storage-smoke.yaml
```

`storage-smoke.yaml` явно указывает checkpoint в
`${DS_PROJECT_HOME}/c0/models/Qwen/Qwen3.5-0.8B` и включает offline-режим Hugging Face.
При отсутствии модели job завершится ошибкой, скачивания с Hugging Face не будет.
Этот путь задан только в конфигурации теста; при другом расположении измените `--model`.
Тест проверяет CUDA и три detection-кейса длиной 1000, depth 50, без masking.
Результаты находятся в `datasphere-results/storage-smoke/`;
успех загрузки подтверждает `[model] source=local path=...` в логах,
завершение всех кейсов — `detection/run.json` с `complete=true`.

`--model` задаёт Hugging Face ID или прямой путь к папке с checkpoint.
`--adapter` задаёт код для архитектуры: `qwen35` для гибридной Qwen3.5,
`qwen3` для обычной dense Qwen3 (включая Qwen3-4B-Thinking-2507).
Смена папки хранения не требует смены адаптера. Произвольная модель другой
архитектуры потребует отдельного адаптера, одного изменения `--model` недостаточно.

### Smoke для Qwen3-4B-Thinking-2507

#### Онлайн-метрики и несколько метрик за один прогон

Новая `needle_attention_mass_v1`: если сгенерированный **token ID** есть в
needle-span, для каждой головы складываем attention по **всем позициям** span.
Итог головы — среднее этих сумм по подходящим шагам. Повторения учитываются
каждый раз, без multiset-квот; максимальное значение1. При отсутствии подходящих
шагов0. Здесь используются токены tokenizer, не целые слова.
Например, массы0.8/0.4/0.0 на трёх подходящих шагах дают score0.4.

Подсчёт онлайн: GPU передаёт одно FP32 число на голову за шаг. Скоринг хранит
только сумму и счётчик, без истории attention-векторов. Для1152 голов это
несколько KB данных одного шага (Qwen3:1152 ×4 байта для сумм).
Если нужна только новая метрика, top1 не собирается. При нескольких метриках
добавляется top1 для legacy/multiset. Полные traces сохраняются **только** при
явном `--capture full` в detection CLI; для метрики этот режим не нужен.

Для обычной Qwen3-8B доступен `--adapter qwen3_8b`: укажите также
`--model Qwen/Qwen3-8B` либо путь к её локальному checkpoint. Адаптер использует
общую реализацию dense Qwen3 (full attention, cached decode, EOS, отделение thinking),
без включения YaRN. Проверяет до prefill, что весь токенизированный chat prompt
+ `--max-new-tokens` не превышает 32768. Для 30k haystack и max2048
остаётся запас на вопрос/шаблон, но проверяется фактический размер.
На V100 используйте float16 и sdpa_memory_efficient. Thinking и финальный
ответ делят общий лимит генерации; ROUGE по финальному ответу,
retrieval scope задаётся `--attention-scope` как у Thinking-4B.
Значение `--model` по умолчанию в CLI не изменено: задавайте модель явно.

Вариант `--adapter qwen3_8b_yarn` читает тот же checkpoint Qwen3-8B, но при
создании модели передаёт static YaRN (`factor=4`, исходный лимит32768,
max_position_embeddings=131072). Нативный `qwen3_8b` не включает YaRN.
В Transformers5 настройки передаются через `rope_parameters`; config.json
на диске и веса не изменяются, отдельного скачивания не нужно. Все остальные
правила, включая decoding, masking, scope и метрики, общие. Detection/masking
run.json сохраняют имя `adapter`, чтобы варианты различались в результатах.
131072 — лимит конфигурации, не гарантия вместимости GPU. Для коротких
контекстов предпочтителен нативный вариант; YaRN может изменить качество.

В `cmd` job YAML или detection CLI можно указать:

```text
--retrieval-metrics needle_token_multiset_v1,needle_attention_mass_v1,legacy
--attention-scope all_decode_tokens
```

Все метрики считаются на одной генерации и одних attention-событиях.
Никакой основной метрики при нескольких метриках нет. `--retrieval-metric NAME`
задаёт одиночный режим и не используется вместе с `--retrieval-metrics`.
Без нового флага прежнее поведение сохранено:
одна `needle_token_multiset_v1`, либо выбранная через `--retrieval-metric`.
Файл для masking выбирается независимо через `--head-scores PATH`.

Сбор и агрегация теперь разделены. Detection не создаёт общий рейтинг,
а сохраняет per-case оценки всех завершённых генераций независимо от ROUGE.
После скачивания результатов запустите на CPU:

```bash
python aggregate_head_scores.py --run datasphere-results/qwen3-8b-survey
```

По умолчанию выбираются случаи с ROUGE строго >50. Для всех случаев:

```bash
python aggregate_head_scores.py --run datasphere-results/qwen3-8b-survey \
  --all-cases --output-dir datasphere-results/qwen3-8b-survey/detection/aggregation-all
```

Доступны `--success-threshold`, `--case-ids detect-1,detect-2`, `--lengths`,
`--depths`. `--allow-incomplete` разрешает промежуточную агрегацию, manifest
при этом остаётся complete=false. Пустой отбор вызывает ошибку, не пишет рейтинг.
Именованные head_scores JSON создаются только агрегатором, по умолчанию в
`detection/aggregation/`. Его run.json содержит выбранные случаи/хэши и фильтры;
исходный detection/run.json не изменяется. Новый detection run_id защищает от
подмешивания оставшихся результатов прошлых запусков в той же папке.
В full job агрегатор вызывается автоматически между detection и masking;
smoke и отдельный detection только собирают данные. В mask ranking задаётся
готовым `--head-scores`. Графики автоматически используют default aggregation.

Файлы detection и агрегации:

- `aggregation/head_scores_<имя_метрики>.json` — история выбранных агрегатором случаев;
- detection `run.json`: список `retrieval_metrics`, одиночная `retrieval_metric` (null в multi),
  фактический capture и признак `needle_mass_capture`; mapping `head_score_files`
  теперь находится в `aggregation/run.json`;
- каждый result JSON: `experiment.retrieval_scores` со значениями всех метрик,
  в том числе для случаев, не прошедших ROUGE-порог. Для новой метрики также
  `retrieval_qualifying_steps` — число подходящих шагов.

Второй скрипт читает явно выбранный рейтинг:

```text
--head-scores datasphere-results/qwen3-full/detection/aggregation/head_scores_needle_attention_mass_v1.json
```

Этот флаг доступен и напрямую в `needle_in_haystack_with_mask.py`, и в
`datasphere/job.py`, который передаст файл всем masking-условиям.
В multi-metric full он обязателен: первая метрика не выбирается автоматически.
В `qwen3-full.yaml` путь уже явно задан для multiset; смените суффикс файла,
чтобы использовать другую рассчитанную метрику. Если нужны несколько метрик,
замените одиночный `--retrieval-metric` на `--retrieval-metrics` со списком.
При использовании готового локального файла в Jobs дополнительно объявите его
в YAML `inputs`: код не загружает произвольный путь автоматически.
Masking run.json сохраняет `head_scores_file` для воспроизводимости.
Графики multi-metric прогона также требуют явного `--metric NAME`.

Для второго этапа без повторного detection используйте `job.py --profile mask`.
Для удаления голов с наименьшим средним score используйте во втором скрипте
`--head-selection bottom --mask-topk 4`. Рейтинг читается целиком, без обрезки
до100 голов; выбирается его хвост. Условия сохраняются отдельно в `bottom4`,
с выбранными головами в manifest. При равных score выбор детерминированный
по порядку рейтинга, а не дополнительный случайный повтор. Низкий retrieval
score не доказывает низкую причинную важность; именно это проверяет masking.
В job.py full/mask доступны `--mask-selections top,bottom,random` либо поднабор
(например, `top,bottom`). Default `top,random` оставлен, baseline всегда один
на контекст/сетку; `--random-repeats` применяется только к random.
Графики masking выделяют bottom отдельной серией.
Обязательны `--mask-data` и `--head-scores` (уже существующий JSON); `--detection-data`
не нужен. Режим выполняет тот же цикл, что и второй этап `full`: baseline,
top-head masks и random-head masks по всем `--topks`, `--context-count`
контекстам и `--random-repeats` случайным маскам. Сетки `--lengths`/`--depths`,
seed, модель и параметры генерации передаются как в `full`.
`--retrieval-metric(s)` в режиме mask не применяются и отклоняются:
метрика уже определяется выбранным JSON рейтинга.
Для DataSphere YAML задайте `--profile mask`, уберите параметры detection,
объявите готовый JSON в `inputs` (например, `путь/к/head_scores_needle_token_multiset_v1.json: HEAD_SCORES`)
и передайте `--head-scores ${HEAD_SCORES}`. Укажите отдельный `--output-root`
для новых evaluation-результатов. Существующие full/smoke YAML не изменены.

Общие истории включают случаи, выбранные отдельным агрегатором; default ROUGE>50,
а `--all-cases` включает и неудачные ответы.
Параметр `--attention-scope` определяет, учитывать ли thinking; ROUGE не меняется.
Графики выбранной метрики:

```bash
MPLCONFIGDIR=/tmp/retrieval-heads-mpl .venv-datasphere/bin/python plot_results.py \
  --run datasphere-results/ИМЯ-ПРОГОНА --metric needle_attention_mass_v1
```

Графики попадут в `plots/needle_attention_mass_v1/`. Старые результаты без
attention-вероятностей нельзя пересчитать в эту метрику.

#### Включить thinking в анализ retrieval heads

В `cmd` выбранного YAML или в detection CLI явно добавьте:

```text
--attention-scope all_decode_tokens
```

Готовый парный повтор проверки24k: `datasphere/qwen3-thinking-smoke.yaml`.
Он повторяет параметры `qwen3-memory-smoke.yaml`, меняя только область анализа;
результаты сохраняются отдельно в `datasphere-results/qwen3-thinking-smoke`.
Запуск из `updated` после настройки авторизации:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 -c datasphere/qwen3-thinking-smoke.yaml
```

Тогда retrieval-score и сохраняемые attention traces учитывают весь decode:
thinking, его маркеры, финальный ответ и EOS. Это общая оценка, не две отдельные
метрики thinking/answer. Prefill по-прежнему не анализируется.
`--attention-scope answer_only` возвращает анализ после `</think>`; это текущий
default Qwen3. Без флага другие адаптеры сохраняют свою исходную политику.
Опция передаётся через `datasphere/job.py`, значение записывается в detection
run.json/results как `attention_scope`. Модель и prompt от неё не меняются.

ROUGE остаётся по финальному ответу, raw текст сохраняется отдельно. Только
случаи с финальным ROUGE > success-threshold включаются в head_scores.json,
даже если thinking анализируется. Незавершённое thinking теперь может давать
observer-события, но не финальный ответ и не успешный случай в рейтинге.
Multiset-квоты общие для thinking+answer и не сбрасываются на `</think>`:
score остаётся 0..1. С `--retrieval-metric legacy` повторения не ограничены.
Masking по-прежнему применяется ко всему decode независимо от области анализа.
Для сравнения режимов используйте одинаковые параметры и разные output-root/outputs.
Старые рейтинги нельзя пересчитать по одному raw_model_response: нужны исходные
attention-события, которые при top1 режиме не сохранялись для thinking.

#### Переключение prefill attention

В `cmd` любого YAML можно явно выбрать `--prefill-attention`:

- `sdpa_memory_efficient` — для T4: временно повторяет KV до числа query-голов,
  использует только PyTorch EFFICIENT_ATTENTION, без GQA-флага и math-fallback.
  Если ядро недоступно, завершается ошибкой вместо огромного выделения памяти.
- `sdpa` — стандартный выбор ядра Transformers/PyTorch, в том числе GQA;
  на текущем T4/PyTorch 2.6 может откатиться на math и вызвать OOM на длинном prefill.
- `flash_attention_2` — backend Transformers для установленного совместимого
  пакета flash-attn и поддерживаемой GPU. Сейчас пакет не установлен и режим
  в облаке не проверен. Один переключатель не устанавливает библиотеку.

Во всех режимах есть тот же компактный KV-cache. Это переключение вычисления
prefill, а не включение/выключение кеша. Decode, attention capture и masking
используют прежний наблюдаемый eager backend, независимо от prefill.
Разные ядра могут давать небольшие численные отличия, влияющие на greedy-ответы.
Логи `[memory]` показывают пик allocated GPU memory после prefill и генерации;
это не вся память процесса и не включает свободный reserved cache.
Новые survey/full выбирают `sdpa_memory_efficient`; прежний короткий smoke не изменён.
Отдельная проверка 24k: `datasphere/qwen3-memory-smoke.yaml`, три исходных вопроса,
depth 45, 2048 новых токенов; результаты `datasphere-results/qwen3-memory-smoke`.
Проверена 2026-10-08: job `bt15hbvov0v5744p9s4b` SUCCESS на T4,
3/3 EOS, ROUGE recall100, peak allocated12.536GiB; токены825/750/637.
Общая занятая память по GPU-мониторингу около94%, запас для24k небольшой.

Расширенная проверка после smoke задаётся готовыми Qwen3-конфигурациями ниже.
`datasphere/qwen3-survey.yaml` — 36 detection-генераций: длины 4k/8k/16k/24k,
глубины 15/45/75%, три исходных корпуса со своими needle-вопросами;
`datasphere/qwen3-full.yaml` — 324 генерации с парным baseline/top/random masking.
Полный survey с новым backend и full пока не проверены облачным запуском;
прошла отдельная memory-smoke проверка24k. Сначала запускайте survey.

Модель должна лежать на подключаемом диске проекта в
`c0/models/Qwen/Qwen3-4B-Thinking-2507/`, включая tokenizer и все части весов.
Из локальной папки `updated`:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 -c datasphere/qwen3-smoke.yaml
```

Конфиг явно задаёт `--adapter qwen3` и
`--model ${DS_PROJECT_HOME}/c0/models/Qwen/Qwen3-4B-Thinking-2507`.
Скачивание весов отключено через offline-переменные. На T4 используется float16,
3 detection-кейса длиной 1000 на depth 50 и `--max-new-tokens 2048`.
В конфиге явно выбрана `--retrieval-metric needle_token_multiset_v1`.
Лимит включает рассуждение и ответ; для Thinking-2507 нельзя останавливаться
на первом переносе строки. Если лимита не хватает, в результате будет
`finish_reason=max_new_tokens`; незавершённое рассуждение не считается ответом.
Можно увеличить лимит явно в этом YAML.

Результаты: `datasphere-results/qwen3-smoke/`. Проверяйте `source=local`,
`parameter devices=['cuda:0']`, `detection/run.json` с `complete=true` и 3 случаями.
ROUGE считается по финальному `model_response` после `</think>`; исходный вывод
сохраняется как `raw_model_response`, токены — `generated_token_ids`.
Retrieval-score и traces анализируют только финальный ответ после `</think>`;
thinking и закрывающий тег не передаются наблюдателям. Если thinking не завершён,
attention trace пуст. Индексы шагов соответствуют исходной полной генерации.
Новая метрика `needle_token_multiset_v1` ограничивает зачёты каждого token ID
его числом в needle-span отдельно для каждой головы. Score — число зачтённых
токенов / длина span, максимум 1. Название метрики и область анализа записываются
в manifests/results. Старые рейтинги не пересчитываются автоматически.
Для старой формулы добавьте `--retrieval-metric legacy` в `cmd` выбранного YAML
или передайте её напрямую detection CLI. Новая формула выбирается через
`--retrieval-metric needle_token_multiset_v1` (это значение по умолчанию).
Legacy засчитывает повторные совпадения без ограничения и может быть больше 1.
Оба режима для Qwen3 анализируют только ответ после thinking. Job передаёт выбор
в detection; masking использует полученный рейтинг и сам retrieval-score не считает.
Greedy-выбор токена сохранён. Профиль smoke не запускает masking-условия.

Особенности архитектуры и thinking-режима описаны в
[карточке модели Qwen](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).

Для другого расположения добавьте в `cmd` YAML:

```yaml
  --model Qwen/Qwen3.5-0.8B
  --model-search-dir ${DS_PROJECT_HOME}/my-models
  --model-search-dir ${DS_PROJECT_HOME}/backup-models
```

Проверяются `my-models/Qwen/Qwen3.5-0.8B`, затем
`backup-models/Qwen/Qwen3.5-0.8B`, затем Hugging Face. Замените существующий
аргумент поиска в YAML этими аргументами. Чтобы указать саму папку checkpoint, используйте
`--model ${DS_PROJECT_HOME}/models/Qwen/Qwen3.5-0.8B`;
отсутствующий прямой путь вызывает ошибку. Найденная неполная папка (нет
конфига или весов/частей весов) также вызывает ошибку вместо скрытого скачивания.
Локальные веса и tokenizer загружаются с `local_files_only=True`.
Логи `[model] source=local ...` или `[model] source=HuggingFace ...` показывают выбор.
При поиске по ID в результатах сохраняется исходный ID модели.

Те же `--model`, `--adapter`, `--model-search-dir` доступны в обоих
экспериментальных CLI. Папки поиска всегда задаются явно;
вне DataSphere можно передать `--model-search-dir /path/to/models`.

Документация:
[диск проекта в Jobs](https://yandex.cloud/ru/docs/datasphere/concepts/jobs/#config),
[скачивание в локальную папку](https://huggingface.co/docs/huggingface_hub/en/guides/download).

GPU jobs используют закреплённую CUDA-сборку из официального индекса PyTorch.
В проверенной T4 VM драйвер сообщает поддержку CUDA 12.2; установка последнего
PyTorch из PyPI дала CUDA 13.0 и `cuda_available=false`. Поэтому `full` и GPU
smoke используют `torch==2.6.0+cu118` и явное устройство `cuda:0`.

## 1. Проверить доступ

Используется DataSphere project:

```text
Project ID:   bt1u5v72b71eesdhp9k5
Community ID: bt1dv4jmd0u81i806t74
```

У аккаунта должна быть роль `Developer` проекта, а у community должен быть
подключён платёжный аккаунт. Для `full.yaml` в community должна быть разрешена
хотя бы одна из конфигураций `g1.1`, `gt4i.1` или `gt4.1`.

## 2. Один раз установить локальные CLI

Сначала установите `yc` по официальной инструкции:

<https://yandex.cloud/ru/docs/cli/operations/install-cli>

Затем авторизуйте его. Это интерактивный шаг с браузером, его должен выполнить
владелец аккаунта:

```bash
yc init
```

Для `datasphere` нужен Python не новее 3.12. На этой машине системный Python
3.14, поэтому используем уже установленный `uv`:

```bash
cd /home/controldev/projects/ai360/ai360-heads-in-the-haystack/updated
uv python install 3.12
uv venv --python 3.12 .venv-datasphere
uv pip install --python .venv-datasphere/bin/python datasphere
```

Windows PowerShell:

```powershell
cd C:\path\to\ai360-heads-in-the-haystack\updated
uv python install 3.12
uv venv --python 3.12 .venv-datasphere
uv pip install --python .venv-datasphere\Scripts\python.exe datasphere
```

## 3. Авторизовать текущий терминал

Bash или WSL:

```bash
export YC_IAM_TOKEN="$(yc iam create-token)"
```

PowerShell:

```powershell
$env:YC_IAM_TOKEN = (yc iam create-token)
```

Токен нельзя отправлять в чат, записывать в Git или вставлять в YAML. Он живёт
ограниченное время; если CLI позже вернёт `401`, выполните команду ещё раз.

Проверка доступа из Bash/WSL:

```bash
.venv-datasphere/bin/python datasphere/cli.py project get --id bt1u5v72b71eesdhp9k5
```

Из PowerShell:

```powershell
.venv-datasphere\Scripts\python.exe datasphere/cli.py project get --id bt1u5v72b71eesdhp9k5
```

## 4. Smoke: сначала обязательно выполнить его

Smoke использует `c1.4`, грузит модель в `float32` и выполняет три коротких
детекционных случая: длина 1000, depth 50. Он проверяет загрузку модели,
подготовку данных, генерацию и перехват attention. Маскирование в smoke не
запускается.

Bash/WSL:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 \
  -c datasphere/smoke.yaml
```

PowerShell:

```powershell
.venv-datasphere\Scripts\python.exe datasphere/cli.py project job execute `
  -p bt1u5v72b71eesdhp9k5 `
  -c datasphere/smoke.yaml
```

После успеха должны появиться:

```text
datasphere-results/smoke/detection/run.json
datasphere-results/smoke/detection/head_scores.json
datasphere-results/smoke/detection/results/*.json
```

В `run.json` поле `complete` должно быть `true`.

## 5. Полный GPU-прогон

Перед полным прогоном можно проверить именно GPU, CUDA и скорость на 6k:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 \
  -c datasphere/gpu-smoke.yaml
```

GPU smoke выполняет три detection-кейса на depth 50 и сохраняет данные в
`datasphere-results/gpu-smoke/`. В `runtime.json` сохраняются версия PyTorch,
CUDA и обнаруженная GPU. При недоступной CUDA запуск останавливается сразу.

`full.yaml` запускает один и тот же набор пар context/depth для всех этапов:

1. поиск retrieval heads;
2. baseline без маски;
3. для каждого k из `4,8,16`: маскирование top-k и три random-k повтора;
4. сохранение отдельных результатов каждого прогона без усреднения.

Используется длина `10000`, depths `25,50,75`, три контекста с seeds `42,43,44`,
`float16`, SDPA для prefill и исходное `legacy_uniform`-маскирование.

`--seed 42` задаёт случайный порядок текстовых файлов и выбор случайных голов.
Для текстов используется отдельный генератор случайности: все условия получают
одинаковые prompt-токены для каждой пары длина/depth. SHA-256 prompt сохраняется
для последующей проверки отдельным скриптом анализа. Чтобы изменить только тексты, добавьте
`--context-seed 123`; чтобы сменить оба вида случайности, измените `--seed`.
При неизменных исходных файлах seed воспроизводит контексты. Если в корпусе
один файл, перестановка файлов не меняет текст. Длины используют префиксы одного
перемешанного корпуса; depth задаёт положение needle в этом тексте.

Detection выполняется один раз (9 генераций). Для каждого из трёх контекстов
baseline и каждый top-k выполняются один раз на каждой глубине. Random-k
повторяется с seeds `42,43,44`; контекст при этом фиксирован отдельно.
Итого 126 генераций. Job сохраняет исходные ответы и scores каждого случая,
номера маскированных голов, seeds в `run.json` и хеш входных токенов в result
JSON. Усреднение и построение графиков выполняются отдельно после загрузки.

После каждой завершённой генерации в логах появляется общий прогресс, например
`[progress] 63/126 generations (50.0%)`. Это процент выполненных тестов,
а не затраченного времени; установка зависимостей и скачивание файлов в него
не входят. Счётчик учитывает detection и все masking-условия, включая smoke.

Bash/WSL:

```bash
.venv-datasphere/bin/python datasphere/cli.py project job execute \
  -p bt1u5v72b71eesdhp9k5 \
  -c datasphere/full.yaml
```

PowerShell:

```powershell
.venv-datasphere\Scripts\python.exe datasphere/cli.py project job execute `
  -p bt1u5v72b71eesdhp9k5 `
  -c datasphere/full.yaml
```

Результаты окажутся в:

```text
datasphere-results/full/detection/
datasphere-results/full/evaluation/context-42/evaluation/baseline/
datasphere-results/full/evaluation/context-42/evaluation/top4/  # также top8, top16
datasphere-results/full/evaluation/context-42/random/repeat-0/evaluation/random4/
# Аналогично repeat-1/2, random8/16 и context-43/44.
```

Скачанные JSON содержат данные каждого случая. Усреднение и визуализация —
отдельная задача; локальные графики строит `plot_results.py`.

## 6. Изменить сетку эксперимента

Профиль `full` задаётся в `datasphere/job.py`. Без правки Python можно
добавить аргументы в `cmd` файла `full.yaml`, например:

```yaml
  --lengths 10000
  --depths 25,50,75
  --topks 4,8,16
  --context-count 3
  --random-repeats 3
  --seed 42
```

Все masking-условия автоматически получат одну и ту же сетку и контексты.
Для одного k вместо `--topks` можно указать `--topk 8`.

## 7. Если job упала

- `401` или `Unauthenticated`: обновить `YC_IAM_TOKEN` из шага 3.
- `Permission denied`: выдать аккаунту роль Developer именно в указанном
  DataSphere project.
- `No available cloud instance`: разрешить одну из GPU-конфигураций в community
  или изменить список `cloud-instance-types` в `full.yaml`.
- Ошибка при скачивании Hugging Face: проверить исходящий интернет у job. В этом
  случае checkpoint надо заранее скачать и передать как DataSphere input; не
  добавляйте веса в Git.
- `Only N eligible heads are ranked`: detection не нашла восемь голов на
  успешных ответах. Сначала открыть `detection/head_scores.json`; не подменять
  результат случайными головами. Увеличить detection-сетку или временно
  уменьшить `--topk` одинаково для job и сравнения.

## Что необходимо от пользователя, если job запускает агент

Пользователь один раз выполняет `yc init`, проверяет доступ к project и явно
разрешает платный запуск GPU-job. IAM-токен агенту присылать не нужно. После
этого агент может установить `datasphere`, отправить smoke, проверить логи и
запустить полный эксперимент.
