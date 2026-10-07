# Запуск эксперимента в Yandex DataSphere

Эта инструкция относится к коду в `updated/`. Корневой `requirements.txt`
относится к старой реализации и для этого запуска не используется.

В репозитории уже есть:

- `updated/datasphere/job.py` — запускает этапы эксперимента по очереди;
- `updated/datasphere/smoke.yaml` — короткая проверка на CPU;
- `updated/datasphere/full.yaml` — полный прогон на одной GPU;
- `updated/requirements.txt` — зависимости удалённого Python-окружения.

Все команды ниже выполняются **из корня репозитория**, а не из `updated/`.

Корпуса находятся в `updated/data/haystack_for_detect/` и
`updated/data/PaulGrahamEssays/`; YAML передаёт их в job через `inputs`.

Драйвер компилируется, оба YAML разбираются локально без ошибок. Удалённый
smoke также проверен в указанном проекте: job `bt1mpuoprch2p2dmfsoh` успешно
загрузила Qwen, выполнила все три случая и вернула scores 100, 100 и 94.44.
Несмотря на это, после изменения зависимостей сначала повторяйте smoke, а не
запускайте платный полный прогон.

## Что произойдёт

Локальный `datasphere` CLI загрузит в job код и тексты. На отдельной VM
DataSphere создаст Python 3.12 environment, установит зависимости, а
Transformers скачает `Qwen/Qwen3.5-0.8B` с Hugging Face. Результаты job будут
скачаны обратно в `updated/datasphere-results/`.

Модель не надо заранее класть в репозиторий. В полном эксперименте все этапы
идут в одной job, поэтому checkpoint скачивается один раз и затем используется
четырьмя дочерними процессами из одного Hugging Face cache этой VM.

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
cd /home/controldev/projects/ai360/ai360-heads-in-the-haystack
uv python install 3.12
uv venv --python 3.12 .venv-datasphere
uv pip install --python .venv-datasphere/bin/python datasphere
```

Windows PowerShell:

```powershell
cd C:\path\to\ai360-heads-in-the-haystack
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
.venv-datasphere/bin/datasphere project get --id bt1u5v72b71eesdhp9k5
```

Из PowerShell:

```powershell
.venv-datasphere\Scripts\datasphere.exe project get --id bt1u5v72b71eesdhp9k5
```

## 4. Smoke: сначала обязательно выполнить его

Smoke использует `c1.4`, грузит модель в `float32` и выполняет три коротких
детекционных случая: длина 1000, depth 50. Он проверяет загрузку модели,
подготовку данных, генерацию и перехват attention. Маскирование в smoke не
запускается.

Bash/WSL:

```bash
.venv-datasphere/bin/datasphere project job execute \
  -p bt1u5v72b71eesdhp9k5 \
  -c updated/datasphere/smoke.yaml
```

PowerShell:

```powershell
.venv-datasphere\Scripts\datasphere.exe project job execute `
  -p bt1u5v72b71eesdhp9k5 `
  -c updated/datasphere/smoke.yaml
```

После успеха должны появиться:

```text
updated/datasphere-results/smoke/detection/run.json
updated/datasphere-results/smoke/detection/head_scores.json
updated/datasphere-results/smoke/detection/results/*.json
```

В `run.json` поле `complete` должно быть `true`.

## 5. Полный GPU-прогон

`full.yaml` запускает один и тот же набор пар context/depth для всех этапов:

1. поиск retrieval heads;
2. baseline без маски;
3. для каждого k из `4,8,16`: маскирование top-k и три random-k повтора;
4. сохранение отдельных результатов каждого прогона без усреднения.

Используется длина `6000`, depths `25,50,75`, три контекста с seeds `42,43,44`,
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

Bash/WSL:

```bash
.venv-datasphere/bin/datasphere project job execute \
  -p bt1u5v72b71eesdhp9k5 \
  -c updated/datasphere/full.yaml
```

PowerShell:

```powershell
.venv-datasphere\Scripts\datasphere.exe project job execute `
  -p bt1u5v72b71eesdhp9k5 `
  -c updated/datasphere/full.yaml
```

Результаты окажутся в:

```text
updated/datasphere-results/full/detection/
updated/datasphere-results/full/evaluation/context-42/evaluation/baseline/
updated/datasphere-results/full/evaluation/context-42/evaluation/top4/  # также top8, top16
updated/datasphere-results/full/evaluation/context-42/random/repeat-0/evaluation/random4/
# Аналогично repeat-1/2, random8/16 и context-43/44.
```

Скачанные JSON содержат данные каждого случая. Усреднение и визуализация —
отдельная задача; скриптов анализа в текущей версии нет.

## 6. Изменить сетку эксперимента

Профиль `full` задаётся в `updated/datasphere/job.py`. Без правки Python можно
добавить аргументы в `cmd` файла `full.yaml`, например:

```yaml
  --lengths 6000
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
