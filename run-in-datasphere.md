# Запуск `retrieval-heads` в Yandex DataSphere Jobs

Инструкция для агента: как прогнать пайплайн этого репозитория не на ноутбуке, а
заданием (job) в DataSphere.

| | |
|---|---|
| Проект | **Сергей Андреевич Копылов** |
| Project ID | `bt1u5v72b71eesdhp9k5` |
| Community ID | `bt1dv4jmd0u81i806t74` |
| Страница проекта | <https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5> |
| CLI | `datasphere` 0.10.0 (PyPI), окружение `.venv-datasphere/` |
| Точка входа в job | `scripts/datasphere_job.py` (в этом репозитории) |
| Окружение job | `scripts/requirements-datasphere.txt` |

Все команды выполняются **из корня репозитория** `~/projects/ai360-heads-in-the-haystack`,
потому что относительные пути в job-конфиге (входные файлы, `requirements-file`,
`local-paths`) разрешаются относительно текущего каталога CLI, а не относительно
самого конфига.

---

## 0. Кратко: путь целиком

```bash
cd ~/projects/ai360-heads-in-the-haystack
export PATH="$HOME/yandex-cloud/bin:$PATH"
source scripts/datasphere_auth.sh                    # -> export YC_IAM_TOKEN (без лишних вкладок, см. §3.1)
CLI=.venv-datasphere/bin/datasphere                 # CLI уже установлен в этот venv
PROJECT=bt1u5v72b71eesdhp9k5

$CLI project get --id "$PROJECT"                    # 1. доступ есть?
mkdir -p .cache/datasphere            # IAM-token cache only; configs live in configs/datasphere/

# 2. smoke: описать + прогнать детекцию на маленькой модели (см. §3)
$CLI project job execute -p "$PROJECT" -c configs/datasphere/smoke.yaml

# 3. рабочий прогон: CPU-laptop (§4) или GPU-paper (§4, §6)
$CLI project job execute -p "$PROJECT" -c configs/datasphere/laptop.yaml

# 4. результаты окажутся в ./ds-results/ (см. §5)
```

Конфиги `smoke.yaml` / `laptop.yaml` / `paper.yaml` уже разложены в
`configs/datasphere/` (в git, не в `.cache/`) и проверены парсером DataSphere CLI;
их содержимое — в §3–§4, воспроизведи заново, если каталог пропал.

---

## 1. Что уже лежит в репозитории и зачем

* **`scripts/datasphere_job.py`** — единственная точка входа задания. Он:
  выставляет веса как `./models` (симлинк на входной каталог `${WEIGHTS}` либо
  скачивание с HuggingFace), при `--dtype` пишет копию реестра
  `configs/models.runtime.json` и выставляет `RETRIEVAL_HEADS_MODELS_JSON`
  (трекнутый `configs/models.json` не меняется), а затем последовательно
  запускает стадии
  `describe / detect / mask / qa / cot / compare / figures` через
  `python -m retrieval_heads.cli`. Масштаб стадий берётся из таблицы `SCALES`,
  которая повторяет `scripts/reproduce_laptop.sh` и `scripts/reproduce_gpu.sh`
  (`--profile smoke|laptop|paper`), чтобы job-прогон был сравним с локальным.

  ```bash
  python3 scripts/datasphere_job.py --help
  ```

* **`scripts/datasphere_auth.sh`** — выдаёт проверенный `YC_IAM_TOKEN` и кэширует
  его на ~11 часов, чтобы не запускать `yc` перед каждым заданием (при истёкшей
  федеративной сессии `yc` открывает вкладку консоли, см. §3.1).

  ```bash
  source scripts/datasphere_auth.sh
  ```

* **`scripts/requirements-datasphere.txt`** — окружение job (torch, transformers,
  numpy, matplotlib, tqdm, pytest). Важно: DataSphere CLI **не принимает** в
  requirements-файле маркеры (`; python_version < ...`) и прямые URL — только
  обычные спецификаторы.

Дополнительно ничего создавать не нужно: `retrieval_heads/`, `configs/` и
`scripts/` уезжают в job через `local-paths` (см. §2).

---

## 2. Как устроено выполнение: проверенные факты

Всё ниже проверено реальными job'ами в этом проекте (история запусков, id — в
круглых скобках). Это важно, потому что часть поведения DataSphere не описана в
документации:

| Факт | Что именно |
|---|---|
| Рабочий каталог задания | `/job`; `local-paths` распаковываются **прямо в него** с сохранением имени (`/job/retrieval_heads`, `/job/configs`, `/job/scripts`) — поэтому `REPO_ROOT` внутри `retrieval_heads/cli.py` указывает на `/job`, и реестр `configs/models.json` находится (`bt1dv1trd45adas0jl2r`) |
| Python в job | при `version: "3.10"` — CPython 3.10.12, виртуальное окружение `/job/.job_python_venv_*/bin/python3`; `sys.executable` в драйвере уже указывает на него, поэтому `subprocess` запускает стадии в правильном окружении (`bt1dv1trd45adas0jl2r`) |
| Входные файлы (`inputs`) | файл → `/job/onefile_<rand>`, каталог → `/job/src_<rand>` (имя **выводится из `var`**, содержимое сохраняется). Обращаться к ним можно **только** через `${VAR}` — предсказуемого пути нет (`bt1camkdnampgt108k98`) |
| `local-paths` | и каталоги, и отдельные файлы; имя сохраняется, файл кладётся в `/job`. Это же переводит окружение в режим `fully manual` (см. §8, п. 2) |
| `outputs` | относительные пути читаются с job-VM **и скачиваются локально по тому же относительному пути**. Каталоги скачиваются как zip и распаковываются (`bt1i9v1skv31dn3o6s0o`, `bt1e21ljrc4kl7eo7b36`) |
| `args` | **не** подставляются в `cmd`: ни `{{FOO}}`, ни `$FOO` не раскрылись — это просто дополнительные аргументы командной строки. Плейсхолдеры — только `${VAR}` для объявленных `inputs`/`outputs`/`s3-mounts`/`datasets` (`bt1camkdnampgt108k98`) |
| `cloud-instance-types` | список приоритетов: первая доступная конфигурация. `c1.4` в сообществе разрешена, сборка окружения заняла ~50 с (`bt1rdo5vtiurnqi0f4r1`) |
| Драйвер + веса + каталоги-выходы | проверено двумя мок-заданиями: smoke-ветка — `bt1rdo5vtiurnqi0f4r1`, paper-ветка (9 длин в `detect` и `--dtype bfloat16`) — `bt1bltpnglcl57orolg9`: все стадии, симлинк `/job/models` → `/job/weights_*`, выходной каталог скачался |

Ссылка на любое задание:
`https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5/job/<job_id>`

**Не проверено** (проверяй smoke-заданием, прежде чем строить на этом план):
GPU-конфигурации и квоты на них (§6), скачивание весов из job (`--download-weights`,
§7), `flags: [attach-project-disk]` и `${DS_PROJECT_HOME}` (§7), сборка окружения с
torch/transformers (в проверках ставился только маленький requirements).

---

## 3. Подготовка и smoke-прогон

### 3.1 Доступ и токен

```bash
cd ~/projects/ai360-heads-in-the-haystack
source scripts/datasphere_auth.sh       # -> export YC_IAM_TOKEN (см. ниже про вкладки)
CLI=.venv-datasphere/bin/datasphere
PROJECT=bt1u5v72b71eesdhp9k5

$CLI project get --id "$PROJECT"        # ожидаем таблицу с именем проекта
```

Если `.venv-datasphere/` ещё нет (новый клон репозитория):

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv .venv-datasphere
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-datasphere/bin/python datasphere
```

**Почему `yc iam create-token` открывает вкладки браузера.** Аккаунт в этом облаке
**федеративный**: в `~/.config/yandex-cloud/credentials/default` лежит сессионный
токен федерации (`yctk…`) со сроком жизни ~12 часов. Пока сессия свежая, `yc`
работает молча и вкладок не открывает (проверено: 0 обращений к браузеру). Когда
сессия истекла, `yc` обязан пройти веб-аутентификацию: печатает
`Authentication web site will be opened`, в интерактивном терминале ждёт
`Press 'enter' to continue...`, а затем открывает `https://console.yandex.cloud/`
через `cmd.exe` / `powershell.exe` (WSL). Песочница агента вдобавок запрещает
запись в `~/.config/yandex-cloud`, поэтому `yc` не может сохранить обновлённую
сессию и переаутентифицируется **при каждом вызове** — по вкладке на задание.
`YC_NO_BROWSER=1` в CLI 1.40.0 этого **не** подавляет (проверено: вкладка всё
равно открылась).

Важно: токен из хранилища (`yctk…`) **нельзя** подставлять в `YC_IAM_TOKEN` —
DataSphere API отвечает на него `401`. Нужен именно IAM-токен `t1.…`, который
выдаёт `yc iam create-token`.

`scripts/datasphere_auth.sh` делает две попытки по порядку:

1. кэш агента `.cache/datasphere/iam-token`, если файл свежее ~11 часов — токен
   дополнительно проверяется запросом к API, и `yc` не вызывается вообще;
2. иначе — один вызов `yc`; результат проверяется и кладётся в кэш, поэтому
   следующее обновление будет не раньше чем через ~11 часов. Вкладка при этом
   появится только если истекла федеративная сессия.

Совет: раз в ~12 часов обновляй сессию сам — `yc iam create-token` в своём
терминале (одна вкладка, по Enter на приглашении). После этого агент работает
молча, потому что `yc` уже не уходит на веб-аутентификацию.

Пути переопределяются переменными `DATASPHERE_TOKEN_CACHE`,
`DATASPHERE_PROJECT_ID` (пусто — не проверять токен), `YC_BIN`. Кэш — это секрет:
он лежит в `.cache/` (в `.gitignore`), с правами `600`, и его стоит удалять после
работы.

Про авторизацию — два подводных камня:

* Флаг `-t` у `datasphere` принимает **OAuth-токен** (CLI сам обменивает его на
  IAM) и отвечает `OAuth token is invalid or expired`, если передать IAM-токен.
  Учти, что с 1 июня 2026 года Яндекс ID больше не выдаёт **новые** OAuth-токены
  ([документация](https://yandex.cloud/ru/docs/iam/concepts/authorization/oauth-token)),
  так что этот путь годится только для токена, выданного раньше.
* IAM-токен передаётся только переменной окружения **`YC_IAM_TOKEN`** (плюс
  поддерживаются `YC_TOKEN` / `YC_OAUTH_TOKEN` для OAuth). Он не обновляется сам и
  живёт ~12 ч; для долгих прогонов используй `--async` — токен нужен лишь чтобы
  создать задание и опросить его статус, сам job от истечения токена не страдает.

Никогда не записывай токен в конфиг, в репозиторий или в логи.

### 3.2 smoke.yaml

```yaml
# configs/datasphere/smoke.yaml
name: rh-smoke
desc: retrieval-heads smoke run (describe + detect, qwen3-0.6b)

cmd: python3 scripts/datasphere_job.py --weights ${WEIGHTS} --models qwen3-0.6b --profile smoke --stages describe,detect --out-prefix ds-results

inputs:
  - models:            # 3.2 ГБ; адрес каталога отдаётся переменной WEIGHTS
      var: WEIGHTS

outputs:
  - ds-results         # всё, что драйвер напишет под этим префиксом

env:
  python:
    type: manual
    version: "3.10"
    requirements-file: scripts/requirements-datasphere.txt
    local-paths:       # уезжает в /job как есть
      - retrieval_heads
      - configs
      - scripts

cloud-instance-types:
  - c1.4               # 4 vCPU / 32 ГБ — минимум, разрешён в сообществе
```

Запуск:

```bash
$CLI project job execute -p "$PROJECT" -c configs/datasphere/smoke.yaml
```

CLI стримит логи до конца задания. Признак успеха: в логах `job completed
successfully`, локально появился `ds-results/qwen3-0.6b/scores_next_step.json`,
а `job get` показывает `SUCCESS`:

```bash
$CLI project job get --id <job_id> --format json
```

Первый запуск грузит веса в проект (несколько минут в зависимости от канала);
DataSphere кеширует входные данные в проекте, поэтому последующие запуски с теми
же весами быстрее.

---

## 4. Рабочие прогоны

### 4.1 CPU, масштаб `laptop` (аналог `reproduce_laptop.sh`)

```yaml
# configs/datasphere/laptop.yaml
name: rh-laptop
desc: full pipeline, laptop scale, CPU

cmd: python3 scripts/datasphere_job.py --weights ${WEIGHTS} --models qwen3.5-0.8b qwen3-0.6b --profile laptop --stages describe,detect,mask,qa,cot,compare,figures --out-prefix ds-results

inputs:
  - models:
      var: WEIGHTS

outputs:
  - ds-results

env:
  python:
    type: manual
    version: "3.10"
    requirements-file: scripts/requirements-datasphere.txt
    local-paths:
      - retrieval_heads
      - configs
      - scripts

cloud-instance-types:
  - c1.8               # 8 vCPU / 64 ГБ; при недоступности упадёт на c1.4
  - c1.4
```

### 4.2 GPU, масштаб `paper` (аналог `reproduce_gpu.sh`)

Отличия: конфигурация с GPU, `bfloat16` (на GPU `float32` из реестра — это
заведомо медленный вариант, см. README, раздел Limitations) и `--async`, потому
что прогон долгий:

```yaml
# configs/datasphere/paper.yaml
name: rh-paper
desc: paper-scale grid on GPU

cmd: python3 scripts/datasphere_job.py --weights ${WEIGHTS} --models qwen3.5-0.8b qwen3-0.6b --profile paper --dtype bfloat16 --stages describe,detect,mask,qa,cot,compare,figures --out-prefix ds-results

inputs:
  - models:
      var: WEIGHTS

outputs:
  - ds-results

env:
  python:
    type: manual
    version: "3.10"
    requirements-file: scripts/requirements-datasphere.txt
    local-paths:
      - retrieval_heads
      - configs
      - scripts

cloud-instance-types:
  - g2.1               # 1x A100 80 ГБ
  - g1.1               # 1x V100 32 ГБ — запасной вариант
```

```bash
$CLI project job execute -p "$PROJECT" -c configs/datasphere/paper.yaml --async
# выведет job id; дальше:
$CLI project job get    --id <job_id>            # статус
$CLI project job attach --id <job_id>            # досмотреть логи
$CLI project job cancel --id <job_id> --graceful # остановить
```

Полезные оговорки по GPU:

* `float32` для Qwen3.5-0.8B — ~3.2 ГБ весов + активации; на V100 32 ГБ помещается,
  но медленно. `--dtype bfloat16` пишет копию реестра
  `configs/models.runtime.json` внутри job и выставляет
  `RETRIEVAL_HEADS_MODELS_JSON`; трекнутый `configs/models.json` не меняется.
* Код сам выбирает устройство: `retrieval_heads/cli.py` грузит модель сразу на
  `cuda`, если `torch.cuda.is_available()`. Отдельного флага для устройства нет.
* Ускорение Qwen3.5: слои Gated DeltaNet без `flash-linear-attention` /
  `causal-conv1d` работают на чистом PyTorch. Это только скорость; добавляй эти
  пакеты в requirements осознанно (нужно совпадение с версией torch/CUDA).

---

## 5. Результаты и артефакты

Драйвер пишет всё под `--out-prefix` (`ds-results` по умолчанию), в job это
`/job/ds-results`, локально — `./ds-results`:

| Стадия | Файлы |
|---|---|
| `describe` | `ds-results/<key>/model_info.json` |
| `detect` | `scores_<pairing>.json`, `scores_<pairing>.npz`, `summary_<pairing>.json`, `instances_<pairing>.jsonl` |
| `mask` | `masking_curve.json` (+ `mixer_ablation.json` для гибридных моделей) |
| `qa` / `cot` | `task_qa.json` / `task_cot.json` |
| `compare` | `ds-results/correlation.json` (+ `overlap.json` для двух моделей) |
| `figures` | `ds-results/figures/*.pdf` (ring_graph, score_distribution, heat_map, layer_profile, corr_map, masking_heads, task_qa, task_cot, mixer_ablation) |

`<key>` — ключ реестра (`qwen3.5-0.8b`, `qwen3-0.6b`). Стадии `mask`, `qa`, `cot`
**читают** `scores_<pairing>` из того же `--out`-каталога, поэтому в одном job они
идут после `detect` (драйвер соблюдает порядок `--stages`).

Важные следствия из §2:

* Путь в `outputs` — это одновременно путь на job-VM и путь загрузки на ноутбуке.
  Префикс `ds-results` выбран специально, чтобы **не перезаписать** локальные
  результаты ноутбука в `results/`. Если нужен именно `results/`, поменяй
  `--out-prefix` и `outputs` осознанно.
* Если объявленный выходной каталог не существует после прогона, задание ругается
  на выходные файлы — объявляй только то, что стадия реально создаёт (в шаблонах
  это единый `ds-results`, который создаёт драйвер).
* Ручное скачивание (например, для `--async`-прогонов или если автоскачивание
  пропустило файлы):
  ```bash
  $CLI project job download-files --id <job_id> --output-dir ./ds-download --with-logs
  ```
  Учти лимит: суммарно скачивается до 1 ГБ, остальное — со страницы задания.
* Данные задания (логи, кеш входов, выходы) живут 14 дней. Продлить:
  ```bash
  $CLI project job set-data-ttl --id <job_id> --days 30
  ```
* Полезная команда: `$CLI project job fork --id <job_id> --cloud-instance-type g2.1 --arg NAME=VALUE`
  — перезапуск с изменениями без перезагрузки всего.

---

## 6. Конфигурации вычислительных ресурсов

Из документации DataSphere (актуальный список — <https://yandex.cloud/ru/docs/datasphere/concepts/configurations>):

| Конфигурация | vCPU | GPU | RAM, ГБ | VRAM, ГБ |
|---|---|---|---|---|
| `c1.4` (минимум) | 4 | — | 32 | — |
| `c1.8` | 8 | — | 64 | — |
| `c1.32` ¹ | 32 | — | 256 | — |
| `c1.80` ² | 80 | — | 640 | — |
| `g1.1` / `g1.2` / `g1.4` ¹ | 8/16/32 | 1/2/4 V100 | 48–384 | 32/64/128 |
| `g2.1` / `g2.2` / `g2.4` ¹ | 28/56/112 | 1/2/4 A100 | 119–476 | 80/160/320 |
| `g2.8` ² | 224 | 8 A100 | 952 | 320–640 |
| `gt4.1` ¹ | 4 | 1 T4 | 16 | 16 |
| `gt4i.1` ¹ | 8 | 1 T4i | 32 | 24 |

¹ доступна после пополнения баланса ≥ 500 ₽ или по запросу в поддержку;
² только для юрлиц, по запросу.

Перед запуском на GPU проверь: (1) конфигурация **разрешена в сообществе**
(иначе задание упадёт на создании), (2) есть квота на GPU в облаке, (3) выбранный
`g*` доступен по балансу. Дешёвая проверка — тот же конфиг с
`cloud-instance-types: [c1.4]` и `--stages describe`: он валидирует конфиг,
окружение и веса, не тратя GPU-час.

---

## 7. Как доставить веса (3.2 ГБ)

| Вариант | Как | Когда |
|---|---|---|
| **A. `inputs`** (в шаблонах) | `- models: {var: WEIGHTS}` | По умолчанию. Первый прогон — загрузка, дальше проект кеширует входные данные. Лимиты: ≤ 5 ГБ на файл, ≤ 10 ГБ суммарно, ≤ 100 записей |
| **B. Скачивание внутри job** | вместо `--weights ${WEIGHTS}` передай `--download-weights`, а вход `models` убери | Медленный канал на ноутбуке. Требует исходящий доступ в интернет с job-VM; **не проверялось** |
| **C. Диск проекта** | загрузи веса в JupyterLab проекта, в конфиг добавь `flags: [attach-project-disk]`, а в `cmd` подставь `${DS_PROJECT_HOME}/models` вместо `${WEIGHTS}` | Веса уже лежат в проекте. `${DS_PROJECT_HOME}` доступна **только** с этим флагом (CLI проверяет это при разборе конфига) |
| **D. S3-коннектор** | `s3-mounts: [<connector_id>]`, в `cmd` — `${<connector_id>}` | Веса уже в объектном хранилище |

---

## 8. Грабли (все проверены или выведены из исходников CLI 0.10.0)

1. **`args` — не шаблоны.** `{{FOO}}` и `$FOO` не раскрываются; `args` — это
   дополнительные аргументы командной строки. Динамические пути — только через
   `${VAR}` от `inputs`/`outputs`.
2. **`env.python: manual` требует `local-paths`.** `is_fully_manual` истинно только
   когда заданы `version`, `requirements-file` **и** `local-paths`; иначе CLI
   запускает envzy-разбор и пытается импортировать точку входа **на ноутбуке**
   (`ModuleNotFoundError`), то есть требует torch в окружении самого CLI. Поэтому
   `local-paths` в шаблонах — не оптимизация, а необходимость.
3. **Никаких маркеров и URL в requirements-файле** — CLI валидирует строки через
   `packaging.Requirement` и падает на `;` и на `http(s)://`.
4. **`cmd` должен начинаться с интерпретатора Python** (`python3 ...`), иначе CLI
   не найдёт главный модуль (`Python root module(-s) was not found`). `bash -c ...`
   в `cmd` работает, только если окружение не задавать вовсе. Всё шелл-подобное
   (симлинки, скачивание весов) делает драйвер.
5. **Неизвестный `--model`**: пути в реестре (`models/Qwen3-0.6B`) относительны
   рабочего каталога, поэтому `./models` внутри job должен существовать. Драйвер
   создаёт его симлинком на `${WEIGHTS}`.
6. **Пересечение `local-paths` и `inputs`.** CLI предупреждает, если локальный
   модуль и входной путь вложены друг в друга. В шаблонах пересечения нет (код
   уезжает через `local-paths`, веса — через `inputs`); предупреждение появится,
   если начать передавать репозиторий целиком как `inputs`.
7. **Перезапись результатов.** Автоскачивание кладёт файлы по относительному пути
   `outputs`. Прогон с `--out-prefix results` затрёт локальные прогоны ноутбука.
8. **Пути в конфиге** (`inputs`, `requirements-file`, `local-paths`) разрешаются от
   CWD CLI — запускай `datasphere` из корня репозитория.
9. **`models/` не должна уезжать целиком** вместе с кодом: `.venv` (1.2 ГБ) и
   `models/` (3.2 ГБ) в job не нужны как код. Именно поэтому код передаётся через
   `local-paths`, а не как каталог репозитория.
10. **Токен и вкладки браузера.** `YC_IAM_TOKEN` не обновляется (CLI предупреждает
    об этом) и живёт ~12 ч; в конфиг и в git его не писать. Вкладку
    `console.yandex.cloud` открывает `yc`, когда истекла федеративная сессия: пока
    сессия свежая, `yc` молчит, после истечения — переаутентифицируется через
    браузер (`YC_NO_BROWSER=1` не помогает, а песочница не даёт `yc` сохранить
    обновлённую сессию, поэтому там вкладка появляется на каждый вызов). Бери токен
    через `scripts/datasphere_auth.sh`, а не «руками» в цикле.
11. **`WARNING: Cannot connect to YC tool initialization service`.** Это проверка
    версии CLI, к авторизации и заданиям отношения не имеет; глушится
    `export YC_CLI_INITIALIZATION_SILENCE=true`. Похоже на ту же сетевую
    особенность WSL, из-за которой часть хостов резолвится только в IPv6 (`curl -6`
    в этой сети отдаёт «network unreachable»).

---

## 9. Стоимость, уборка, следы в проекте

* Задание тарифицируется по конфигурации и времени; самый дешёвый шаг — smoke на
  `c1.4`. Актуальные цены: <https://yandex.cloud/ru/docs/datasphere/pricing>.
* Долгие прогоны запускай с `--async`, а не с открытым стримом логов.
* История заданий видна на вкладке **DataSphere Jobs** проекта; лишние
  probe-задания можно удалить: `$CLI project job delete --id <job_id>`.
* В этом репозитории остаются только `ds-results/` (артефакты) и
  `configs/datasphere/*.yaml` (конфиги, в git). Всё остальное — в проекте
  DataSphere.  В `.cache/` лежит только кеш IAM-токена — он одноразовый и
  игнорируется.

---

## 10. Чеклист агента

1. `$CLI project get --id bt1u5v72b71eesdhp9k5` — доступ есть.
2. `source scripts/datasphere_auth.sh` — получить `YC_IAM_TOKEN` (не вызывай
   `yc iam create-token` в цикле: в WSL это вкладка консоли на каждый вызов).
3. Создать `configs/datasphere/smoke.yaml` из §3.2 и запустить его.
4. Убедиться: статус `SUCCESS`, в логах видны строки драйвера `[entry] === detect: ...`,
   локально появился `ds-results/qwen3-0.6b/scores_next_step.json`.
5. Выбрать масштаб: `laptop` (§4.1) или `paper` (§4.2); для GPU сначала проверить
   разрешённую в сообществе конфигурацию и квоту.
6. Запустить рабочий прогон; при `--async` следить через `job get` / `job attach`.
7. Проверить артефакты: `ds-results/<key>/` заполнен, `ds-results/figures/*.pdf`
   на месте; при необходимости — `job download-files --id <job_id>`.
8. Отчитаться: job id (ссылка), конфигурация, время, список полученных файлов и
   расхождения с локальными `results/`.

Если что-то расходится с §2 (например, GPU-конфигурация не разрешена, или
`--download-weights` не смог достать веса), это ожидаемо неизвестный участок:
зафиксируй факт в отчёте, не «подгоняй» инструкцию молча.
