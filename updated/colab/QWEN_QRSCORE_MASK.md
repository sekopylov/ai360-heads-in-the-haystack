# Qwen3-8B no-thinking Paul Graham masking по QRScore

Конфиг: `qwen3-8b-no-thinking-paul-graham-qrscore-mask.yaml`.
Источник рейтинга — завершённый `qwen3-8b-no-thinking-qrscore-survey`, run_id
`4613d57e-887d-4e59-8328-08ff936c2322`. На Drive уже есть
`detection/aggregation/head_scores_query_retrieval_score_v1.json`: 1152 головы,
по 27 значений. Рейтинг — среднее по всем случаям, без фильтра по качеству ответа.

Перед masking worker пересобирает рейтинг из исходных результатов через
`aggregate_head_scores.aggregate(..., all_cases=True)` в отдельную папку нового
прогона `ranking/`; сохраняет историю, отсортированные `mean_scores.json`,
SHA256 и provenance. Исходный survey и старые multiset YAML не изменяются.

Протокол совпадает с прежним GLM Paul Graham masking: 11 длин × 10 глубин,
baseline + top/bottom 5/10/15/20/40/60/80/100, всего 1870 генераций.
FP16 Qwen, seed/context_seed=42, один контекст, max_new_tokens=256,
Flash prefill/decode, legacy_uniform Q-head masking только при decode.
Используется неизменённый `datasphere/job.py --profile mask`; QRScore лишь
задаёт порядок голов, не вычисляется заново на Paul Graham.

После подключения Drive к существующей `qwen3-qrscore-mask`:

```bash
colab --auth oauth2 exec -s qwen3-qrscore-mask --timeout 86400 -f updated/colab/launch_qwen_qrscore_mask.py
```

Архив должен быть загружен как `/content/retrieval-heads-mask.zip`.
Результаты: `MyDrive/retrieval-heads-colab/qwen3-8b-no-thinking-paul-graham-qrscore-mask/`.
Лог и status JSON находятся рядом с папкой результатов. Повторный запуск в
существующую папку запрещён без явно выбранного режима продолжения.

После обрыва 11 октября добавлен opt-in `--resume`: проверяется исходный YAML,
ревизия и SHA256 замороженного рейтинга. Сохранённые результаты проверяются
по заново собранному prompt_sha256, модели, case, параметрам и blocked_heads;
только затем пропускаются без генерации и перезаписи. Несоответствие вызывает
ошибку. Обычный запуск без resume работает как прежде; random resume запрещён.
Каждая попытка имеет отдельный bootstrap log и runtime record.

Для созданной сессии продолжения:

```bash
colab --auth oauth2 drivemount -s qwen3-qrscore-mask-resume
colab --auth oauth2 exec -s qwen3-qrscore-mask-resume --timeout 86400 -f updated/colab/resume_qwen_qrscore_mask.py
```

Прогресс включает проверенные сохранённые случаи: их строки явно помечены
`[resume] skipped validated saved case`; новые генерации по-прежнему дают `score=`.

## Быстрое продолжение после повторного отключения

`mask_resume_plan.py` перед скачиванием/загрузкой модели проверяет manifest каждого
блока: параметры, список голов из замороженного рейтинга, completed_cases и
полную сетку имён атомарно записанных результатов. Блок с complete=true и
точной сеткой пропускается целиком, без чтения/токенизации каждого промпта и
без запуска masking subprocess. Это структурная проверка завершённого блока,
не повторная проверка prompt_sha256 каждого его результата.

Незавершённые блоки сохраняют строгую проверку prompt_sha256 и полный исходный
grid (важно для неизменного seeded corpus window). 12 октября осталось только
top100 18 случаев + bottom100 110: загружается модель только для этих двух
блоков. 15 завершённых блоков не используют GPU. Рейтинг, FP16, RoPE,
seed policy, prompts и masking unchanged. Повреждённый manifest/неполная
сетка complete-блока вызывает ошибку, а не молчаливый пропуск.

## Ограничения непрерывного выполнения

Проверка `colab --auth oauth2 usage` 12 октября: 178.93 CU, активных VM нет.
Причина двух удалений VM неизвестна; текущий положительный баланс не является
историческим аудитом баланса в момент отключения. `colab exec --timeout 86400`
задаёт таймаут клиента, а не гарантированный срок аренды GPU.

Запуск выполняется в активной foreground ячейке. По официальной документации
CLI backend учитывает активность ядра; keep-alive daemon не используется:
https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md

Consumer Colab не гарантирует непрерывную работу даже с платным тарифом.
Pro+ поддерживает до 24 часов непрерывного выполнения при достаточном балансе;
это не гарантия отсутствия сбоев. Для ресурса без Colab-таймаутов Google
предлагает выделенную VM, например через GCP Marketplace. Это отдельная
оплата/настройка и требует выбора пользователя; автоматически не подключалось.
Обычный drive.mount на выделенных/Enterprise/local runtime не поддерживается,
поэтому хранение потребуется перенастроить. Источник:
https://research.google.com/colaboratory/faq.html

Искусственная активность/обход ограничений не добавлялись. Быстрый resume
минимизирует цену обрывов, но не предотвращает серверное удаление VM.
