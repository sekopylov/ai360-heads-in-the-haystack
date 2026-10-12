# GLM: decode attention mass на вопрос и needle

Гипотеза: при генерации ответа часть голов читает информацию из контекстных
представлений вопроса, поэтому их внимание к исходной needle может быть небольшим.
Этот survey измеряет направление внимания; перенос содержания требует отдельного
причинного вмешательства.

## Протокол

Отдельный запуск `glm4-6v-flash-no-thinking-question-mass-survey` повторяет
конфиг прошлого GLM QRScore survey: `zai-org/GLM-4.6V-Flash`, revision
`411bb4d77144a3f03accbf4b780f5acb8b7cde4e`, BF16/A100, no-thinking,
context-only prompt адаптера, Flash SDPA prefill, greedy observed eager decode.
Три прежних detection-case × длины 8000/16000/30000 × глубины 15/45/75 = 27
генераций; context seed 42, max_new_tokens 2048. Здесь survey, без masking.

На каждой генерации одновременно считаются:

- `query_retrieval_score_v1`: прежний отдельный prefill, вопрос → полная needle;
- `needle_attention_mass_v1`: прежний decode → span expected_answer;
- `question_attention_mass_v1`: тот же decode → span входного вопроса.

Для двух decode-метрик сумма вероятностей на целевом span усредняется по **одним
и тем же шагам**, где generated token ID встречается в span expected_answer.
Вопрос не включает инструкции шаблона или `Answer:`. Деления на длину целевого
span нет. При отсутствии таких шагов score = 0, qualifying_steps = 0: это отсутствие
наблюдений, а не свидетельство нулевого внимания. Число qualifying_steps и
question_attention_span сохраняются в JSON. Валидация проверяет парность шагов.

Каждая метрика содержит 1280 Q-голов (40 × 32), индексация с нуля.
Агрегация выполняется по всем 27 случаям, без отбора по качеству ответа.
Полные attention-матрицы не сохраняются: на GPU суммируются нужные spans,
на CPU передаются только по одному значению на голову для каждого span.

## Запуск

Из корня репозитория подготовить новый bundle:

```bash
python3 updated/colab/prepare_bundle.py --output /tmp/retrieval-heads-glm-question-mass.zip
```

В уже выделенной A100-сессии с подключённым Google Drive:

```bash
colab --auth oauth2 upload -s SESSION /tmp/retrieval-heads-glm-question-mass.zip /content/retrieval-heads-survey.zip
colab --auth oauth2 exec -s SESSION --timeout 86400 -f updated/colab/launch_glm_question_mass_survey.py
```

Результаты: `MyDrive/retrieval-heads-colab/glm4-6v-flash-no-thinking-question-mass-survey/`.
Рядом сохраняются bootstrap log и status JSON. Launcher запрещает повторный
запуск поверх существующего каталога/лога. Этот survey launcher не устанавливает
автоматический stop VM; остановка сессии выполняется отдельно после сохранения
результатов. Подготовка кода сама GPU-запуск не выполняет.

В `detection/results/` лежат три score-набора каждого случая; в
`detection/aggregation/` — три истории средних оценок для построения рейтингов.
Для локального checkpoint можно собрать команду функцией
`colab.run_smoke.command(config, model_path, output_path)` с новым YAML, затем
запустить `aggregate_head_scores.py --run OUTPUT --all-cases`.

## Анализ

Сравнить вопрос/needle по каждой голове на одних случаях и посмотреть
13 голов из прежнего QRScore top-20, не попавших в прежний mass top-20.
Высокая question mass при низкой needle mass согласуется с гипотезой, но сама
по себе не доказывает наличие ответа в представлениях вопроса.
Не складывать два score без проверки непересечения spans.
Для сравнения с прежними результатами сначала проверить prompt_sha256.
Новый mass-рейтинг агрегирован по всем случаям, как QRScore; прежний GLM
mass-рейтинг использовал порог успешности, который тогда прошли все 27 случаев.
