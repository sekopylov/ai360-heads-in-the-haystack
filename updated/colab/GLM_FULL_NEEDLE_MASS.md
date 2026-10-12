# GLM: mass на полной needle → top-5/10/15/20/40 masking

Проверка: меняет ли рейтинг и причинный эффект маскирования сумма decode attention
на **всей вставленной needle**, вместо более короткого expected_answer.

## Этап 1: detection и рейтинг

Модель и checkpoint прежние: GLM-4.6V-Flash, revision
`411bb4d77144a3f03accbf4b780f5acb8b7cde4e`, BF16/A100, no-thinking,
context-only prompt, greedy, Flash prefill и observed eager decode.
Три detection-case × длины 8000/16000/30000 × глубины 15/45/75 = **27 случаев**.
Context seed42, max_new_tokens2048.

С одного attention forward одновременно собираются старый `needle_attention_mass_v1`
и новый `full_needle_attention_mass_v1`. В дополнительном prefill собирается QRScore.
Новый full-needle span определяется точным вхождением `case.needle` в prompt,
тем же locator, что gold span QRScore. Span сохраняется в каждом JSON.

**Меняется целевой span, правило отбора decode-шагов прежнее:** учитывается шаг,
если generated token ID встречается в историческом expected_answer span.
Обе decode-метрики используют те же шаги, частоты и одинаковый denominator.
Это тест границ целевого span, а не одновременная замена token gate.
QRScore использует ту же полную needle, но на позиции входного вопроса во время prefill.
Новые scores конечны и находятся в [0,1]; отсутствие qualifying steps даёт 0.
Все 27 случаев входят в агрегацию поровну, независимо от ROUGE.

Результаты: `glm4-6v-flash-no-thinking-full-needle-mass-survey/` на Drive.
Старые/новые decode-счётчики шагов проверяются на равенство, а full span —
на совпадение с gold span QRScore. Сохраняется и исторический qualifying span
для последующей проверки locator.

## Этап 2: маскирование

Рейтинг из **нового full_needle_attention_mass_v1**, все 1280 Q-голов.
Launcher фиксирует конкретный run_id только после завершения survey и validation.
Копия рейтинга, SHA256, конфиг, источник и run_id сохраняются рядом с masking.

- Baseline: 0 голов.
- Top-5, top-10, top-15, top-20, top-40.
- 11 прежних длин 1000–30000 × 10 глубин = 110 случаев на условие.
- Всего **660 генераций** (6 × 110), один Paul Graham/San Francisco case.
- BF16/A100, Flash prefill/decode, no-thinking, max_new_tokens256, seed42.
- `legacy_uniform` только во время decode, Q выбранных голов; K/V и prefill прежние.

Итого новый workflow: **27 detection + 660 masking = 687 генераций**.
Detection и validation-корпусы разделены. Bottom/random условия не включены.
Masking output: `glm4-6v-flash-no-thinking-paul-graham-full-needle-mass-top-mask/`.
Завершённый masking автоматически архивируется в ZIP на Drive.

## Запуск в Colab

Из корня репозитория:

```bash
python3 updated/colab/prepare_bundle.py --output /tmp/retrieval-heads-glm-full-needle.zip
colab --auth oauth2 upload -s SESSION /tmp/retrieval-heads-glm-full-needle.zip /content/retrieval-heads-full-needle.zip
colab --auth oauth2 exec -s SESSION --timeout 86400 -f updated/colab/launch_glm_full_needle_experiment.py
```

Нужны выделенная A100 и подключённый Drive. Launcher выполняет два этапа
последовательно и запрещает повторный запуск поверх существующих каталогов.
`ranking_source_run_id: null` в локальном YAML — шаблон: standalone mask_worker
его не принимает; workflow связывает его с валидированным только что созданным survey.
Автоматического выключения VM у этого workflow нет. После завершения скачать
и проверить ZIP, затем остановить сессию. Подготовка кода GPU-job не запускает.

## Сравнение

В survey сравнить новый full-needle рейтинг со старым mass и QRScore:
Pearson/Spearman, top-k пересечения, изменения конкретных голов.
Для masking сопоставить baseline/top5/10/15/20/40 с прежними GLM mass/QRScore
на совпадающих prompt_sha256: ROUGE-1/2/3 recall, точная фраза, EOS/обрывы.
Полная needle у разных detection-case отличается от expected_answer разным числом
токенов; прирост score трактовать с учётом этого. Масса на расширенном span должна
не уменьшаться, если сохранённый старый span целиком внутри полного span.
Нарушение этого отношения сигнализирует о проблеме исторического locator,
а не автоматически об ошибке новой метрики.
