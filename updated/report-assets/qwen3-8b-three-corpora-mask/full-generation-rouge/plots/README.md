# Полная галерея: ROUGE по think + ответу

Все графики оформлены заново. В графиках качества используется пересчитанный full-generation ROUGE-1 recall со stemming. Диагностики остановки, токенов, повторов и выбранных голов используют исходные наблюдения: пересчёт ROUGE не меняет их.

504 файла двух заданий; baseline продублирован, не является независимым повтором. Входной контекст исключён. Ненулевой score не означает правильный ответ: общие слова, гипотезы и отрицания засчитываются.

## Сводные оценки

![Сводные оценки](01_scores.png)

[PNG](01_scores.png) · [SVG](01_scores.svg)

## Завершение генерации

![Завершение генерации](02_termination.png)

[PNG](02_termination.png) · [SVG](02_termination.svg)

## Оценки по длине и глубине

![Оценки по длине и глубине](03_length_depth_scores.png)

[PNG](03_length_depth_scores.png) · [SVG](03_length_depth_scores.svg)

## Пустые финальные ответы

![Пустые финальные ответы](04_length_depth_failures.png)

[PNG](04_length_depth_failures.png) · [SVG](04_length_depth_failures.svg)

## Отдельные корпуса

![Отдельные корпуса](05_corpora.png)

[PNG](05_corpora.png) · [SVG](05_corpora.svg)

## Токены

![Токены](06_tokens.png)

[PNG](06_tokens.png) · [SVG](06_tokens.svg)

## Повторы

![Повторы](07_repetition.png)

[PNG](07_repetition.png) · [SVG](07_repetition.svg)

## Детали внутри think

![Детали внутри think](08_evidence_in_unfinished_think.png)

[PNG](08_evidence_in_unfinished_think.png) · [SVG](08_evidence_in_unfinished_think.svg)

## Распределение голов

![Распределение голов](09_head_layers.png)

[PNG](09_head_layers.png) · [SVG](09_head_layers.svg)

## Изменение оценки

![Изменение оценки](10_score_change.png)

[PNG](10_score_change.png) · [SVG](10_score_change.svg)

## Воспроизведение

Из `updated/`: `.venv/bin/python plot_full_generation_analysis.py`.

[Данные всех графиков](plot_data.json) · [Таблица пересчитанных оценок](../README.md). Исходная галерея answer-only не изменена.