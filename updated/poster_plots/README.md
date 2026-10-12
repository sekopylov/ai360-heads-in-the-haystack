# Генераторы графиков для плаката

| Скрипт | Что строит |
|---|---|
| `plot_poster_glm_score_pies.py` | Три круговые диаграммы распределения голов по score |
| `plot_poster_glm_head_scores.py` | Три тепловые карты всех голов: слои по X, головы по Y |
| `plot_poster_glm_top40.py` | Зелёно-красная сетка 4×3 для 20/40/80 голов и отдельные карты top20/40/60/80 |
| `plot_poster_glm_archived.py` | Архивные варианты top100 и сетка 40/80/100 |

Запуск из корня репозитория, с существующим окружением `updated/.venv`:

```bash
updated/.venv/bin/python updated/poster_plots/plot_poster_glm_score_pies.py
updated/.venv/bin/python updated/poster_plots/plot_poster_glm_head_scores.py
updated/.venv/bin/python updated/poster_plots/plot_poster_glm_top40.py
```

Скрипты можно запускать и из другой рабочей директории, указав абсолютные пути.
Также из `updated` доступен модульный запуск, например:

```bash
.venv/bin/python -m poster_plots.plot_poster_glm_score_pies
```

Результаты остаются в [report-assets/poster/glm-top40-three-metrics](../report-assets/poster/glm-top40-three-metrics/): PNG, SVG, PDF и таблицы. Фон прозрачный.
Исходные survey/masking данные не перемещены. Генераторы используют общие загрузчики из `updated` и проверяют совпадение запросов, рейтингов и полноту данных.

Скрипты схем метрик `generate_retrieval_metrics.py` и `draw_poster_attention_mass_scheme.py` остаются в `updated`: они рисуют схемы, а не эти графики.
