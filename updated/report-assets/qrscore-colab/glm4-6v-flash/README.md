# GLM-4.6V-Flash — QRScore / no thinking

990 случаев masking; survey: все 27 случаев, 1280 Q-голов.

Survey: одна круговая диаграмма количества голов по диапазонам среднего QRScore.
Masking: legacy_uniform только при decode; prefill и K/V не изменены.
ROUGE пересчитан по сохранённым ответам со stemming. Среднее ROUGE-1/2/3 — нестандартный композит.
«Точная фраза» — присутствие нормализованного эталона подряд, не семантическое совпадение.
Все ответы включены; исходные ZIP не изменены. Prompt-хэши совпадают между условиями.

[Сводка](summary.csv) · [Случаи](cases.csv) · [Доли голов](survey_distribution.csv) · [Источники](provenance.json)

## Survey — доли голов (не доли суммарного attention)

![Survey — доли голов (не доли суммарного attention)](plots/survey_head_distribution.png)

[SVG](plots/survey_head_distribution.svg)

## baseline — ROUGE-1/2/3, среднее и точная фраза

![baseline — ROUGE-1/2/3, среднее и точная фраза](plots/baseline_heatmaps.png)

[SVG](plots/baseline_heatmaps.svg)

## top5 — ROUGE-1/2/3, среднее и точная фраза

![top5 — ROUGE-1/2/3, среднее и точная фраза](plots/top5_heatmaps.png)

[SVG](plots/top5_heatmaps.svg)

## top10 — ROUGE-1/2/3, среднее и точная фраза

![top10 — ROUGE-1/2/3, среднее и точная фраза](plots/top10_heatmaps.png)

[SVG](plots/top10_heatmaps.svg)

## top15 — ROUGE-1/2/3, среднее и точная фраза

![top15 — ROUGE-1/2/3, среднее и точная фраза](plots/top15_heatmaps.png)

[SVG](plots/top15_heatmaps.svg)

## top20 — ROUGE-1/2/3, среднее и точная фраза

![top20 — ROUGE-1/2/3, среднее и точная фраза](plots/top20_heatmaps.png)

[SVG](plots/top20_heatmaps.svg)

## top40 — ROUGE-1/2/3, среднее и точная фраза

![top40 — ROUGE-1/2/3, среднее и точная фраза](plots/top40_heatmaps.png)

[SVG](plots/top40_heatmaps.svg)

## top60 — ROUGE-1/2/3, среднее и точная фраза

![top60 — ROUGE-1/2/3, среднее и точная фраза](plots/top60_heatmaps.png)

[SVG](plots/top60_heatmaps.svg)

## top80 — ROUGE-1/2/3, среднее и точная фраза

![top80 — ROUGE-1/2/3, среднее и точная фраза](plots/top80_heatmaps.png)

[SVG](plots/top80_heatmaps.svg)

## top100 — ROUGE-1/2/3, среднее и точная фраза

![top100 — ROUGE-1/2/3, среднее и точная фраза](plots/top100_heatmaps.png)

[SVG](plots/top100_heatmaps.svg)

## top5 — изменение относительно baseline

![top5 — изменение относительно baseline](plots/top5_delta_baseline.png)

[SVG](plots/top5_delta_baseline.svg)

## top10 — изменение относительно baseline

![top10 — изменение относительно baseline](plots/top10_delta_baseline.png)

[SVG](plots/top10_delta_baseline.svg)

## top15 — изменение относительно baseline

![top15 — изменение относительно baseline](plots/top15_delta_baseline.png)

[SVG](plots/top15_delta_baseline.svg)

## top20 — изменение относительно baseline

![top20 — изменение относительно baseline](plots/top20_delta_baseline.png)

[SVG](plots/top20_delta_baseline.svg)

## top40 — изменение относительно baseline

![top40 — изменение относительно baseline](plots/top40_delta_baseline.png)

[SVG](plots/top40_delta_baseline.svg)

## top60 — изменение относительно baseline

![top60 — изменение относительно baseline](plots/top60_delta_baseline.png)

[SVG](plots/top60_delta_baseline.svg)

## top80 — изменение относительно baseline

![top80 — изменение относительно baseline](plots/top80_delta_baseline.png)

[SVG](plots/top80_delta_baseline.svg)

## top100 — изменение относительно baseline

![top100 — изменение относительно baseline](plots/top100_delta_baseline.png)

[SVG](plots/top100_delta_baseline.svg)

## ROUGE-1 recall — длины, глубины и число голов

![ROUGE-1 recall — длины, глубины и число голов](plots/comparison_rouge1_recall.png)

[SVG](plots/comparison_rouge1_recall.svg)

## ROUGE-2 recall — длины, глубины и число голов

![ROUGE-2 recall — длины, глубины и число голов](plots/comparison_rouge2_recall.png)

[SVG](plots/comparison_rouge2_recall.svg)

## ROUGE-3 recall — длины, глубины и число голов

![ROUGE-3 recall — длины, глубины и число голов](plots/comparison_rouge3_recall.png)

[SVG](plots/comparison_rouge3_recall.svg)

## Среднее ROUGE-1/2/3 recall — длины, глубины и число голов

![Среднее ROUGE-1/2/3 recall — длины, глубины и число голов](plots/comparison_rouge123_mean_recall.png)

[SVG](plots/comparison_rouge123_mean_recall.svg)

## Точная фраза иголки — длины, глубины и число голов

![Точная фраза иголки — длины, глубины и число голов](plots/comparison_exact_match.png)

[SVG](plots/comparison_exact_match.svg)

## Диагностика генерации

![Диагностика генерации](plots/generation_diagnostics.png)

[SVG](plots/generation_diagnostics.svg)
