# GLM-4.6V-Flash: Paul Graham — multiset

Job `bt1af8nq5baf9a9t6b6f`; 1870/1870 результатов.

11 длин ×10 глубин ×17 условий: baseline, top/bottom k=5,10,15,20,40,60,80,100.
Prompt-хэши совпадают между всеми условиями. Маска legacy_uniform при decode; K/V и prefill не менялись.

Метрики пересчитаны по сохранённому model_response тем же кодом, что для Qwen. Исходные JSON не изменены.
ROUGE recall со stemming; среднее 1/2/3 — сводная нестандартная метрика. Exact match — присутствие полной
нормализованной эталонной фразы подряд, не семантическое совпадение. Recall не штрафует лишний текст.
Все результаты, включая обрывы по лимиту, включены. Сравнение top/bottom не доказывает специализацию голов.

| Условие | ROUGE-1 | ROUGE-2 | ROUGE-3 | Среднее | Exact | По лимиту |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0 |
| top5 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0 |
| top10 | 98.18 | 98.02 | 97.82 | 98.01 | 94.55 | 0 |
| top15 | 90.30 | 89.42 | 88.36 | 89.36 | 70.91 | 0 |
| top20 | 75.76 | 73.55 | 70.91 | 73.41 | 27.27 | 0 |
| top40 | 84.02 | 82.40 | 80.55 | 82.32 | 51.82 | 0 |
| top60 | 64.77 | 60.33 | 55.09 | 60.06 | 1.82 | 0 |
| top80 | 45.76 | 40.50 | 34.45 | 40.24 | 0.00 | 0 |
| top100 | 25.08 | 18.10 | 10.00 | 17.72 | 0.00 | 0 |
| bottom5 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0 |
| bottom10 | 99.70 | 99.67 | 99.64 | 99.67 | 99.09 | 0 |
| bottom15 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0 |
| bottom20 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0 |
| bottom40 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 56 |
| bottom60 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 106 |
| bottom80 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 93 |
| bottom100 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 110 |

[CSV случаев](cases.csv) · [Сводка](summary.csv) · [JSON оценок](scores.json)

## baseline: ROUGE-1/2/3, среднее и exact match

![baseline: ROUGE-1/2/3, среднее и exact match](plots/baseline_heatmaps.png)

[SVG](plots/baseline_heatmaps.svg)

## top5: ROUGE-1/2/3, среднее и exact match

![top5: ROUGE-1/2/3, среднее и exact match](plots/top5_heatmaps.png)

[SVG](plots/top5_heatmaps.svg)

## top10: ROUGE-1/2/3, среднее и exact match

![top10: ROUGE-1/2/3, среднее и exact match](plots/top10_heatmaps.png)

[SVG](plots/top10_heatmaps.svg)

## top15: ROUGE-1/2/3, среднее и exact match

![top15: ROUGE-1/2/3, среднее и exact match](plots/top15_heatmaps.png)

[SVG](plots/top15_heatmaps.svg)

## top20: ROUGE-1/2/3, среднее и exact match

![top20: ROUGE-1/2/3, среднее и exact match](plots/top20_heatmaps.png)

[SVG](plots/top20_heatmaps.svg)

## top40: ROUGE-1/2/3, среднее и exact match

![top40: ROUGE-1/2/3, среднее и exact match](plots/top40_heatmaps.png)

[SVG](plots/top40_heatmaps.svg)

## top60: ROUGE-1/2/3, среднее и exact match

![top60: ROUGE-1/2/3, среднее и exact match](plots/top60_heatmaps.png)

[SVG](plots/top60_heatmaps.svg)

## top80: ROUGE-1/2/3, среднее и exact match

![top80: ROUGE-1/2/3, среднее и exact match](plots/top80_heatmaps.png)

[SVG](plots/top80_heatmaps.svg)

## top100: ROUGE-1/2/3, среднее и exact match

![top100: ROUGE-1/2/3, среднее и exact match](plots/top100_heatmaps.png)

[SVG](plots/top100_heatmaps.svg)

## bottom5: ROUGE-1/2/3, среднее и exact match

![bottom5: ROUGE-1/2/3, среднее и exact match](plots/bottom5_heatmaps.png)

[SVG](plots/bottom5_heatmaps.svg)

## bottom10: ROUGE-1/2/3, среднее и exact match

![bottom10: ROUGE-1/2/3, среднее и exact match](plots/bottom10_heatmaps.png)

[SVG](plots/bottom10_heatmaps.svg)

## bottom15: ROUGE-1/2/3, среднее и exact match

![bottom15: ROUGE-1/2/3, среднее и exact match](plots/bottom15_heatmaps.png)

[SVG](plots/bottom15_heatmaps.svg)

## bottom20: ROUGE-1/2/3, среднее и exact match

![bottom20: ROUGE-1/2/3, среднее и exact match](plots/bottom20_heatmaps.png)

[SVG](plots/bottom20_heatmaps.svg)

## bottom40: ROUGE-1/2/3, среднее и exact match

![bottom40: ROUGE-1/2/3, среднее и exact match](plots/bottom40_heatmaps.png)

[SVG](plots/bottom40_heatmaps.svg)

## bottom60: ROUGE-1/2/3, среднее и exact match

![bottom60: ROUGE-1/2/3, среднее и exact match](plots/bottom60_heatmaps.png)

[SVG](plots/bottom60_heatmaps.svg)

## bottom80: ROUGE-1/2/3, среднее и exact match

![bottom80: ROUGE-1/2/3, среднее и exact match](plots/bottom80_heatmaps.png)

[SVG](plots/bottom80_heatmaps.svg)

## bottom100: ROUGE-1/2/3, среднее и exact match

![bottom100: ROUGE-1/2/3, среднее и exact match](plots/bottom100_heatmaps.png)

[SVG](plots/bottom100_heatmaps.svg)

## ROUGE-1 recall: длины, глубины и k; top и bottom

![ROUGE-1 recall: длины, глубины и k; top и bottom](plots/comparison_rouge1_recall.png)

[SVG](plots/comparison_rouge1_recall.svg)

## ROUGE-2 recall: длины, глубины и k; top и bottom

![ROUGE-2 recall: длины, глубины и k; top и bottom](plots/comparison_rouge2_recall.png)

[SVG](plots/comparison_rouge2_recall.svg)

## ROUGE-3 recall: длины, глубины и k; top и bottom

![ROUGE-3 recall: длины, глубины и k; top и bottom](plots/comparison_rouge3_recall.png)

[SVG](plots/comparison_rouge3_recall.svg)

## Среднее ROUGE-1/2/3 recall: длины, глубины и k; top и bottom

![Среднее ROUGE-1/2/3 recall: длины, глубины и k; top и bottom](plots/comparison_rouge123_mean_recall.png)

[SVG](plots/comparison_rouge123_mean_recall.svg)

## Точная фраза иголки: длины, глубины и k; top и bottom

![Точная фраза иголки: длины, глубины и k; top и bottom](plots/comparison_exact_match.png)

[SVG](plots/comparison_exact_match.svg)

## k=5: парное изменение относительно baseline

![k=5: парное изменение относительно baseline](plots/delta_baseline_k005.png)

[SVG](plots/delta_baseline_k005.svg)

## k=10: парное изменение относительно baseline

![k=10: парное изменение относительно baseline](plots/delta_baseline_k010.png)

[SVG](plots/delta_baseline_k010.svg)

## k=15: парное изменение относительно baseline

![k=15: парное изменение относительно baseline](plots/delta_baseline_k015.png)

[SVG](plots/delta_baseline_k015.svg)

## k=20: парное изменение относительно baseline

![k=20: парное изменение относительно baseline](plots/delta_baseline_k020.png)

[SVG](plots/delta_baseline_k020.svg)

## k=40: парное изменение относительно baseline

![k=40: парное изменение относительно baseline](plots/delta_baseline_k040.png)

[SVG](plots/delta_baseline_k040.svg)

## k=60: парное изменение относительно baseline

![k=60: парное изменение относительно baseline](plots/delta_baseline_k060.png)

[SVG](plots/delta_baseline_k060.svg)

## k=80: парное изменение относительно baseline

![k=80: парное изменение относительно baseline](plots/delta_baseline_k080.png)

[SVG](plots/delta_baseline_k080.svg)

## k=100: парное изменение относительно baseline

![k=100: парное изменение относительно baseline](plots/delta_baseline_k100.png)

[SVG](plots/delta_baseline_k100.svg)

## Длина выхода, обрывы и длительность

![Длина выхода, обрывы и длительность](plots/generation_diagnostics.png)

[SVG](plots/generation_diagnostics.svg)
