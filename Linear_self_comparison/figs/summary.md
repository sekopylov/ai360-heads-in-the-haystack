# Retrieval heads in Qwen3.5: linear vs full attention

Scores: |A|·||v|| (contrib); detection used 571/600 successful samples.

| head type | #heads | max score | #score>=0.1 | %score>=0.1 | #score>=0.3 | mean score |
|---|---|---|---|---|---|---|
| linear | 288 | 0.036 | 0 | 0.0% | 0 | 0.0002 |
| full | 48 | 0.854 | 27 | 56.2% | 16 | 0.2188 |

## Top-20 heads

| rank | head (layer-head) | type | score |
|---|---|---|---|
| 1 | 15-7 | full | 0.854 |
| 2 | 11-1 | full | 0.700 |
| 3 | 23-0 | full | 0.665 |
| 4 | 19-5 | full | 0.609 |
| 5 | 23-1 | full | 0.535 |
| 6 | 23-5 | full | 0.524 |
| 7 | 19-6 | full | 0.493 |
| 8 | 19-7 | full | 0.477 |
| 9 | 23-3 | full | 0.476 |
| 10 | 23-7 | full | 0.470 |
| 11 | 19-2 | full | 0.452 |
| 12 | 15-5 | full | 0.383 |
| 13 | 19-0 | full | 0.379 |
| 14 | 19-3 | full | 0.375 |
| 15 | 15-3 | full | 0.345 |
| 16 | 19-1 | full | 0.328 |
| 17 | 15-2 | full | 0.261 |
| 18 | 15-1 | full | 0.233 |
| 19 | 11-2 | full | 0.198 |
| 20 | 11-4 | full | 0.168 |

## NIAH mean score after masking

no masking: **100.0**

| setting | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| full random | 100.0 ±0.0 | 97.4 ±3.7 | 99.6 ±0.1 | 99.4 ±0.5 | 60.3 ±27.2 |
| full top | 100.0 | 100.0 | 100.0 | 95.7 | 34.2 |
| linear random | 100.0 ±0.0 | 100.0 ±0.0 | 100.0 ±0.0 | 100.0 ±0.0 | 100.0 ±0.0 |
| linear top | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
