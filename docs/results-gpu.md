# Results — `ds-results`

## Architecture census

| model | layers | scoreable layers | scoreable heads | head_dim | hybrid |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 28 | 448 | 128 | False |
| Qwen3.5-0.8B | 24 | 6 | 48 | 256 | True |

## Retrieval-head detection

| model | pairing | instances | recited | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | next_step | 75 | 73/75 | 0.958 | `L16H14` | 0.86 | 22/448 (4.9%) | 5/448 (1.1%) |
| Qwen3.5-0.8B | next_step | 75 | 53/75 | 0.701 | `L11H1` | 0.74 | 30/48 (62.5%) | 8/48 (16.7%) |

## Pairing sensitivity

| model | top-10 overlap between pairings | next_step top heads | same_step top heads |
|---|---|---|---|
| Qwen3-0.6B | 0/10 | L16H14, L21H8, L20H14, L18H5, L6H11 | L6H6, L2H10, L11H2, L1H15, L6H7 |
| Qwen3.5-0.8B | 0/10 | L11H1, L15H7, L19H5, L23H0, L23H5 | L7H7, L11H3, L3H7, L7H6, L7H2 |

## Masking curve

| model | K | % heads | retrieval f1 | retrieval exact | random f1 | random exact |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 9 | 2.0% | 66.5 | 0.0 | 95.5 | 100.0 |
| Qwen3-0.6B | 18 | 4.0% | 67.2 | 0.0 | 50.9 | 33.3 |
| Qwen3-0.6B | 36 | 8.0% | 56.8 | 0.0 | 51.7 | 38.9 |
| Qwen3-0.6B | 76 | 17.0% | 56.8 | 0.0 | 3.3 | 0.0 |
| Qwen3-0.6B | 148 | 33.0% | 56.8 | 0.0 | 0.0 | 0.0 |
| | baseline | | 95.5 | 100.0 | | |
| Qwen3.5-0.8B | 1 | 2.1% | 84.0 | 100.0 | 89.3 | 100.0 |
| Qwen3.5-0.8B | 2 | 4.2% | 79.0 | 100.0 | 92.9 | 100.0 |
| Qwen3.5-0.8B | 4 | 8.3% | 45.1 | 0.0 | 69.5 | 66.7 |
| Qwen3.5-0.8B | 8 | 16.7% | 70.4 | 100.0 | 75.5 | 61.1 |
| Qwen3.5-0.8B | 16 | 33.3% | 32.1 | 0.0 | 35.3 | 0.0 |
| | baseline | | 86.7 | 100.0 | | |

## Token-mixer ablation

**qwen3.5-0.8b** — 6 full-attention layers, 18 linear layers, baseline f1=86.7

| K | full-attention masked | linear masked |
|---|---|---|
| 1 | 100.0 | 0.0 |
| 2 | 85.0 | 1.2 |
| 4 | 42.6 | 1.2 |


## Downstream tasks

**qwen3-0.6b — extractive QA**: baseline F1 67.5 over 8 samples

| K | % heads | retrieval F1 (drop) | random F1 (drop) |
|---|---|---|---|
| 18 | | 24.4 (43.1) | 55.1 (12.4) |
| 36 | | 16.2 (51.3) | 20.3 (47.2) |
| 76 | | 16.2 (51.3) | 4.2 (63.3) |

**qwen3-0.6b — chain-of-thought**: K=36, 8 samples

| variant | baseline | retrieval masked | random masked |
|---|---|---|---|
| answer_only | 75.0 | 50.0 | 12.5 |
| cot | 100.0 | 75.0 | 81.2 |

**qwen3.5-0.8b — extractive QA**: baseline F1 100.0 over 8 samples

| K | % heads | retrieval F1 (drop) | random F1 (drop) |
|---|---|---|---|
| 2 | | 3.9 (96.1) | 61.4 (38.6) |
| 4 | | 87.5 (12.5) | 97.2 (2.8) |
| 8 | | 44.6 (55.4) | 30.0 (70.0) |

**qwen3.5-0.8b — chain-of-thought**: K=4, 8 samples

| variant | baseline | retrieval masked | random masked |
|---|---|---|---|
| answer_only | 75.0 | 25.0 | 68.8 |
| cot | 100.0 | 50.0 | 93.8 |


## Cross-model correlation

Retrieval-score correlation (mode `sorted`):

| | Qwen3.5-0.8B | Qwen3-0.6B |
|---|---|---|
| Qwen3.5-0.8B | 1.00 | 0.93 |
| Qwen3-0.6B | 0.93 | 1.00 |

## Artifacts

- `correlation.json`
- `figures/corr_map.pdf`
- `figures/corr_map.png`
- `figures/heat_map.pdf`
- `figures/heat_map.png`
- `figures/layer_profile.pdf`
- `figures/layer_profile.png`
- `figures/masking_heads.pdf`
- `figures/masking_heads.png`
- `figures/mixer_ablation.pdf`
- `figures/mixer_ablation.png`
- `figures/ring_graph.pdf`
- `figures/ring_graph.png`
- `figures/score_distribution.pdf`
- `figures/score_distribution.png`
- `figures/task_cot.pdf`
- `figures/task_cot.png`
- `figures/task_qa.pdf`
- `figures/task_qa.png`
- `overlap.json`
- `qwen3-0.6b/instances_next_step.jsonl`
- `qwen3-0.6b/masking_curve.json`
- `qwen3-0.6b/model_info.json`
- `qwen3-0.6b/scores_next_step.json`
- `qwen3-0.6b/scores_next_step.npz`
- `qwen3-0.6b/summary_next_step.json`
- `qwen3-0.6b/task_cot.json`
- `qwen3-0.6b/task_qa.json`
- `qwen3.5-0.8b/instances_next_step.jsonl`
- `qwen3.5-0.8b/masking_curve.json`
- `qwen3.5-0.8b/mixer_ablation.json`
- `qwen3.5-0.8b/model_info.json`
- `qwen3.5-0.8b/scores_next_step.json`
- `qwen3.5-0.8b/scores_next_step.npz`
- `qwen3.5-0.8b/summary_next_step.json`
- `qwen3.5-0.8b/task_cot.json`
- `qwen3.5-0.8b/task_qa.json`
