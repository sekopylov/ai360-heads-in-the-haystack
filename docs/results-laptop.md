# Results — `results`

## Architecture census

| model | layers | scoreable layers | scoreable heads | head_dim | hybrid |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 28 | 448 | 128 | False |
| Qwen3.5-0.8B | 24 | 6 | 48 | 256 | True |

## Retrieval-head detection

| model | pairing | instances | recited | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | next_step | 18 | 18/18 | 1.000 | `L16H14` | 0.86 | 23/448 (5.1%) | 5/448 (1.1%) |
| Qwen3.5-0.8B | next_step | 18 | 18/18 | 0.991 | `L15H7` | 0.80 | 30/48 (62.5%) | 11/48 (22.9%) |

## Pairing sensitivity

| model | top-10 overlap between pairings | next_step top heads | same_step top heads |
|---|---|---|---|
| Qwen3-0.6B | 0/10 | L16H14, L20H14, L6H11, L21H8, L18H5 | L11H2, L6H6, L1H15, L2H10, L2H11 |
| Qwen3.5-0.8B | 0/10 | L15H7, L11H1, L23H5, L23H0, L23H1 | L7H7, L11H3, L3H7, L7H6, L7H2 |

## Masking curve

| model | K | % heads | retrieval f1 | retrieval exact | random f1 | random exact |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 1 | 2.1% | 94.7 | 100.0 | 87.1 | 100.0 |
| Qwen3.5-0.8B | 2 | 4.2% | 89.3 | 100.0 | 84.0 | 100.0 |
| Qwen3.5-0.8B | 4 | 8.3% | 73.2 | 0.0 | 90.1 | 100.0 |
| Qwen3.5-0.8B | 8 | 16.7% | 70.4 | 33.3 | 83.1 | 83.3 |
| Qwen3.5-0.8B | 16 | 33.3% | 32.1 | 0.0 | 30.7 | 0.0 |
| | baseline | | 84.0 | 100.0 | | |

## Token-mixer ablation

**qwen3.5-0.8b** — 6 full-attention layers, 18 linear layers, baseline f1=84.0

| K | full-attention masked | linear masked |
|---|---|---|
| 1 | 84.0 | 1.2 |
| 2 | 87.7 | 0.0 |
| 4 | 24.7 | 0.0 |


## Downstream tasks

**qwen3.5-0.8b — extractive QA**: baseline F1 100.0 over 8 samples

| K | % heads | retrieval F1 (drop) | random F1 (drop) |
|---|---|---|---|
| 2 | | 92.5 (7.5) | 45.8 (54.2) |
| 4 | | 92.5 (7.5) | 92.1 (7.9) |
| 8 | | 88.3 (11.7) | 77.8 (22.2) |


## Artifacts

- `qwen3-0.6b/instances_next_step.jsonl`
- `qwen3-0.6b/scores_next_step.json`
- `qwen3-0.6b/scores_next_step.npz`
- `qwen3-0.6b/summary_next_step.json`
- `qwen3.5-0.8b/instances_next_step.jsonl`
- `qwen3.5-0.8b/masking_curve.json`
- `qwen3.5-0.8b/mixer_ablation.json`
- `qwen3.5-0.8b/scores_next_step.json`
- `qwen3.5-0.8b/scores_next_step.npz`
- `qwen3.5-0.8b/summary_next_step.json`
- `qwen3.5-0.8b/task_qa.json`
