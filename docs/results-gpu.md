# Results — `ds-results`

## Architecture census

| model | layers | scoreable layers | scoreable heads | head_dim | hybrid |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 28 | 448 | 128 | False |
| Qwen3.5-0.8B | 24 | 6 | 48 | 256 | True |

## Retrieval-head detection

| model | pairing | instances | recited | with copy | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | next_step | 75 | 75/75 | 75/75 | 0.963 | `L16H14` | 0.89 | 28/448 (6.2%) | 5/448 (1.1%) |
| _Qwen3-0.6B (recited only)_ | next_step | 75 | (same) | (same) | 0.963 | `L16H14` | 0.89 | 28/448 (6.2%) | 5/448 (1.1%) |
| Qwen3-0.6B | same_step | 75 | 75/75 | 75/75 | 0.963 | `L6H6` | 0.92 | 27/448 (6.0%) | 6/448 (1.3%) |
| _Qwen3-0.6B (recited only)_ | same_step | 75 | (same) | (same) | 0.963 | `L6H6` | 0.92 | 27/448 (6.0%) | 6/448 (1.3%) |
| Qwen3.5-0.8B | next_step | 75 | 75/75 | 75/75 | 0.925 | `L11H1` | 0.74 | 33/48 (68.8%) | 9/48 (18.8%) |
| _Qwen3.5-0.8B (recited only)_ | next_step | 75 | (same) | (same) | 0.925 | `L11H1` | 0.74 | 33/48 (68.8%) | 9/48 (18.8%) |
| Qwen3.5-0.8B | same_step | 75 | 75/75 | 75/75 | 0.925 | `L7H7` | 0.91 | 20/48 (41.7%) | 5/48 (10.4%) |
| _Qwen3.5-0.8B (recited only)_ | same_step | 75 | (same) | (same) | 0.925 | `L7H7` | 0.91 | 20/48 (41.7%) | 5/48 (10.4%) |

- `qwen3-0.6b`/next_step: criterion (2) searched the **prompt** domain, which includes the question and the chat template; the paper's `a in R^{|x|}` is the `haystack` domain (pass `--argmax-domain haystack`, the current default)
- `qwen3-0.6b`/same_step: criterion (2) searched the **prompt** domain, which includes the question and the chat template; the paper's `a in R^{|x|}` is the `haystack` domain (pass `--argmax-domain haystack`, the current default)
- `qwen3.5-0.8b`/next_step: criterion (2) searched the **prompt** domain, which includes the question and the chat template; the paper's `a in R^{|x|}` is the `haystack` domain (pass `--argmax-domain haystack`, the current default)
- `qwen3.5-0.8b`/same_step: criterion (2) searched the **prompt** domain, which includes the question and the chat template; the paper's `a in R^{|x|}` is the `haystack` domain (pass `--argmax-domain haystack`, the current default)

## Cross-model agreement

> **caveat**: mode='sorted' correlates sorted score vectors, not head positions; a high value does not mean the models use the same heads
- correlation: mode `sorted`, models: Qwen3.5-0.8B, Qwen3-0.6B
- head overlap: not comparable (different layer/head layouts); Jaccard suppressed

## Strict-aligned matching

| model | pairing | strict-aligned top heads (mean score) |
|---|---|---|
| Qwen3-0.6B | next_step | L16H14 (0.89), L21H8 (0.60), L20H14 (0.57), L6H11 (0.56), L18H5 (0.53) |
| Qwen3-0.6B | same_step | L6H6 (0.92), L11H2 (0.89), L6H7 (0.87), L2H10 (0.85), L1H15 (0.77) |
| Qwen3.5-0.8B | next_step | L11H1 (0.73), L15H7 (0.70), L19H5 (0.59), L23H1 (0.59), L23H5 (0.57) |
| Qwen3.5-0.8B | same_step | L7H7 (0.89), L11H3 (0.63), L3H7 (0.61), L23H6 (0.53), L23H2 (0.49) |

## Pairing sensitivity

| model | top-10 overlap between pairings | next_step top heads | same_step top heads |
|---|---|---|---|
| Qwen3-0.6B | 0/10 | L16H14, L21H8, L20H14, L6H11, L18H5 | L6H6, L11H2, L6H7, L2H10, L1H15 |
| Qwen3.5-0.8B | 0/10 | L11H1, L15H7, L19H5, L23H1, L23H5 | L7H7, L11H3, L3H7, L23H6, L23H2 |

## Masking curve

| model | K | % heads | retrieval f1 | retrieval exact | random f1 | random exact |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 9 | 2.0% | 93.0 ±3.9 | 50.0 | 95.3 | 63.3 |
| Qwen3-0.6B | 18 | 4.0% | 83.2 ±6.6 | 0.0 | 59.8 | 13.3 |
| Qwen3-0.6B | 36 | 8.0% | 71.0 ±7.0 | 0.0 | 92.6 | 76.7 |
| Qwen3-0.6B | 76 | 17.0% | 18.0 ±3.4 | 0.0 | 14.9 | 0.0 |
| Qwen3-0.6B | 148 | 33.0% | 0.0 | 0.0 | 0.2 | 0.0 |
| | baseline | | 94.3 | 90.0 | | |
| | _qwen3-0.6b: 10 samples per point; ± is the spread across them_ | | | | | |
| Qwen3.5-0.8B | 1 | 2.1% | 42.1 ±13.4 | 0.0 | 51.2 | 0.0 |
| Qwen3.5-0.8B | 2 | 4.2% | 40.2 ±11.0 | 0.0 | 56.4 | 0.0 |
| Qwen3.5-0.8B | 4 | 8.3% | 43.1 ±13.9 | 0.0 | 43.6 | 0.0 |
| Qwen3.5-0.8B | 8 | 16.7% | 35.0 ±9.9 | 0.0 | 65.7 | 0.0 |
| Qwen3.5-0.8B | 16 →15 | 31.2% | 23.0 ±4.4 | 0.0 | 6.6 | 0.0 |
| | baseline | | 46.4 | 0.0 | | |
| | _qwen3.5-0.8b: 10 samples per point; ± is the spread across them_ | | | | | |

## Token-mixer ablation

**qwen3.5-0.8b** — 6 full-attention layers, 18 linear layers, baseline f1=46.4

| K | full-attention masked | linear masked |
|---|---|---|
| 1 | 38.5 ±8.6 | 56.4 ±3.7 |
| 2 | 25.9 ±5.6 | 51.8 ±8.3 |
| 4 | 11.2 ±2.1 | 14.6 ±18.8 |

_± is the spread across the sampled layer subsets (n_trials); this artifact predates `distinct_subsets`, so the number of distinct subsets is not recorded._


## Downstream tasks

**qwen3-0.6b — extractive QA**: baseline F1 67.5 over 8 samples

| K | % heads | retrieval F1 (drop) | random F1 (drop) |
|---|---|---|---|
| 18 | 4.0% | 19.6 (47.9) | 58.8 (8.7) |
| 36 | 8.0% | 10.5 (57.0) | 28.8 (38.7) |
| 76 | 17.0% | 0.0 (67.5) | 0.0 (67.5) |

**qwen3-0.6b — chain-of-thought**: K=36, 8 samples

| variant | baseline | retrieval masked | random masked |
|---|---|---|---|
| answer_only | 75.0 | 75.0 | 25.0 |
| cot | 100.0 | 50.0 | 81.2 |

**qwen3.5-0.8b — extractive QA**: baseline F1 100.0 over 8 samples

| K | % heads | retrieval F1 (drop) | random F1 (drop) |
|---|---|---|---|
| 2 | 4.2% | 75.0 (25.0) | 88.1 (11.9) |
| 4 | 8.3% | 62.5 (37.5) | 66.7 (33.3) |
| 8 | 16.7% | 46.9 (53.1) | 56.9 (43.1) |

**qwen3.5-0.8b — chain-of-thought**: K=4, 8 samples

| variant | baseline | retrieval masked | random masked |
|---|---|---|---|
| answer_only | 75.0 | 87.5 | 62.5 |
| cot | 100.0 | 75.0 | 31.2 |


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
- `figures/masking_recall.pdf`
- `figures/masking_recall.png`
- `figures/mixer_ablation.pdf`
- `figures/mixer_ablation.png`
- `figures/ring_graph.pdf`
- `figures/ring_graph.png`
- `figures/score_distribution.pdf`
- `figures/score_distribution.png`
- `figures/task_cot_Qwen3-0.6B.pdf`
- `figures/task_cot_Qwen3-0.6B.png`
- `figures/task_cot_Qwen3.5-0.8B.pdf`
- `figures/task_cot_Qwen3.5-0.8B.png`
- `figures/task_qa_Qwen3-0.6B.pdf`
- `figures/task_qa_Qwen3-0.6B.png`
- `figures/task_qa_Qwen3.5-0.8B.pdf`
- `figures/task_qa_Qwen3.5-0.8B.png`
- `overlap.json`
- `qwen3-0.6b/instances_next_step.jsonl`
- `qwen3-0.6b/masking_curve.json`
- `qwen3-0.6b/model_info.json`
- `qwen3-0.6b/scores_next_step.json`
- `qwen3-0.6b/scores_next_step.npz`
- `qwen3-0.6b/scores_next_step_recited.json`
- `qwen3-0.6b/scores_next_step_recited.npz`
- `qwen3-0.6b/scores_same_step.json`
- `qwen3-0.6b/scores_same_step.npz`
- `qwen3-0.6b/scores_same_step_recited.json`
- `qwen3-0.6b/scores_same_step_recited.npz`
- `qwen3-0.6b/summary_next_step.json`
- `qwen3-0.6b/summary_same_step.json`
- `qwen3-0.6b/task_cot.json`
- `qwen3-0.6b/task_qa.json`
- `qwen3.5-0.8b/instances_next_step.jsonl`
- `qwen3.5-0.8b/masking_curve.json`
- `qwen3.5-0.8b/mixer_ablation.json`
- `qwen3.5-0.8b/model_info.json`
- `qwen3.5-0.8b/scores_next_step.json`
- `qwen3.5-0.8b/scores_next_step.npz`
- `qwen3.5-0.8b/scores_next_step_recited.json`
- `qwen3.5-0.8b/scores_next_step_recited.npz`
- `qwen3.5-0.8b/scores_same_step.json`
- `qwen3.5-0.8b/scores_same_step.npz`
- `qwen3.5-0.8b/scores_same_step_recited.json`
- `qwen3.5-0.8b/scores_same_step_recited.npz`
- `qwen3.5-0.8b/summary_next_step.json`
- `qwen3.5-0.8b/summary_same_step.json`
- `qwen3.5-0.8b/task_cot.json`
- `qwen3.5-0.8b/task_qa.json`
