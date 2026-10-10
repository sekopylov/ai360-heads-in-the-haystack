# Results — `ds-results-a100`

## Architecture census

| model | layers | scoreable layers | scoreable heads | head_dim | hybrid |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 28 | 448 | 128 | False |
| Qwen3.5-0.8B | 24 | 6 | 48 | 256 | True |

## Retrieval-head detection

| model | pairing | instances | recited | with copy | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | next_step | 210 | 209/210 | 210/210 | 0.969 | `L21H9` | 0.93 | 126/448 (28.1%) | 41/448 (9.2%) |
| _Qwen3-0.6B (recited only)_ | next_step | 209 | (same) | (same) | 0.969 | `L21H9` | 0.94 | 127/448 (28.3%) | 42/448 (9.4%) |
| Qwen3-0.6B | same_step | 210 | 209/210 | 210/210 | 0.969 | `L6H6` | 0.97 | 76/448 (17.0%) | 24/448 (5.4%) |
| _Qwen3-0.6B (recited only)_ | same_step | 209 | (same) | (same) | 0.969 | `L6H6` | 0.97 | 76/448 (17.0%) | 24/448 (5.4%) |
| Qwen3.5-0.8B | next_step | 270 | 269/270 | 270/270 | 0.942 | `L15H7` | 0.89 | 35/48 (72.9%) | 15/48 (31.2%) |
| _Qwen3.5-0.8B (recited only)_ | next_step | 269 | (same) | (same) | 0.942 | `L15H7` | 0.90 | 35/48 (72.9%) | 15/48 (31.2%) |
| Qwen3.5-0.8B | same_step | 270 | 269/270 | 270/270 | 0.942 | `L7H7` | 0.91 | 23/48 (47.9%) | 10/48 (20.8%) |
| _Qwen3.5-0.8B (recited only)_ | same_step | 269 | (same) | (same) | 0.942 | `L7H7` | 0.91 | 23/48 (47.9%) | 10/48 (20.8%) |

- `qwen3-0.6b`/next_step: criterion (2) searched the **haystack** domain; it moved on 90.3% of 2350656 (layer, head, step) positions relative to the prompt domain
- `qwen3-0.6b`/next_step: the attention sink (sequence position 0) is **outside** the haystack span, so it cannot suppress criterion (2); the paper's template-free prompt puts it inside `x`
- `qwen3-0.6b`/next_step: the raw per-token denominator (`|k|` read literally, repeats counted) puts 124/448 heads above 0.1, against 126/448 under `|unique(k)|`
- `qwen3-0.6b`/next_step: the >0.1 share by argmax domain: `full` 21/448 (4.7%), `haystack` 126/448 (28.1%), `prompt` 23/448 (5.1%)
- `qwen3-0.6b`/next_step: 8/10 of the `haystack` top heads are also top under `full` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3-0.6b`/next_step: 8/10 of the `haystack` top heads are also top under `prompt` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3-0.6b`/next_step: the random arm's pool (heads at or below 0.10) by argmax domain: `full` 427/448, `haystack` 322/448, `prompt` 425/448 -- `mask` draws its control from it, and a pool of a few heads collapses every large-K point into the same intervention
- `qwen3-0.6b`/same_step: criterion (2) searched the **haystack** domain; it moved on 90.1% of 2350656 (layer, head, step) positions relative to the prompt domain
- `qwen3-0.6b`/same_step: the attention sink (sequence position 0) is **outside** the haystack span, so it cannot suppress criterion (2); the paper's template-free prompt puts it inside `x`
- `qwen3-0.6b`/same_step: the raw per-token denominator (`|k|` read literally, repeats counted) puts 74/448 heads above 0.1, against 76/448 under `|unique(k)|`
- `qwen3-0.6b`/same_step: the >0.1 share by argmax domain: `full` 21/448 (4.7%), `haystack` 76/448 (17.0%), `prompt` 22/448 (4.9%)
- `qwen3-0.6b`/same_step: 6/10 of the `haystack` top heads are also top under `full` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3-0.6b`/same_step: 6/10 of the `haystack` top heads are also top under `prompt` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3-0.6b`/same_step: the random arm's pool (heads at or below 0.10) by argmax domain: `full` 427/448, `haystack` 372/448, `prompt` 426/448 -- `mask` draws its control from it, and a pool of a few heads collapses every large-K point into the same intervention
- `qwen3.5-0.8b`/next_step: criterion (2) searched the **haystack** domain; it moved on 38.3% of 327840 (layer, head, step) positions relative to the prompt domain
- `qwen3.5-0.8b`/next_step: the attention sink (sequence position 0) is **outside** the haystack span, so it cannot suppress criterion (2); the paper's template-free prompt puts it inside `x`
- `qwen3.5-0.8b`/next_step: the raw per-token denominator (`|k|` read literally, repeats counted) puts 35/48 heads above 0.1, against 35/48 under `|unique(k)|`
- `qwen3.5-0.8b`/next_step: the >0.1 share by argmax domain: `full` 30/48 (62.5%), `haystack` 35/48 (72.9%), `prompt` 33/48 (68.8%)
- `qwen3.5-0.8b`/next_step: 8/10 of the `haystack` top heads are also top under `full` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3.5-0.8b`/next_step: 8/10 of the `haystack` top heads are also top under `prompt` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3.5-0.8b`/next_step: the random arm's pool (heads at or below 0.10) by argmax domain: `full` 18/48, `haystack` 13/48, `prompt` 15/48 -- `mask` draws its control from it, and a pool of a few heads collapses every large-K point into the same intervention
- `qwen3.5-0.8b`/same_step: criterion (2) searched the **haystack** domain; it moved on 37.4% of 327840 (layer, head, step) positions relative to the prompt domain
- `qwen3.5-0.8b`/same_step: the attention sink (sequence position 0) is **outside** the haystack span, so it cannot suppress criterion (2); the paper's template-free prompt puts it inside `x`
- `qwen3.5-0.8b`/same_step: the raw per-token denominator (`|k|` read literally, repeats counted) puts 23/48 heads above 0.1, against 23/48 under `|unique(k)|`
- `qwen3.5-0.8b`/same_step: the >0.1 share by argmax domain: `full` 14/48 (29.2%), `haystack` 23/48 (47.9%), `prompt` 17/48 (35.4%)
- `qwen3.5-0.8b`/same_step: 5/10 of the `haystack` top heads are also top under `full` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3.5-0.8b`/same_step: 9/10 of the `haystack` top heads are also top under `prompt` (the masking arm ranks by the primary domain, so this is how much a domain change would change what it masks)
- `qwen3.5-0.8b`/same_step: the random arm's pool (heads at or below 0.10) by argmax domain: `full` 34/48, `haystack` 25/48, `prompt` 31/48 -- `mask` draws its control from it, and a pool of a few heads collapses every large-K point into the same intervention

## Cross-model agreement

> **caveat**: mode='sorted' correlates sorted score vectors, not head positions; a high value does not mean the models use the same heads
- correlation: mode `sorted`, models: Qwen3.5-0.8B, Qwen3-0.6B
- head overlap: not comparable (different layer/head layouts); Jaccard suppressed

## Strict-aligned matching

| model | pairing | strict-aligned top heads (mean score) |
|---|---|---|
| Qwen3-0.6B | next_step | L21H9 (0.93), L16H14 (0.93), L16H15 (0.92), L21H8 (0.91), L3H10 (0.90) |
| Qwen3-0.6B | same_step | L6H6 (0.97), L6H7 (0.97), L11H2 (0.95), L3H11 (0.92), L11H3 (0.90) |
| Qwen3.5-0.8B | next_step | L15H7 (0.89), L23H5 (0.87), L11H1 (0.84), L23H0 (0.82), L23H1 (0.80) |
| Qwen3.5-0.8B | same_step | L7H7 (0.91), L23H6 (0.90), L23H2 (0.89), L3H7 (0.88), L7H6 (0.84) |

## Pairing sensitivity

| model | top-10 overlap between pairings | next_step top heads | same_step top heads |
|---|---|---|---|
| Qwen3-0.6B | 0/10 | L21H9, L16H14, L16H15, L21H8, L3H10 | L6H6, L6H7, L11H2, L3H11, L11H3 |
| Qwen3.5-0.8B | 0/10 | L15H7, L23H5, L11H1, L23H0, L23H1 | L7H7, L23H6, L23H2, L3H7, L7H6 |

## Masking curve

| model | K | % heads | retrieval f1 | retrieval exact | random f1 | random exact |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 4 | 0.9% | 89.3 ±10.3 | 26.7 | 95.8 | 67.6 |
| Qwen3-0.6B | 9 | 2.0% | 81.4 ±12.4 | 0.0 | 72.8 | 52.0 |
| Qwen3-0.6B | 18 | 4.0% | 86.0 ±8.9 | 0.0 | 66.8 | 35.6 |
| Qwen3-0.6B | 36 | 8.0% | 53.7 ±16.0 | 0.0 | 38.4 | 16.4 |
| Qwen3-0.6B | 76 | 17.0% | 20.2 ±4.3 | 0.0 | 0.9 | 0.0 |
| Qwen3-0.6B | 148 | 33.0% | 0.0 | 0.0 | 1.8 | 0.0 |
| | baseline | | 97.4 | 88.9 | | |
| | _qwen3-0.6b: 45 samples per point; ± is the spread across them_ | | | | | |
| Qwen3.5-0.8B | 1 | 2.1% | 70.3 ±22.5 | 33.3 | 65.9 | 36.0 |
| Qwen3.5-0.8B | 2 | 4.2% | 58.2 ±16.4 | 35.6 | 54.6 | 22.7 |
| Qwen3.5-0.8B | 4 | 8.3% | 50.5 ±21.3 | 24.4 | 59.2 | 25.8 |
| Qwen3.5-0.8B | 8 | 16.7% | 44.6 ±19.9 | 0.0 | 48.0 | 24.0 |
| Qwen3.5-0.8B | 16 →13 ⚠ | 27.1% | 33.8 ±14.0 | 0.0 | 21.0 | 2.2 |
| | _⚠ = the retrieval arm hit the control-pool size, so the random arm drew the whole pool (unmatched)_ | | | | | |
| | baseline | | 76.1 | 55.6 | | |
| | _qwen3.5-0.8b: 45 samples per point; ± is the spread across them_ | | | | | |

## Token-mixer ablation

**qwen3.5-0.8b** — 6 full-attention layers, 18 linear layers, baseline f1=76.1

| K | full-attention masked | linear masked |
|---|---|---|
| 1 | 56.6 ±2.6 | 74.7 ±10.2 |
| 2 | 20.5 ±10.5 | 60.1 ±7.0 |
| 4 | 10.7 ±1.6 | 17.8 ±24.2 |

_± is the spread across the sampled layer subsets (n_trials); `distinct_subsets` says how many were distinct._


## Cross-model correlation

Retrieval-score correlation (mode `sorted`):

| | Qwen3.5-0.8B | Qwen3-0.6B |
|---|---|---|
| Qwen3.5-0.8B | 1.00 | 0.98 |
| Qwen3-0.6B | 0.98 | 1.00 |

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
- `overlap.json`
- `qwen3-0.6b/case_study.json`
- `qwen3-0.6b/figures/retrieval_attention_dist.pdf`
- `qwen3-0.6b/figures/retrieval_attention_dist.png`
- `qwen3-0.6b/instances_next_step.jsonl`
- `qwen3-0.6b/masking_curve.json`
- `qwen3-0.6b/model_info.json`
- `qwen3-0.6b/scores_next_step.json`
- `qwen3-0.6b/scores_next_step.npz`
- `qwen3-0.6b/scores_next_step_full.json`
- `qwen3-0.6b/scores_next_step_full.npz`
- `qwen3-0.6b/scores_next_step_prompt.json`
- `qwen3-0.6b/scores_next_step_prompt.npz`
- `qwen3-0.6b/scores_next_step_raw.json`
- `qwen3-0.6b/scores_next_step_raw.npz`
- `qwen3-0.6b/scores_next_step_recited.json`
- `qwen3-0.6b/scores_next_step_recited.npz`
- `qwen3-0.6b/scores_same_step.json`
- `qwen3-0.6b/scores_same_step.npz`
- `qwen3-0.6b/scores_same_step_full.json`
- `qwen3-0.6b/scores_same_step_full.npz`
- `qwen3-0.6b/scores_same_step_prompt.json`
- `qwen3-0.6b/scores_same_step_prompt.npz`
- `qwen3-0.6b/scores_same_step_raw.json`
- `qwen3-0.6b/scores_same_step_raw.npz`
- `qwen3-0.6b/scores_same_step_recited.json`
- `qwen3-0.6b/scores_same_step_recited.npz`
- `qwen3-0.6b/summary_next_step.json`
- `qwen3-0.6b/summary_same_step.json`
- `qwen3.5-0.8b/case_study.json`
- `qwen3.5-0.8b/figures/retrieval_attention_dist.pdf`
- `qwen3.5-0.8b/figures/retrieval_attention_dist.png`
- `qwen3.5-0.8b/instances_next_step.jsonl`
- `qwen3.5-0.8b/masking_curve.json`
- `qwen3.5-0.8b/mixer_ablation.json`
- `qwen3.5-0.8b/model_info.json`
- `qwen3.5-0.8b/scores_next_step.json`
- `qwen3.5-0.8b/scores_next_step.npz`
- `qwen3.5-0.8b/scores_next_step_full.json`
- `qwen3.5-0.8b/scores_next_step_full.npz`
- `qwen3.5-0.8b/scores_next_step_prompt.json`
- `qwen3.5-0.8b/scores_next_step_prompt.npz`
- `qwen3.5-0.8b/scores_next_step_raw.json`
- `qwen3.5-0.8b/scores_next_step_raw.npz`
- `qwen3.5-0.8b/scores_next_step_recited.json`
- `qwen3.5-0.8b/scores_next_step_recited.npz`
- `qwen3.5-0.8b/scores_same_step.json`
- `qwen3.5-0.8b/scores_same_step.npz`
- `qwen3.5-0.8b/scores_same_step_full.json`
- `qwen3.5-0.8b/scores_same_step_full.npz`
- `qwen3.5-0.8b/scores_same_step_prompt.json`
- `qwen3.5-0.8b/scores_same_step_prompt.npz`
- `qwen3.5-0.8b/scores_same_step_raw.json`
- `qwen3.5-0.8b/scores_same_step_raw.npz`
- `qwen3.5-0.8b/scores_same_step_recited.json`
- `qwen3.5-0.8b/scores_same_step_recited.npz`
- `qwen3.5-0.8b/summary_next_step.json`
- `qwen3.5-0.8b/summary_same_step.json`
- `run_state.json`
