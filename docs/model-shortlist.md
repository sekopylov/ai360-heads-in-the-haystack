# Model shortlist: what to add after Qwen3-0.6B / Qwen3.5-0.8B

Research for the next round of the study, on **one A100 80 GB** and the existing
1K-49K grid.  Architecture numbers are from each model's `config.json` (fetched from
the Hub); the *compatibility* column is verified against the code in this repo and the
pinned `transformers` 5.18, not against prose.  Where a fact comes from a model card or
a mirror rather than a config, it says so.

## 1. What a candidate has to provide

The method needs a real attention map, so a model is only scoreable if all three hold:

1. **A softmax attention module per layer** — the discovery in `retrieval_heads/models.py`
   is signature-based (`q_proj`/`k_proj`/`v_proj`/`o_proj` + `layer_idx`), so a renamed
   architecture is discovered automatically, and a *linear/recurrent* mixer is excluded by
   class name (`DeltaNet`, `LinearAttention`, `Mamba`, `Recurrent`, `SSM`, `RWKV`).
2. **Every layer recognised as one or the other.**  `build_model_info` refuses a model
   whose layer is neither ("layers [i] have neither a scoreable attention module nor a
   recognised token mixer"), which is deliberate — a renamed mixer would otherwise look
   dense and change `n_all_heads` silently.
3. **The capture can patch it** — `eager_attention_forward` must be reachable from the
   attention class's module globals.  Verified for `qwen3`, `qwen3_moe`, `qwen3_5`,
   `qwen3_5_moe`, `qwen3_next`, `gemma3`, `gemma4`, `gemma3n` in the installed
   `transformers` 5.18: **all of them expose it**, so no capture work is needed for any
   candidate below.

At 49K the budget is `weights + KV + workspace <= 80 GiB`, with the KV cache sized from
the layers that keep one (a linear layer's recurrent state does not grow with context):

```
KV_GiB = 2 (K,V) * sum_over_layers(kv_heads * head_dim) * seq * 2 bytes / 2**30
```

## 2. Three architecture classes, and what each does to the score

| class | example | scoreable heads at 49K | why |
|---|---|---|---|
| **dense** | Qwen3-8B/14B/32B, Qwen3-30B-A3B (MoE FFN, dense attention) | all `layers x Q-heads` | every layer attends over the whole context |
| **linear hybrid** | Qwen3.5-*, Qwen3-Next-80B | `full_attention layers x Q-heads` | the other 3/4 of layers are Gated DeltaNet: a recurrent state, no map |
| **sliding-window hybrid** | Gemma 3, Gemma 4 | only the **global** layers | the local layers *do* have maps, but a 512/1024-token window: at 4K-49K a mid-context needle is invisible to them, so their score is structurally 0 |

The third class is the interesting one and the one the code does **not** handle yet.  Two
concrete obstacles, both read out of the installed implementation:

* **Gemma 4's global layers have no `v_proj`.**  In `modeling_gemma4.py`,
  `use_alternative_attention = config.attention_k_eq_v and not is_sliding`, and
  `self.v_proj = (... if not self.use_alternative_attention else None)`.  With
  `attention_k_eq_v: true` (Gemma 4 12B / 26B-A4B / 31B) the global layers therefore fail
  the `q/k/v/o` signature, are not linear either, and `build_model_info` **raises**.  A
  small change is needed to accept the unified-K/V form (`k_proj` supplies both K and V).
* **KV sharing.**  Gemma 4 E2B/E4B (`num_kv_shared_layers` 20/18) and Gemma 3n (10/15)
  drop `k_proj`/`v_proj` on the last N layers entirely: their attention maps are computed
  against another layer's K/V, so those "heads" are not their own and must be excluded or
  marked.  Gemma 3 and Gemma 4 12B/26B/31B have no sharing (`0`).
* **Gemma 3's sliding layers would dilute the number.**  They are discovered as scoreable
  (all four projections exist), so the `>0.1` share would be computed over ~6x more heads
  than can ever fire.  Excluding `layer_types == "sliding_attention"` from the score (and
  reporting them like the linear mixers) is the honest fix — and `ModelInfo.layer_types`
  already carries the config's own list.

## 3. Compatibility, verified

| family / repo | `model_type` | in `transformers` 5.18 | capture | discovery | verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B/4B/8B/14B/32B | `qwen3` | yes | yes | all layers dense | **drop-in** |
| Qwen3-30B-A3B, 235B-A22B | `qwen3_moe` | yes | yes | dense attention, MoE FFN | **drop-in** |
| Qwen3.5-2B/4B/9B/27B | `qwen3_5` (VL) | yes | yes | 3/4 Gated DeltaNet, `layer_types` in the config | **drop-in** (same path as the 0.8B) |
| Qwen3.5-35B-A3B/122B-A10B | `qwen3_5_moe` | yes | yes | as above | **drop-in** |
| Qwen3-Next-80B-A3B | `qwen3_next` | yes | yes | `full_attention_interval: 4` -> 12 of 48 layers | drop-in, but **151 GiB** of weights: not one A100 |
| Gemma 3 4B/12B/27B | `gemma3` | yes | yes | all layers scoreable, 5 local : 1 global | needs the sliding-window decision; **repos are gated** (401) |
| Gemma 4 12B/26B/31B | `gemma4`, `gemma4_unified` | yes (>= 5.5) | yes | global layers have `v_proj = None` | **needs code** (unified K/V) |
| Gemma 4 E2B/E4B, Gemma 3n | `gemma4`, `gemma3n` | yes | yes | + KV sharing on the last N layers | needs both changes; weakest candidates |

Provenance caveat: `google/gemma-3-*` and `google/gemma-3n-*` configs answer **401**
(gated licence), so their numbers here come from ungated mirrors
(`unsloth/gemma-3-*`, `RedHatAI/gemma-3-*`), cross-checked where two mirrors exist.  All
Gemma 4 configs are open (Apache-2.0) and were read directly.  Qwen3/Qwen3.5 configs are
open and were read directly.

## 4. Fits on one A100 80 GB at 49K?

Weights are `params x 2 bytes`; KV is computed from the configs above.

| model | scoreable heads | weights GiB | KV @49K GiB | total GiB | verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B | 448 | 3.8 | 5.2 | 9.0 | trivially fits |
| Qwen3-4B | 1152 | 7.5 | 6.8 | 14.2 | fits |
| **Qwen3-8B** | **1152** | **15.3** | **6.8** | **22.0** | **fits comfortably** |
| **Qwen3-14B** | **1600** | **27.5** | **7.5** | **35.0** | **fits comfortably** |
| Qwen3-32B | 4096 | 61.0 | 12.0 | 73.0 | borderline: ~7 GiB for activations/workspace, no margin |
| **Qwen3-30B-A3B** (MoE, 3.3B active) | **1536** | **56.9** | **4.5** | **61.4** | **fits** (~19 GiB spare) |
| Qwen3.5-2B | 48 | 4.2 | 0.6 | 4.8 | fits (same head count as the 0.8B) |
| **Qwen3.5-9B** | **128** | **18.0** | **1.5** | **19.5** | **fits comfortably** |
| Qwen3.5-27B | 384 | 51.7 | 3.0 | 54.7 | fits |
| Qwen3.5-35B-A3B | 160 | 67.0 | 0.9 | 67.9 | fits, ~12 GiB spare (load the text-only class to drop the vision tower) |
| Gemma 4 12B Unified | 128 (8 global) | 22.3 | 15.8 | 38.0 | fits, after the code change |
| Gemma 4 26B-A4B | 80 (5 global) | 46.9 | 10.3 | 57.3 | fits, after the code change |
| Gemma 3 12B | 128 (8 global) | 22.4 | 18.0 | 40.4 | fits, but gated + sliding decision |
| Gemma 4 31B | 160 (10 global) | 57.2 | 41.2 | 98.4 | **no** |
| Gemma 3 27B | 320 (10 global) | 50.3 | 23.2 | 73.5 | borderline, and gated |
| Qwen3-Next-80B-A3B | 192 (12 full) | 151.5 | 1.1 | 152.6 | **no** (needs >= 2 cards) |
| Qwen3-122B/235B/397B, Coder-480B | — | 234-894 | — | — | **no** |

## 5. Recommendation

**Tier 1 — the matched pair at ~8-9B (drop-in, no code change).**
Add **Qwen3-8B** (dense, 1152 heads) and **Qwen3.5-9B** (hybrid, 128 scoreable heads of
32 layers).  This is the highest-value addition per rouble: the project's central
comparison (a dense model vs a linear hybrid of the same lab and vintage) currently rests
on **0.6B vs 0.8B**, and both new ones fit with room to spare.  If the hybrid's
non-sparsity is architectural, it must survive an 11x scale-up; if the dense model's
sparsity is a template artefact, that must hold too.

**Tier 2 — the scaling ladder and a new regime (drop-in).**
**Qwen3-4B** and **Qwen3-14B** extend the dense ladder to four points in one family
(0.6 -> 4 -> 8 -> 14B), which is the paper's "a few percent, and it grows with scale"
question asked directly.  **Qwen3-30B-A3B** is a genuinely new axis — dense attention with
a sparse FFN — and because only 3.3B parameters are active it is *cheaper in wall clock
than the 14B* while having 1536 scoreable heads.  **Qwen3.5-27B** (384 heads, 55 GiB)
extends the hybrid ladder if Tier 1 shows the effect is real.

**Tier 3 — needs code (defer until Tier 1/2 are reported).**
**Gemma 4 12B** is the most interesting *architecture* on the list (open licence, a third
attention class: 5 local : 1 global with different `head_dim` for the two kinds), but the
global layers' unified K/V means `build_model_info` currently refuses the model.  The work
is bounded: accept the K==V signature, exclude `sliding_attention` layers from the score
(they cannot see a mid-context needle at 49K), and report them like the linear mixers.
Gemma 3 is the same idea behind a gated licence — and `scripts/download_models.sh` has no
`HF_TOKEN` support, so a gated repo is not currently fetchable at all.

**Skip.**  Qwen3-32B (73 GiB leaves no margin; the fp32 SDPA fallback would OOM),
Qwen3.5-35B-A3B (68 GiB, and its 3B-active MoE makes it the least informative per GiB),
Gemma 4 31B / Gemma 3 27B (over budget), the E-variants and Gemma 3n (KV-shared heads are
not their own — a confound, not a measurement), and everything at 122B+ (multi-card).

## 6. Cost, from the measured 0.6B baseline

Measured on the A100: `detect` 25 min (210 dense instances, 1K-49K, 96-token budget) and
`mask` 61 min (45 samples/point, 6 K points, 5 random trials).  `detect` is dominated by
the capture (`layers x heads` per decode step), `mask` by prefill+decode work
(parameters).  Rough extrapolations, to be checked against the first hour of a real run:

| model | detect | mask | total | ~RUB |
|---|---|---|---|---|
| Qwen3-8B | ~1 h | ~2-3 h | ~3-4 h | 1.6-2.2k |
| Qwen3.5-9B | ~1 h | ~2-3 h | ~3-4 h | 1.6-2.2k |
| Qwen3-14B | ~1.5 h | ~3-4 h | ~4.5-5.5 h | 2.4-3.0k |
| Qwen3-30B-A3B | ~1.5 h | ~1.5-2 h | ~3-3.5 h | 1.6-1.9k |

The knobs that move this: `--random-trials 5 -> 3` cuts ~40% of the `mask` arms, and
`--lengths` trims the grid (the t4 scale's 2 lengths are ~1/3 of the A100's 3).

## 7. What adding one model involves — and the 10 GiB ceiling

1. A `configs/models.json` entry: `repo`, `path`, `dtype`, `architectures`, the file list
   and **SHA-256 pins** (the registry is what `--verify-hashes` checks, and it is now part
   of the `--resume` fingerprint).
2. `scripts/download_models.sh` for the weights (locally; there is no `HF_TOKEN` support,
   so a gated repo — Gemma 3/3n — is not fetchable at all today).
3. **Delivery to the job, which is the binding constraint.**  The tested path is the
   `inputs` variable (`models: {var: WEIGHTS}`), and the CLI caps `inputs` **plus** the
   `local-paths` zips at **10 GiB total, 5 GiB per file**
   (`UPLOAD_FILES_MAX_TOTAL_SIZE_BYTES`).  The current 3.2 GB of weights sit far below it;
   the moment a model is bigger than that, the path breaks:

   | model | weights | `inputs` path (<= 10 GiB) |
   |---|---|---|
   | Qwen3-1.7B / Qwen3.5-2B | 4.1 / 4.5 GiB | fits |
   | **Qwen3-4B / Qwen3.5-4B** | **7.5 / 8.7 GiB** | **fits — the largest pair the tested path carries** |
   | Qwen3-8B / Qwen3.5-9B | 15.3 / 18.0 GiB | **does not fit** |
   | Qwen3-14B, Qwen3-30B-A3B | 27.5 / 56.9 GiB | does not fit |

   So the 8-9B pair (and everything above) needs one of the two untested routes:
   **`--download-weights`** (fetch inside the job; needs egress from the job VM, re-downloads
   per job, never exercised here) or the **project disk** (`attach-project-disk` +
   `${DS_PROJECT_HOME}/models`, uploaded once through JupyterLab; the disk is shared, so
   its quota has to be checked before pulling 30-110 GiB into it).
4. For a *new architecture class* only (Gemma): the two changes in §2, each with a test,
   plus a `describe` artifact to confirm the scoreable/linear/windowed split on the real
   checkpoint.

**Consequence for the plan.**  The recommendation in §5 is about *science per rouble*; the
logistics reorder it slightly.  If the next launch must use the already-proven delivery
path, the pair to add is **Qwen3-4B + Qwen3.5-4B** (7.5 and 8.7 GiB — both fit, both
drop-in, and 1152 vs 128 scoreable heads at 4B already dwarfs the 0.6B/0.8B pair).  If the
project disk (or `--download-weights`) is sorted out first, go straight to the 8-9B pair,
which is the better experiment for the same number of GPU hours.

## 8. Not verified here

* No candidate was loaded end-to-end: the per-architecture probe with tiny random weights
  got as far as the configs and the class-level checks (support, `eager_attention_forward`,
  `v_proj = None`, KV sharing) but not to a `describe` on a real checkpoint.  That is a
  `describe`-only GPU (or big-RAM) step, and it is the first thing to run per model.
* The cost table is an extrapolation from the 0.6B baseline, not a measurement.
* Gemma 3/3n numbers come from ungated mirrors because the originals are gated.
* Whether `output_attentions`/the patch yields maps for Qwen3.5's *text* stack inside the
  VL wrapper is already proven by the existing 0.8B run, so it is assumed for the larger
  sizes; nothing else about the larger sizes was executed.
