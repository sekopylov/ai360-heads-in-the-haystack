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
   `qwen3_5_moe`, `qwen3_next`, `gemma3`, `gemma4`, `gemma3n` and — added when the
   third family entered the round — `ministral3` in the installed `transformers` 5.18:
   **all of them expose it**, so no capture work is needed for any candidate below.
   (`ministral3` matters most of the nine: it is one of the ten checkpoints actually
   pinned, and the list above it was written before it was chosen.)

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

* **Gemma 4's global layers have no `v_proj` — and that is *not* a crash.**  In
  `modeling_gemma4.py`, `use_alternative_attention = config.attention_k_eq_v and not
  is_sliding`, and `self.v_proj = (... if not self.use_alternative_attention else None)`.
  An earlier version of this document said the global layers therefore "fail the `q/k/v/o`
  signature ... and `build_model_info` **raises**".  That is wrong, and the reason is worth
  keeping: the signature test is `all(hasattr(module, attr) ...)`, and
  `nn.Module.__setattr__` with a `None` value still creates the attribute
  (`object.__setattr__`, nothing in `_modules`).  So `hasattr(module, "v_proj")` is **True**
  and the layer is discovered as ordinary scoreable attention.  The real consequence is
  quieter and worse: with `attention_k_eq_v: true` (Gemma 4 12B / 26B-A4B / 31B) those
  layers are scored as if V were their own projection when it is `k_proj`'s output, and
  nothing raises.  The fix is the same kind of decision as the sliding-window one below --
  recognise the unified-K/V form and mark it, rather than rely on a signature that cannot
  see the difference between "absent" and "`None`".
* **KV sharing is the form that does raise.**  Gemma 4 E2B/E4B (`num_kv_shared_layers`
  20/18) and Gemma 3n (10/15) never assign `k_proj`/`v_proj` on the last N layers at all,
  so `hasattr` is false, the layer is neither attention nor a known mixer, and the coverage
  invariant in `build_model_info` fires.  Their attention maps are also computed against
  another layer's K/V, so those "heads" are not their own and must be excluded or marked.
  Gemma 3 and Gemma 4 12B/26B/31B have no sharing (`0`).
* **Gemma 3's sliding layers would dilute the number.**  They are discovered as scoreable
  (all four projections exist), so the `>0.1` share would be computed over ~6x more heads
  than can ever fire.  Excluding `layer_types == "sliding_attention"` from the score (and
  reporting them like the linear mixers) is the honest fix — and `ModelInfo.layer_types`
  already carries the config's own list.
* **"Structurally 0" is a mid-context statement.**  The depth grid is `iter_depths(10)`,
  which includes 0 and 1, so at depth 1 the needle sits next to the query and even a
  512-token window sees it; at 4K with a 1024-token window depths ~0.75 and above are
  visible too.  The conclusion (exclude the sliding layers) is unchanged, but the phrase
  should not be read as "zero on every sample".

## 3. Compatibility, verified

| family / repo | `model_type` | in `transformers` 5.18 | capture | discovery | verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B/4B/8B/14B/32B | `qwen3` | yes | yes | all layers dense | **drop-in** |
| Qwen3-30B-A3B, 235B-A22B | `qwen3_moe` | yes | yes | dense attention, MoE FFN | **drop-in** |
| Qwen3.5-2B/4B/9B/27B | `qwen3_5` (VL) | yes | yes | 3/4 Gated DeltaNet, `layer_types` in the config | **drop-in** (same path as the 0.8B) |
| Qwen3.5-35B-A3B/122B-A10B | `qwen3_5_moe` | yes | yes | as above | **drop-in** |
| Qwen3-Next-80B-A3B | `qwen3_next` | yes | yes | `full_attention_interval: 4` -> 12 of 48 layers | drop-in, but **151 GiB** of weights: not one A100 |
| Gemma 3 4B/12B/27B | `gemma3` | yes | yes | all layers scoreable, 5 local : 1 global | needs the sliding-window decision; **repos are gated** (401) |
| Gemma 4 12B/26B/31B | `gemma4`, `gemma4_unified` | yes (>= 5.5) | yes | global layers carry `v_proj = None` (unified K/V) | **needs code**, but silently: the layer is discovered as ordinary attention, so nothing raises — see §2 |
| Gemma 4 E2B/E4B, Gemma 3n | `gemma4`, `gemma3n` | yes | yes | `k_proj`/`v_proj` never assigned on the shared layers | needs both changes; weakest candidates (this one *does* raise) |

Provenance caveat: `google/gemma-3-*` and `google/gemma-3n-*` configs answer **401**
(gated licence), so their numbers here come from ungated mirrors
(`unsloth/gemma-3-*`, `RedHatAI/gemma-3-*`), cross-checked where two mirrors exist.  All
Gemma 4 configs are open (Apache-2.0) and were read directly.  Qwen3/Qwen3.5 configs are
open and were read directly.

## 4. Fits on one A100 80 GB at 49K?

Weights are `params x 2 bytes`; KV is computed from the configs above.  The budget is
`weights + KV + workspace <= 80 GiB`, and **the workspace column is not optional**: at
`--prefill-chunk 8192` and 49152 tokens a *materialised* score matrix is
`8192 x 49152 x 4 B = 1.61 GB per Q-head`, i.e. ~26 GiB at 16 Q-heads and ~52 GiB at 32.
That is not hypothetical -- `docs/datasphere-findings.md` section 18 is a 16K float32
prefill that asked for 20.6 GiB in one allocation on a 22 GiB L4.  It is a *fallback*
risk, not the normal path: the run is bf16, the prefill uses `sdpa` (`capture_impl`
`eager` applies only to the last token and the decode steps, where the matrix is
`(heads, 1, seq)`), and bf16 SDPA reaches flash or mem-efficient, neither of which
materialises.  So the rows below are the weights+KV budget, and the 32-Q-head models
(Qwen3-4B/8B/30B-A3B, both Ministral) are the ones where a math-backend fallback would
not fit.  `report_environment` probes the flash backend at job start.

Resident weights are smaller than the checkpoint for the Qwen3.5 models:
`AutoModelForCausalLM` resolves `qwen3_5` to `Qwen3_5ForCausalLM`, so the vision tower
and the `mtp.*` block are skipped (the committed
`ds-results-a100/qwen3.5-0.8b/model_info.json` records `model_class =
Qwen3_5ForCausalLM`).  The Ministral checkpoints go the other way: `mistral3` is not in
the causal-LM mapping, so `load_model` falls through to
`AutoModelForImageTextToText` and the 24-layer pixtral tower **is** resident.  Both
columns are shown where they differ.

| model | scoreable heads | weights GiB | KV @49K GiB | total GiB | verdict |
|---|---|---|---|---|---|
| Qwen3-0.6B | 448 | 1.4 | 5.25 | 6.7 | trivially fits |
| Qwen3-1.7B | 448 | 3.8 | 5.2 | 9.0 | trivially fits |
| Qwen3-4B | 1152 | 7.5 | 6.8 | 14.2 | fits (32 Q-heads: workspace is the risk, not weights) |
| **Qwen3-8B** | **1152** | **15.3** | **6.8** | **22.0** | **fits comfortably** |
| **Qwen3-14B** | **1600** | **27.5** | **7.5** | **35.0** | **fits comfortably** |
| Qwen3-32B | 4096 | 61.0 | 12.0 | 73.0 | borderline: ~7 GiB for activations/workspace, no margin |
| **Qwen3-30B-A3B** (MoE, 3.3B active) | **1536** | **56.9** | **4.5** | **61.4** | **fits** (~19 GiB spare) -- but 32 Q-heads, so a math fallback needs ~52 GiB and would OOM |
| Qwen3.5-0.8B | 48 | 1.63 (1.40 resident) | 0.56 | 2.2 | trivially fits |
| Qwen3.5-2B | 48 | 4.2 | 0.6 | 4.8 | fits (same head count as the 0.8B) |
| Qwen3.5-4B | 128 | 8.68 (7.6 resident) | 1.50 | 10.2 | fits comfortably |
| **Qwen3.5-9B** | **128** | **18.0 (16.6 resident)** | **1.5** | **19.5** | **fits comfortably** |
| Qwen3.5-27B | 384 | 51.7 | 3.0 | 54.7 | fits |
| Qwen3.5-35B-A3B | 160 | 67.0 (64.4 resident) | 0.9 | 67.9 | fits, ~15 GiB spare resident; 16 Q-heads, so even a math fallback needs ~26 GiB and does **not** fit |
| Ministral-3-8B | 1088 | 16.61 | 6.38 | 23.0 | fits (32 Q-heads: workspace is the risk) |
| Ministral-3-14B | 1280 | 25.97 | 7.50 | 33.5 | fits comfortably on weights+KV; 32 Q-heads, so a math fallback needs ~52 GiB and would OOM |
| Gemma 4 12B Unified | 128 (8 global) | 22.3 | 15.8 | 38.0 | fits, after the code change |
| Gemma 4 26B-A4B | 80 (5 global) | 46.9 | 10.3 | 57.3 | fits, after the code change |
| Gemma 3 12B | 128 (8 global) | 22.4 | 18.0 | 40.4 | fits, but gated + sliding decision |
| Gemma 4 31B | 160 (10 global) | 57.2 | 41.2 | 98.4 | **no** |
| Gemma 3 27B | 320 (10 global) | 50.3 | 23.2 | 73.5 | borderline, and gated |
| Qwen3-Next-80B-A3B | 192 (12 full) | 151.5 | 1.1 | 152.6 | **no** (needs >= 2 cards) |
| Qwen3-122B/235B/397B, Coder-480B | — | 234-894 | — | — | **no** |

## 5. Recommendation — the chosen set

**Decided (2026-10-10):** the round adds **Qwen3-4B, Qwen3.5-4B, Qwen3-8B, Qwen3.5-9B,
Qwen3-30B-A3B, Qwen3.5-35B-A3B**, with **Ministral-3-8B** (and 14B) researched and pinned
as the third family.  All are in `configs/models.json` now, pinned with SHA-256 digests
(`scripts/pin_registry.py`), and the project disk is the delivery path
(`configs/datasphere/a100-weights.yaml` -> `${DS_PROJECT_HOME}/ai360-heads-in-the-haystack/models`).

Why this set, in order of what it buys:

1. **A matched dense-vs-hybrid pair at ~4B and again at ~8-9B.**  The project's central
   comparison currently rests on 0.6B vs 0.8B; if the hybrid's non-sparsity is
   architectural it has to survive an 11x scale-up, and if the dense model's sparsity is a
   template artefact that has to hold too.  Qwen3-4B (1152 heads) / Qwen3.5-4B (128 of
   32 layers) and Qwen3-8B (1152) / Qwen3.5-9B (128) give two independent rungs.
2. **A scaling ladder inside one family**: 0.6 -> 4 -> 8B dense, which is the paper's
   "a few percent, and it grows with scale" question asked directly.
3. **A different regime**: Qwen3-30B-A3B is dense *attention* with a sparse FFN, and only
   3.3B parameters are active, so it is cheaper in wall clock than the 14B while having
   1536 scoreable heads.  Qwen3.5-35B-A3B is the same idea on the hybrid side.
4. **A third family**: Ministral-3-8B (Mistral, Apache-2.0, **no sliding window at all** —
   `sliding_window: null` and no `layer_types` in any of the nine 2512 configs, confirmed
   in `Ministral3Model.forward`, so 34 x 32 = 1088 heads are all scoreable).  A result
   that only holds inside one lab's recipe is weak; this is the cheapest way to test that.
   *Caveat*: its Instruct variants ship **FP8** on disk, so the `-BF16` re-releases
   (`mistralai/Ministral-3-{8B,14B}-Instruct-2512-BF16`, same architecture, bf16) are the
   ones pinned; and its native window is 16K stretched by YaRN x16, so 49K is
   extrapolation for it even though the config says 262144.
5. **Deliberately not taken**: Qwen3-32B (73 GiB, no margin), Gemma 3/4 (needs the
   unified-K/V and sliding-window work in §2; Gemma 3 is gated), the Gemma E-variants and
   Gemma 3n (KV-shared heads are not their own), Qwen3-Next-80B and everything at 122B+
   (multi-card), and Ministral 3 3B (redundant with Qwen3-4B for 7 GiB).
   The old **Ministral-8B-Instruct-2410** is worth noting as a *wide-window* hybrid
   (9 global : 27 local, window 32768): unlike Gemma's 1024-token window it can still see
   a mid-context needle at our grid lengths, so it is a scoreable sliding-window model
   rather than a structurally blinded one.

**Cost, from the measured 0.6B baseline** (A100, 542.88 RUB/h).  `detect` is dominated by
the capture (`layers x heads` per decode step), `mask` by prefill+decode work
(parameters).  Rough extrapolations, to be checked against the first hour of a real run:

| model | detect | mask | total | ~RUB |
|---|---|---|---|---|
| Qwen3-4B | ~40 min | ~1.5-2 h | ~2.5 h | 1.4k |
| Qwen3.5-4B | ~40 min | ~1.5-2 h | ~2.5 h | 1.4k |
| Qwen3-8B | ~1 h | ~2-3 h | ~3-4 h | 1.6-2.2k |
| Qwen3.5-9B | ~1 h | ~2-3 h | ~3-4 h | 1.6-2.2k |
| Ministral-3-8B | ~1 h | ~2-3 h | ~3-4 h | 1.6-2.2k |
| Qwen3-30B-A3B | ~1.5 h | ~1.5-2 h | ~3-3.5 h | 1.6-1.9k |
| Qwen3.5-35B-A3B | ~1.5 h | ~2 h | ~3.5 h | 1.9k |
| _all seven_ | | | **~20-24 h** | **~11-13k** |

The knobs that move this: `--random-trials 5 -> 3` cuts ~40% of the `mask` arms,
`--lengths` trims the grid (the t4 scale's 2 lengths are ~1/3 of the A100's 3), and the
stages can be split (`describe,detect` first for every model, then `mask` only for the
ones that earn it -- the detect half is ~30% of the total and answers the sparsity-vs-scale
question on its own).

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
logistics reorder it slightly.  With the project-disk cache in place (the job in
`configs/datasphere/a100-weights.yaml`) the ceiling disappears and the set in §5 is
reachable; before that, the largest drop-in pair the *proven* path carries is
**Qwen3-4B + Qwen3.5-4B** (7.5 and 8.7 GiB), which already takes the dense-vs-hybrid
comparison from 0.6B/0.8B to 4B.

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
