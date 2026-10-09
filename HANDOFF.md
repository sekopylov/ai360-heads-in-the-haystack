# HANDOFF — state of the retrieval-heads reproduction

**Working document, deliberately not committed.** It is a snapshot for whoever
picks this up next, not part of the deliverable. Delete it when it goes stale.

Read in this order: `README.md` (what this is, how to run it) →
`docs/datasphere-findings.md` (19 recorded divergences from the infra doc, each
tied to a job id) → this file (where things stand and what is left).

---

## 1. State at a glance

**Done and verified end to end.**

* `retrieval_heads/` — the paper's method, architecture-aware. 12 modules.
* 222 tests: 205 fast (`pytest -m "not integration"`, ~13 s), 17 integration against
  the real checkpoints. All green.
* **The committed artifacts now match the code.** The GPU run was refreshed in two
  jobs on an NVIDIA L4, 75 instances per model:
  `bt1ethsb4jlpds7m6i32` (`t4-cached`, ~31 min: describe/detect/mask for both
  models, then failed on `qa` because of a bug of mine) and
  `bt18bmvbuggt68cnel9p` (`t4-resume`, ~17 min: qa/cot/compare/figures reusing the
  first job's artifacts through `local-paths`).  `ds-results/` holds the merged
  tree (schema 5) and `docs/results-gpu.md` is generated from it.  The `results/`
  CPU tree is **not** refreshed and stays historical.
* DataSphere path works: cached venv on the project disk, ~40 s job startup
  instead of ~9 min. Job `bt107kjm3es8vung7130` proves all seven stages run on GPU.
* **Each model is now loaded once per run.** The driver executes a model's stages
  back to back in one process (`stage_plan`, model-major) and `cli._load` keeps
  exactly one resident model, evicting the previous one; a full run used to load
  each checkpoint once per stage (ten loads, ~50 s each on the L4, i.e. ~8 of the
  30 minutes).  `run_state.json` in the output prefix records every stage
  (`running` before, `ok`/`failed` after), and the service still collects the
  outputs of a failed job -- that is what makes the resume path possible.
* **The ablations now reuse every condition `detect` recorded**, not just the
  system prompt: `resolve_detection_settings` takes `chat_template`,
  `enable_thinking`, `argmax_domain`, `threshold` and the corpus *path* from
  `scores.meta` (and warns when the command line explicitly disagrees).  A run of
  `detect --corpus essays.txt` followed by `mask` without `--corpus` used to select
  heads on natural text and measure the effect on the synthetic filler; the corpus
  is now reused, and the ablation artifact records all of those fields plus
  `detection_pairing`, `max_new_tokens`, `prefill_chunk` and `capture_method`.
  Adding this broke `mask` twice (it has no `--argmax-domain`/`--capture-method`),
  so a new test parses each subcommand's argv and asserts that every `args.<flag>`
  its `cmd_*` reads actually exists.
* **Per-sample ablation detail is saved**: `masking_curve.json` now carries
  `per_sample[k]` with the retrieval arm's `NiahMetrics.as_dict()` (per-sample f1,
  exact-match, recall, prefix-recall and the *generated texts*) and the same for
  each random trial, so a specific failure can be inspected from the artifact
  instead of only its mean.
* **Review round (the second report), verified against the code:**
  * `case_study.py` passed `--argmax-domain` to `find_copy_step` (the figure) but not
    to `decode_with_attention` (the capture the credits come from), so with `full`
    the artifact's `top_heads` described the prompt domain while the JSON claimed
    `full`.  Fixed, and the script finally has a test (it had none).
  * `mask` wrote `capture_method: "patch"` unconditionally (`getattr(args, ...)` --
    the flag does not exist there); it now takes what `detect` recorded.
  * README's `denominator_inflation` was stale (1.16; the schema-5 artifacts say
    1.0405, so the equivalent raw threshold is ~0.096, not 0.086).
  * `credits_from_trace` built a `zeros()` on every step via `setdefault`, and its
    `considered` accumulator silently restarted if a layer's head count changed
    (dropping the tally); now a plain lookup and an explicit `ValueError`.
  * `cmd_mask` ran the unmasked baseline twice (the curve and then the mixer
    ablation); `token_mixer_ablation(baseline=...)` reuses it -- on the hybrid that
    pass was the stage's most expensive step.
  * `_load` cleared the cache *after* loading, so both models were resident during
    the load despite the comment; it now evicts first.
  * Provenance gaps closed: `task_qa`/`task_cot` record `threshold`, `pairing` and
    `argmax_domain`; `scores_*_recited.json` carries `config`/`argmax_domain`;
    `code_sha256` hashes `scripts/` too (it defines `SCALES`, i.e. the grid).
  * Ablations now record truncation (`retrieval_truncated`, `random_truncated_mean`,
    `baseline_truncated`), so a drop can be told apart from an exhausted budget.
  * Guards: `plot_heat_map({})` raises a message instead of `max()` on empty;
    `case_study --needle-index` is bounds-checked; the duplicate `scoring` import is
    merged.
  * **Methodology, now stated in the README instead of implied:** the dense masking
    effect is clear only from ~8% of heads (at K=9 one random trial scores 10/10
    exact against a 9/10 baseline; at K=18 the random mean is one collapsed trial);
    on the hybrid the K=16 point caps to the whole 15-head pool, which is a
    deterministic intervention that drives recall to 0.0 with degenerate text while
    the top-15 arm stays fluent -- so there the lowest-scoring heads are collectively
    more critical.  Also added: the 0.759 sink rate and what it does to criterion
    (2); that Sec. 4.3's intrinsic experiment needs a base/derivative pair this
    registry does not have; and that the downstream section is a pipeline check
    (8 items = 12.5 points per item), not a measurement.
  * Not done: splitting `test_regressions.py` (1600+ lines) by topic, and extending
    the ablation to 2-3 eval needles instead of one (that is a GPU run; the artifact
    does record that all ten "samples" are one (q,k) at 2 lengths x 5 depths).
* **Review round (the first report), verified against the code:**
  * `--thinking` did nothing on Qwen3.5.  It passed `None`, which *omits* the
    `enable_thinking` kwarg, and that template tests `is defined and is true`, so the
    flag rendered the empty ` thinking` block while the artifact recorded `null`
    ("left on").  Now `True`, which means on for both shipped templates; pinned by a
    test that renders with the real Qwen3.5 and Qwen3-0.6B tokenizers.
  * `scripts/case_study.py` called `.numpy()` on a CUDA tensor (crash on the GPU box)
    and hard-coded the prompt argmax domain.  Now `.cpu().numpy()`, plus
    `--argmax-domain` recorded in its JSON.
  * `build_model_info` checked coverage one way only: a module with
    `layer_idx >= num_hidden_layers` (MTP block, wrong config) reached
    `empty_matrix`/`aggregate_scores` and died with an IndexError.  Now rejected with
    a message.
  * `DetectionConfig.plan()` cached on identity, and `as_dict()` (called by every
    `summary()`) triggered it -- so editing `lengths`/`limit` afterwards was silently
    ignored.  The cache is keyed on the grid fields now.
  * Dead/loose ends: `CorrelationMatrix.as_dict` carries `caveat` (the caller patched
    it in by hand); the mixer markdown table prints ±std (the PDF already had error
    bars); layer subsets are drawn *without replacement* and `distinct_subsets` is
    recorded; `n_instances_with_copy` is now read (a column in the results doc);
    `run_detection`'s always-False `write_instances=stream is None` is gone;
    `pie_data`/`score_histogram`/`same_family_hint` (unused outside tests) removed;
    job artifacts record `code_sha256`, since a job has no `.git` and `git_rev` was
    always None.
  * Claims that did **not** survive checking: `plot_score_pie` already labelled the
    threshold as "the run's own" (`thr_label`, not the loop's last `thr`); the
    "15 of 22 hybrid failures at the longest contexts" sentence was stale (all 75
    instances count as recited under LCS recall) and is gone; the README's
    `activation_freq` bullet contradicted itself and now states the paper's
    definition (`P(score > 0)`, which is what it computes).
  * Not done: the float32-vs-bfloat16 rank comparison (needs a GPU run), and
    defaulting the hybrid's figure to recall (both `masking_heads.pdf` and
    `masking_recall.pdf` ship, so the reader has both).
* `random_retrieval_overlap` is documented for what it measures (overlap with the
  masked retrieval arm) and the honest control audit
  `random_above_threshold` was added beside it.
* **The credit loop is vectorised** (`match_masks`): it used to call
  `int(argmax[head])` once per head per step, i.e. a device->host synchronisation
  for every head of every step (~448 x 48 x 75 x 4 transfers on the dense control);
  now there is one `.cpu()` per layer-step plus one `nonzero()` for the heads that
  matched.  Verified identical to the old per-head logic on a real trace (credits,
  sink counts and considered counts, both pairings) and ~2.5x faster on a
  dense-shaped CPU microbenchmark (the CUDA gain is larger).  A test pins the new
  edge case: `argmax_domain="full"` can point past the prompt, and the vectorised
  index is clamped before it reads `prompt_ids`.
* **A regression I introduced and then fixed:** the ragged-artifact guard in
  `plot_masking_curve` was inserted *before* the `baseline`/`*_yerr` assignments, so
  the default `metric="f1"` path raised `UnboundLocalError` and the `figures` stage
  died after 5 of 9 PDFs (it catches only `ValueError`).  Fixed, and now covered by
  a default-metric tripwire plus an end-to-end `cmd_figures` test on a synthetic
  tree -- the missing test is exactly why it slipped through.
* **`--system-prompt` is now on every command** (it had landed only on `detect`, so
  `mask` read a non-existent attribute and always got `None`).  Ablations prefer the
  value `detect` recorded in `scores.meta['config']` and warn when `--system-prompt`
  disagrees, so a causal run cannot silently measure a different prompt; `qa`/`cot`
  thread it into `_chat` and record it in their artifacts.
* **`figures` now also writes `masking_recall.pdf`** -- the F1/EM panel is the
  confounded series, so the recall one ships beside it instead of only being
  available via `metric="recall"`.
* Other fixes from the same pass: `cmd_figures`' guard catches any exception (a
  malformed artifact's `KeyError` still killed the stage); `reproduce_laptop.sh`
  passes `--out` to `describe` so the census artifact exists; an empty `--corpus`
  file is an error rather than a silent fallback to synthetic filler; `_norm_word`
  punctuation-only tokens no longer match each other; `resolve_k` rejects a mixed
  negative list instead of dropping the negatives; the depth cut snaps to the
  nearest space (depth 0.0 no longer lands after the first word); `_versions` is
  cached (importing matplotlib per artifact write); `AttentionRecorder.method`
  defaults to `patch` as the README says; `qa`/`cot` artifacts record their masked
  heads and random picks; `SCALES['paper']['cot']` pins `--max-new-tokens 256`;
  CI pins torch to the lock's version.  A new test spies on
  `config._attn_implementation` during the prefill body and the capture steps, so
  "the kernel switch is a no-op" can no longer pass by comparing positions alone.
* **The masking curve now records the LCS needle recall** (`retrieval_recall`,
  `random_recall_mean`, `baseline_recall`, `metric="recall"`).  F1/EM compare
  against the whole needle while the question asks for a sub-span, so a correct
  short answer is penalised by the metric; recall is the series without that
  confound.
* **The attention argmax now runs over the prompt by default** (`--argmax-domain`,
  recorded in every artifact).  Criterion (2) is "the *input* token that receives the
  most attention", so letting already-generated positions win made a head's credit
  depend on how much the model had generated.  Measured sensitivity on a 3-instance
  smoke run of Qwen3-0.6B: the top-10 is identical (10/10) but the number of heads
  above 0.1 differs (18 prompt vs 17 full), i.e. the domain moves the threshold
  membership even where the ranking is stable.
* The `same_step` sidecar no longer compares same_step with itself: its
  `pairing_comparison` was reading `self.secondary` unconditionally, so it claimed
  `overlap == top_k` / `jaccard == 1.0` while the real figure is 0/10.
* The per-head numerator `|g_h ∩ k|` is now in the JSONL as `copied_tokens`
  (sparse: only heads with credits), so the token-level claims of the paper's
  Fig. 3 and the raw-denominator score variant are computable from the artifact.
* `needle_stats` carries all four keys again (the tokenization ceiling was written
  under a different name than it was read), and the summary records both head bases
  (`n_scoreable_heads` and `n_all_heads`).
* **`needle_recall` no longer requires the needle from its first word.**  It is now
  a word-level LCS ratio (the old prefix measure is kept as `needle_prefix_recall`);
  a correct sub-span answer used to score 0.0 and be dropped from the recited-only
  matrices, which is exactly the filter those matrices rest on.  `credits_aligned`
  got the same treatment (true token-level LCS, not a prefix-anchored walk).
* **One definition of "a needle token".**  The needle is now inserted with a space
  after it (`"{needle} \n"`), so the tokenizer no longer fuses its last character
  with the newline; the prompt span therefore ends on the needle's own token and
  `span_ids == needle_text_ids`.  Scoring, the denominator, F1's gold and
  `credits_aligned` all use `set(needle_text_ids)`; the earlier *union* of span and
  text ids (an undocumented deviation from the paper) is gone, and each sample
  records `max_attainable_score` (1.0 for every shipped needle/depth).
* `masking.greedy_generate(eos=None)` referenced an undefined `tokenizer` and raised
  `NameError`; the test that "covered" it monkeypatched the function itself.  Fixed,
  with a test that goes through the real default path.
* **The needle's gold tokens now come from the needle text, not the prompt span.**
  The span's last token can be the fused `".\n"` while the model emits `"."`, which
  capped NIAH-F1 at 21/22 = 0.955 (exactly the old baseline) and made the emitted
  final token uncreditable.  Scoring uses `set(span_ids) | set(text_ids)`, the
  denominator is the unique *text* count, F1's gold is the text tokenization, and
  `span_straddles_boundary`/`needle_text_ids` are recorded.  Absolute F1 and score
  ceilings change; ranking barely does.
* Two "safety" warnings were dead and now fire: the summarizer's stale-schema check
  (`warn_if_stale` was called without its required `log=`, and the script never put
  the repo root on `sys.path`, so it died in a bare `except`), and the detection
  budget warning (`InstanceResult.sample` has no `truncated` key -- it is in `.meta`).
  On the smoke profile the latter now reports 1/1 instances hitting
  `max_new_tokens=16` for a 22-token needle, i.e. those numbers are budget-limited.
* **The causal ablation no longer scores on the selection needle.**  Detection uses
  `DETECTION_NEEDLES`; the masking curve uses a held-out `EVAL_NEEDLES` pair (the
  paper's "additional set of needle tests"), `assert_needles_disjoint()` guards it,
  and the needle text/question/source are recorded in the artifact.  Previous
  masking numbers came from needle #0, i.e. the one the heads were chosen on.
* `token_mixer_ablation` now averages over random layer subsets (`--mixer-trials`,
  default 3) instead of always masking the *earliest* K of each stack.  Measured on
  Qwen3.5 smoke: the linear arm moves ±29.3 F1 across just two subsets, so the old
  single deterministic subset (layers 0,1 vs the attention stack's 3,7) was
  confounding layer depth with mixer type.  Numbers must be regenerated.
* The fast suite now builds a **tiny local hybrid** (`tests/test_tiny_model.py`,
  4 layers, ~50k params, no download) and covers architecture discovery, both
  attention-capture methods, masking a real `o_proj`, chunked-prefill equivalence
  and the loader's recorded class/dtype -- the things that used to need the
  integration suite. `capture_method` now defaults to `patch`, which keys maps by
  `layer_idx` and cannot be misled by attention-map ordering.

**Not done.** Paper-scale grid, real datasets, the Paul Graham haystack, and a
re-run of the current code. Details in §6.

Branch: **`Nikita-prog-art`** at `94f81d7`, pushed. `main` on the remote is
untouched at `a18ec09`. The working tree is **dirty**: the current corrections are
not committed yet.

---

## 2. Environment on this machine

| path | what |
|---|---|
| `.venv/` | local CPU env: torch 2.14.1+cpu, transformers 5.18.0, pytest |
| `.venv-datasphere/` | DataSphere CLI 0.10.0 (Python 3.13) |
| `models/` | Qwen3.5-0.8B + Qwen3-0.6B, SHA-256 verified (≈3.2 GB, gitignored) |
| `configs/datasphere/*.yaml` | job configs, **in git** |
| `.cache/datasphere/iam-token` | IAM token cache — a secret, gitignored, never commit |
| `ds-results/`, `results/` | GPU run and earlier CPU run artifacts |

Auth (token lives ~12 h, cached):

```bash
export PATH="$HOME/yandex-cloud/bin:$PATH"
source scripts/datasphere_auth.sh          # -> YC_IAM_TOKEN
CLI=.venv-datasphere/bin/datasphere
PROJECT=bt1u5v72b71eesdhp9k5
```

The account is federated: when the session expires, `yc` opens a browser tab. If
that happens repeatedly, refresh the session by hand once — the script caches the
IAM token for ~11 h afterwards.

---

## 3. Reference artifacts (use these instead of re-running)

| job id | what it is | outcome |
|---|---|---|
| `bt18bmvbuggt68cnel9p` | `rh-t4-resume`, qa/cot/compare/figures on the previous detect/mask | **SUCCESS**, ~17 min, completed the committed `ds-results/` |
| `bt1ethsb4jlpds7m6i32` | `rh-t4-cached`, describe/detect/mask for both models | ran ~31 min, then failed on `qa` (an `AttributeError` of mine); its artifacts were reused by the resume job |
| `bt1hqv1b91ht36s2egdp` | `rh-t4-cached`, full `t4` pipeline, both models | **SUCCESS**, ~39 min, the previous (now superseded) `ds-results/` |
| `bt107kjm3es8vung7130` | `rh-t4-smoke`, all 7 stages at smoke scale | **SUCCESS**, proves the GPU path |
| `bt1k1hndn5lm6ld0lvjc` | venv bootstrap on the project disk | env built; died later on a code bug |
| `bt1e0iq7jakl30kd3qjn` | first full run, float32 | **OOM** at 16K — see findings §18 |

Job link pattern:
`https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5/job/<job_id>`

---

## 4. Resuming

Local sanity first — always, it is 8 seconds:

```bash
.venv/bin/python -m pytest -m "not integration" -q
```

Summarise whatever results exist (works on partial runs too):

```bash
.venv/bin/python scripts/summarize_results.py ds-results --out docs/results-gpu.md
# the older CPU tree is not committed as a report; summarise it on demand:
.venv/bin/python scripts/summarize_results.py results
```

A GPU job (the cached venv already exists, so this starts in ~40 s):

```bash
$CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-cached.yaml --async -o .tmp/job.json
$CLI project job attach --id <job_id>          # streams; blocks until it finishes
```

Cheap reads on the shared disk, writes nothing:

```bash
$CLI project job execute -p "$PROJECT" -c configs/datasphere/inspect-disk.yaml
```

After a run finishes, artifacts download automatically to `ds-results/`.

---

## 5. Gotchas that cost real time here

Ordered by how expensive they were to rediscover.

1. **pip 25.1.1 aborts the env build.** `AssertionError` in
   `get_topological_weights` whenever one project is required twice
   (`huggingface-hub`, by both `transformers` and `tokenizers`). Fix already in
   place: `scripts/requirements-datasphere.txt` is a complete exact lock and every
   job sets `pip.no-deps: 'true'`. **Do not add comments to that file** — the CLI
   runs `packaging.Requirement()` over every line, comments included.
2. **A 16K float32 prefill OOMs on a 22 GiB L4.** float32 SDPA does not reach a
   flash kernel and can fall back to the math backend, materialising
   `(heads, seq, seq)`. Fixed twice over: `--dtype bfloat16` on GPU, and
   `--prefill-chunk 4096` in code, with an equivalence test.
3. **`local-paths` snapshots the code when the job is *created*.** A fix in the
   working copy does nothing for an already-created job. Always launch a new one.
4. **`cmd` cannot start with `${DS_PROJECT_HOME}/…/venv/bin/python`** — the CLI
   validates the first token as an interpreter. The entry point stays `python3`
   and the driver re-execs itself (`--use-venv`).
5. **The entry point is `scripts/datasphere_job.py`**, not `datasphere_job.py`:
   `local-paths` unpacks each entry under its own name into `/job`.
6. **The image has two Pythons.** `/usr/bin/python3` is 3.11 while the job runs
   3.10.12, and the lock pins `cp310` wheels. The bootstrap matches
   `sys.version_info` exactly and aborts otherwise.
7. **venv is rebuilt per job; weights are not.** `inputs` are cached by the
   project (3.1 GB uploaded once), the Python env is not — hence the cached venv
   and `configs/datasphere/t4-cached.yaml`.
8. **Read the realized numbers, not the requested ones.** `instances_*.jsonl`
   records `prompt_tokens`; three separate measurement bugs here were only visible
   as implausible values in a real run (findings §19 and the README Limitations).
9. **`gt4i.1` hands out an L4** (sm_89, 22 GiB), not a T4 — bf16 is native there.
10. **`/tmp` is wiped between tool calls in this harness**, and long `sleep`
    polling blocks the conversation. Use `.tmp/` and background jobs.

---

## 6. Pending work

Each item says what to do, not just what is missing.  Everything that could be
finished offline is done (trace memory, greedy-loop duplication, registry/`--dtype`,
grid-mode guards, NaN handling, figure bugs, download/auth hygiene, and the review
follow-up: `schema_version` + provenance on every artifact, dtype recorded in
`ModelInfo`, real EOS-stop flag, per-sample metrics and spread, decimal-safe
`accuracy`, honest `head_overlap` in sorted mode, `--thinking` wired for qa/cot).
What is left needs a GPU run, external data, or a judgement call.

Known measurement limits, in the artifacts themselves rather than hidden:
`mask` now keeps per-sample F1 and reports the retrieval arm's spread, and its
default is 5 samples per point -- still not a statistically powered curve, which is
what the paper-scale grid is for.  K in the retrieval arm is capped by the control
pool size; the artifact records both requested `k_values` and realized
`k_effective`, and `summarize_results.py` prints "K →k_eff".  `scripts/case_study.py`
(Fig. 1) is still not a job stage, so it has to be run by hand.
`aligned_scores` are now ranked in `summary_*.json` (`aligned_top_heads`) and shown
as the "Strict-aligned matching" section of `summarize_results.py`.  Both pairings
now score exactly the generated stream: the prefill row is `next_step`-only and the
final truncated decode row is `same_step`-only, so neither credits a token that was
never emitted.  `sink_rate` is stored per pairing, and `random_retrieval_overlap`
is measured against the heads the retrieval arm actually masked at that K.
`masking_curve.json`/`mixer_ablation.json` now record the seed and the masked
head/layer subsets, and the detection summary carries recited-only
(`sparsity_recited`, `scores_*_recited.npz`) alongside the unconditional numbers.
`needle_stats.denominator_inflation` records how much the unique-token denominator
inflates the score (measured 22/19 = 1.158 on the eval needle).  The
artifact records which lengths the context window rejected (`dropped_lengths`),
and the masking token-F1 skips special ids so it matches the exact-match metric.

Open oddities that were looked at and left alone: `evaluate_extractive_qa` scores
F1 over the whole completion rather than an extracted span (so short answers can be
inflated); `retrieval_std` is across samples while `random_std` is across trials;
the `except KeyError` around the secondary aggregation is effectively unreachable
because `score_instance` always records both pairings.

Left as-is on purpose (recorded rather than fixed):

* `scripts/test_cli_argv.py` still only checks that the reproduce scripts use
  `--k-frac`; a full `--lengths/--depths/--random-trials` parity check against
  `SCALES` is not written.
* `reference` note: `_patched_eager` patches the modeling module's module-global
  `eager_attention_forward`, so a module that did `from ... import
  eager_attention_forward` into its own namespace would not be intercepted -- the
  "captured nothing" error catches it, but the limitation is documented here.
* `set_attn_implementation` mutates the shared config object, so it is not
  thread-safe; fine for the batch CLI.
* `capture_method="output_attentions"` still trusts the order of the returned map
  tuple; the default `patch` keys by `layer_idx`, and a fast test on the tiny hybrid
  checks that both paths agree, so the public path is covered in CI.
* `build_needle_sample` cannot hit its 2% length contract for `target_tokens < 64`
  (`max(64, budget)`); the real profiles are far above that.

Two methodology items from the review are deliberate non-changes, because both
would move the numbers and want a run to justify them:

* **The unique-token denominator is kept.**  `|g_h ∩ k| / |unique(k)|` deviates from
  a literal per-token reading of the paper's formula; the artifacts now record
  `needle_tokens_mean`, `unique_needle_tokens_mean` and the inflation ratio so a
  reader can rescale, but no parallel per-token matrix is emitted.
* **No conditioning on recitation.** `score = |g_h ∩ k| / |unique(k)|` averages
  over all instances, so "the head did not retrieve" and "the model did not recite
  the needle" are mixed. `considered` is stored per head in the JSONL, so a
  conditioned variant can be computed by hand, but no such column is emitted.
* **`token_mixer_ablation` always masks the first K layers.** Unlike the head
  ablation there is no random-subset arm, so early-vs-late layer position is
  confounded with "how much of this stack matters". Averaging over random subsets
  would be the honest version.

**Committed** as `d283b63` (whole tree, including the previously untracked CI
workflow, `generation.py`, `provenance.py` and the test files).  Not pushed.

**Process blocker: much of the code is untracked.**  `git ls-files` does not know
`.github/workflows/tests.yml`, `retrieval_heads/generation.py`,
`retrieval_heads/provenance.py`, `tests/test_regressions.py`,
`tests/test_tiny_model.py` or `tests/__init__.py`.  Until they are staged, GitHub
Actions will not run at all, and a commit that takes only the *modified* tracked
files will break the imports (`scoring.py`/`masking.py`/`downstream.py` import
`generation` and `provenance`).  Stage the whole tree, not a subset.

**Done / no longer blocked**

1. ~~**Regenerate every artifact.**~~ Done: `ds-results/` and
   `docs/results-gpu.md` were regenerated from the two jobs above (schema 5), and
   the "stale" banners are gone from the README and the results doc.  The numbers
   moved as expected: dense `>0.1` 4.9% -> 6.2% with the zeroed/weak shares now
   inside the paper's Fig. 2 ranges, hybrid 62.5% -> 68.8%, all 75 instances
   recited (LCS recall), and the dense exact-match collapse starts at ~4% of heads
   rather than 2%.  `results/` (CPU) was deliberately left historical.

**Blocked on a GPU run / external input**
2. **Paper-scale grid.** `--profile paper` with
   `--lengths 1024 … 49152` and 10 depths. VRAM is fine with bf16 + chunking, but
   budget the time; consider `gt4i.1` vs `g2.1` (A100) for the widest contexts.
   Edit `SCALES["paper"]` in `scripts/datasphere_job.py` or add a config. Lengths
   past a model's `max_position_embeddings` are now dropped with a warning.
3. **Real datasets.** `--data file.jsonl` on `qa`/`cot`:
   `{"context","question","answer"}` and `{"question","answer"}`. The built-ins are
   calibrated stand-ins, not benchmarks.
4. **Paul Graham haystack.** The canonical NIAH haystack corpus is at
   `source/PaulGrahamEssays/*.txt`; `--corpus <file>` uses it on `detect` and now
   also on `mask` (so detection and the causal experiment share a haystack, and the
   artifact records which one). **Do not read the rest of `source/`** — the user
   explicitly said the original authors' code is not to be used (§7).
5. **`same_step` / `credits_aligned` are stored, but not yet reported.**
   `score_instance` writes `aligned_scores` for both pairings into the per-instance
   JSONL, and `DetectionRun.save` writes `scores_same_step.*` +
   `summary_same_step.json`, so the regenerated run will have a per-head
   `same_step` table. `summarize_results.py` already emits both pairings in the
   detection table and the 0/10 pairing-overlap table. Still pending: the actual
   regenerated numbers (blocked on item 1).
6. **`flash-linear-attention` + `causal-conv1d`.** Qwen3.5's 18 Gated DeltaNet
   layers currently run on the pure-PyTorch fallback. Speed only, not correctness,
   but it dominates its runtime.
7. **bf16 vs fp32 sensitivity.** GPU numbers are bf16, CPU numbers fp32. A single
   same-instance comparison would say whether the argmax over attention is stable
   enough that this does not matter.
8. **`mixer_ablation` needs a fairer control.** 6 full-attention layers vs 18 linear
   ones is not a matched comparison, and one linear layer carries far more of the
   residual stream. The docstring says so; the figure now labels which K it draws,
   but the comparison itself is still unmatched.

---

## 7. Do not

* **Do not push to `main` or to anyone's branch.** `main`, `c0`, `justamouse`,
  `nastya`, `timon` on the remote belong to other people. This work lives on
  `Nikita-prog-art`.
* **Do not read or reuse `source/`** (`retrieval_head_detection.py`, `faiss_attn/`,
  `viz/`, …). The user was explicit that the original authors' code is the one
  thing not to look at, and this implementation was written without it. Keep it
  that way. `source/PaulGrahamEssays/` is the exception — it is a text corpus.
* **Do not delete anything on the shared project disk.** `ai360-heads-in-the-haystack/`
  is yours; `Justamouse/`, `Nikita-prog-art/`, `timon/`, `logs/` are not. The job
  driver has no delete capability *by design* — keep it that way. A stray
  `rh-venv/` at the disk root is an abandoned artifact of a cancelled job and was
  proven ours (findings §12); leave it.
* **Do not put anything important in `.cache/`.** It is disposable and fully
  ignored. Job configs live in `configs/datasphere/`.
* **Do not commit `.cache/datasphere/iam-token`.**
* **Do not trust a baseline of zero.** Two of the four measurement bugs here were
  ablations that could not move. Check that the baseline has headroom before
  believing the effect.

---

## 8. Definition of "still working"

```bash
.venv/bin/python -m pytest -q                     # 222 passed (205 fast + 17 integration)
.venv/bin/python -m retrieval_heads.cli describe --model qwen3.5-0.8b
# -> 6 scoreable layers [3,7,11,15,19,23], 48 scoreable heads, hybrid: True
.venv/bin/python -m retrieval_heads.cli describe --model qwen3-0.6b
# -> 28 scoreable layers, 448 scoreable heads, hybrid: False
```

If `describe` reports 24 scoreable layers for Qwen3.5, the architecture-aware head
discovery has regressed and every retrieval score after that is meaningless.
