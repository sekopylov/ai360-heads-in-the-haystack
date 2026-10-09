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
* 253 tests: 236 fast (`pytest -m "not integration"`, ~10 s), 17 integration against
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
* **The project-disk venv now has `flash-linear-attention`** (job
  `bt1130t5lg7audlevlin`, SUCCESS, stamp `1ae8818bae664fb2`): `fla-core` +
  `flash-linear-attention` + `einops` are in `requirements-datasphere.txt`, and the
  job log confirms `fla.ops.gated_delta_rule: ok`, so the hybrid's 18 linear layers
  no longer run the reference PyTorch delta-rule.  `causal-conv1d` is deliberately
  absent -- the job image ships only CUDA 11.8 against a cu128 torch, so the
  extension cannot be compiled there (see `run-in-datasphere.md` §3.3 for the exact
  error and the nvcc-from-pip alternative).  The same log answers the A100 questions:
  `torch arch list` includes `sm_80`, and `SDPA flash backend: ok`.  Every artifact
  now records `provenance.optional_kernels`, so a run with the fused kernels is
  distinguishable from the committed `ds-results/` (both flags `false` there).
* `configs/datasphere/t4-venv.yaml` refreshes the venv without running any stage, and
  `configs/datasphere/cuda-probe.yaml` is a read-only `--inspect-dir` job that shows
  what the image and the venv contain (it is how the single 11.8 toolkit was found).
* **`configs/datasphere/a100.yaml` is ready for the A100** (g2.1 only, no fallback):
  the `paper` grid with `--prefill-chunk 8192` (not one-shot) and a 96-token detect
  budget, carried by the new `a100` scale in the driver.  Since round six the `a100`
  scale also pins `--argmax-domain haystack` (the paper's `a in R^{|x|}`, and the code
  default) and spends the mask stage's 15-sample budget as 3 held-out needles x 5
  depths instead of 1 x 10.  The first version used
  `--prefill-chunk 0` on the theory that 49K in 4096-token chunks cost twelve prefills
  -- it does not: every token belongs to one chunk, so layer work is unchanged and
  only the attention term grows, by `(n+1)/n` (~8% at 12 chunks).  Chunking is what
  bounds the score matrix at `O(chunk x seq)` and keeps a float32 SDPA fallback from
  materialising `(heads, seq, seq)` (which OOM'd a 22 GiB card at 16K, findings §18),
  so giving it up for ~8% was a bad trade.  Price check (RU, incl. VAT): L4
  234.00 RUB/h vs A100 542.88 RUB/h, i.e. 2.32x -- the A100 buys wall clock, and only
  long-context runs can pay for it.  Order of operations: validate on the L4 `t4` grid,
  then launch this.  Honest grid note: the paper's detection grid is 20 uniform
  lengths (~600 instances/model); this profile is 9 geometric ones (270), and the
  dense model loses the two longest to its 40960 window (210).  Widening `--lengths`
  is the cheapest way to make the A100 run genuinely paper-scale.
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
* **Review round (the sixth report: B1 + C6 + the offline backlog), verified against
  the code:**
  * **B1 -- the `haystack` argmax domain is implemented and is now the default.**
    `NeedleSample` gained `haystack_span` (the context's token span, located by
    anchoring on the needle rather than `prompt.find(context)`, and verified verbatim;
    `meta['haystack_span_verbatim']` says whether it was).  `argmax_positions` takes a
    `span`, `decode_with_attention`/`score_instance` take `argmax_span` and validate
    it *before* the forward pass, `detect`/`case_study` expose the third choice, and
    `score_instance` defaults the span to the sample's own.  `DetectionConfig`,
    the CLI and `case_study` now default to `haystack`; every job scale and both
    reproduce scripts pin it explicitly, so a future default change cannot silently
    alter a job.  No extra forwards: the captured rows are identical in all three
    domains.
  * **The sink rate had to be decoupled from the domain.**  Under `haystack`,
    position 0 (a template token) is ineligible, so a `sink_rate` read off the scoring
    argmax would be a structural zero.  `StepTrace` now carries `argmax_prompt` --
    the prompt-restricted argmax -- under every domain, and `credits_from_trace` takes
    the sink from that.  For the committed `prompt`-domain artifacts the value is
    unchanged (the two argmaxes are the same tensor), so 0.759 still reproduces.
  * **`argmax_domain_shift` is recorded per instance and summed into `scores.meta`**:
    the share of `(layer, head, step)` positions whose domain argmax differs from the
    prompt one.  That is the direct, free evidence of how much B1 moves the numbers,
    and it is what the next run will report instead of an argument.
  * **The measurement that came out of B1, and it is not small: the sink geometry
    decides the answer.**  On one Qwen3-0.6B instance at 512 tokens, 90.6% of argmax
    positions move between the two domains, and the >0.1 share goes from **34/448**
    (`prompt`) to **173/448** (`haystack`) -- with the sink rate identical (0.752) in
    both, which is the sink decoupling working.  But the reason is not the question or
    the template *text*: it is that a chat template puts the attention sink at
    sequence position 0, i.e. **before** `x`, so it can never win the argmax and
    criterion (2) becomes reachable for many more heads.  With `--no-chat-template`
    (the paper's geometry: context first, so position 0 *is* the first haystack token)
    the same instance gives **13/448**.  `NeedleSample.haystack_includes_sink`,
    `scores.meta['sink_in_haystack']` and `summary_*.json` now record which geometry a
    run used, and `configs/datasphere/a100.yaml` carries the open decision (the
    template-free prompt is faithful but cost needle recall on that instance, 0.40
    against 1.00).  Three depths at 512 tokens give 151/173/236 for `haystack` against
    23/34/40 for `prompt`, so the direction is stable; the absolute shares are a
    single-instance demonstration, not an estimate.
  * **C6 -- numbers are compared numerically, not as strings.**  `_normalise`
    canonicalises decimal literals (`63.0` -> `63`, `0.50` -> `0.5`, `3.10.12` left
    alone) and `accuracy` compares the first number with `math.isclose`.  `word_f1`
    gets the same treatment for free because it tokenises the normalised string.  No
    built-in item changes (they use one spelling); the bug only bites on real
    datasets, which is exactly when it would have been expensive.
  * **`EVAL_NEEDLES` now holds three held-out needles** (the ablation was measured on
    one), `mask --needles 3` is a real request rather than a clamp, and the A100 scale
    spends its 15-sample budget as 3 needles x 5 depths instead of 1 x 10 -- needle
    identity is the larger source of variance.  `--needles 0` is now rejected instead
    of clamped to 1.
  * **The QA ablation records the retrieval arm's per-sample F1** (`retrieval_f1s`,
    `retrieval_f1_std`) plus the baseline's, via a new `extractive_qa_scores` that
    `evaluate_extractive_qa` wraps.  `qa_ablation`/`cot_ablation` had **no test at
    all**; there is now `tests/test_downstream_ablation.py` (matched arms, control
    contamination, the per-sample spread), which also exposed that with one
    above-threshold head and K=2 the retrieval and random arms genuinely overlap --
    recorded in `random_retrieval_overlap`, now asserted.
  * **The raw-denominator matrices are emitted** (`scores_<pairing>_raw.npz/.json`,
    `summary_*.json`'s `sparsity_raw`, `InstanceResult.scores_raw`), so the other
    reading of the paper's `|k|` needs no rescaling.  On the committed dense grid it
    puts 24/448 heads above 0.1 against 28 under `|unique(k)|` (computed from the
    committed JSONL, which carries `copied_tokens` and `needle_text_ids`).
  * Smaller: `finite_json`/`_json_default` recurse into `as_dict()` objects and numpy
    scalars (a NaN inside one used to reach `json.dump` and raise);
    `_inhomogeneous` records `None` instead of `0` and `require_matching_scores`
    rejects an unknown width; `activation_gap` flags a single-instance payload and the
    Fig. 3 title stops saying "always-active"; `build_needle_sample` keeps the
    *closest* of its four attempts (the loop oscillates, and at target 256 the last
    attempt could be 8 tokens out) and names the needle+question+template floor when
    the target is below it.
  * **`_patched_eager` patches every reachable namespace, not just
    `type(module).__module__`.**  It now inspects each method's `__globals__` (the dict
    the name is looked up in, which differs from the class's module for a subclass) and
    any imported module object in them (`import x` + `x.eager_attention_forward(...)`).
    Both aliasing forms were silent misses; `tests/test_attention_capture.py` pins
    them plus the foreign-block and nothing-to-patch cases.
  * **The reproduce scripts are now parity-checked against `SCALES`** flag by flag,
    and the check immediately found real drift: `reproduce_gpu.sh` ran `cot` at the
    192-token CLI default while `SCALES['paper']` pins 256.  Fixed, and
    `--argmax-domain` joined the compared set.
  * **`test_regressions.py` (1700 lines) is split by topic** into
    `test_cli_regressions.py`, `test_detection_regressions.py`,
    `test_masking_regressions.py`, `test_models_regressions.py`,
    `test_plotting_regressions.py` and `test_io_regressions.py`, with the four shared
    helpers in `tests/_helpers.py`.  The split was a mechanical move (whole top-level
    blocks verbatim) verified by an identical collected-test count: 253 pass, the same
    as before.
  * Left as documented non-changes, with reasons: `capture_method="output_attentions"`
    still trusts the order of the returned map tuple (count + head counts are
    validated, the tiny hybrid cross-checks both paths, and the default is `patch`
    keyed by `layer_idx`); `set_attn_implementation` still mutates the shared config
    (not thread-safe, fine for the batch CLI); `HeadMasker` still clones the `o_proj`
    input (an out-of-place slice assign allocates the same bytes).
* **Review round (the fifth report, before the A100 run), verified against the code:**
  * **My `--prefill-chunk 0` justification was wrong** (see the A100 bullet above) and
    is corrected in the config, in the driver's scale table, in two docstrings
    (`O(chunk x seq)`, not `O(chunk^2)`) and in the test, which now pins 8192 rather
    than 0.
  * `--max-new-tokens 96` for the A100 detect: on the t4 grid 48 tokens truncate 11 of
    the hybrid's 75 instances, and truncated instances score *lower* (0.684 vs 0.793 on
    the top head) because the model answers and then keeps narrating -- the extra
    narration is exactly where more needle-token copies could come from.
  * `mask` gained `--depths` and `--needles` (the sample set was hard-coded to 3 x 5 x 1
    = 15, and `retrieval_std` is the spread over exactly those); the `a100` scale uses
    10 depths.  `EVAL_NEEDLES` still holds one needle, so widening further needs more
    held-out needles first.
  * The driver's `--profile` is now `--scale` (with `--profile` kept as an alias),
    because the CLI's `--profile {smoke,laptop,paper}` and the driver's scale were two
    different namespaces sharing one word.
  * `provenance._git_state()` remembers the first failure: a job has no `.git`, so it
    was spawning two doomed subprocesses per artifact (thousands on a paper-scale run).
  * A fast tripwire reads the committed `ds-results/` and asserts the dense model's
    >0.1 share stays in a guard band (2-12%; measured 6.2%, marginally above the
    paper's 3-6%) and the hybrid's stays above 50% -- the integration test that used to
    guard this is not in CI.
  * Verified and *not* fixed: the reviewer's numbers all reproduce (dense sink >0.9 for
    285/448 heads, mean 0.759; hybrid 0/48, 0.035; the 11 truncated hybrid instances).
    `HeadMasker`'s full `clone()` is not a real win (an out-of-place slice assign
    allocates the same bytes), and the numeric-comparison nit in the QA scorer only
    matters once real datasets are wired in.
  * **Not done, and the biggest remaining methodological gap:** the `haystack` argmax
    domain.  The paper's `a ∈ R^{|x|}` is the haystack alone, while `prompt` includes
    the question and the chat template; on the dense model that is not a detail (285 of
    448 heads put their argmax on position 0, a template token), so 6.2% is a *lower*
    bound.  Implementing it needs a `context_span` on `NeedleSample` (the char-offset
    machinery is already there), an `argmax_span` argument through
    `decode_with_attention`/`score_instance`, and a third choice for `--argmax-domain`
    -- no extra forward passes, but a real code change, so it is next rather than now.
* **Review round (the fourth report), verified against the code:**
  * **`HeadOverlap.mode` was never set** (`head_overlap` used `mode` for the
    correlation but never passed it to its own constructor), so `overlap.json` said
    `mode: grid` next to a correlation computed in `sorted` and next to a caveat
    about sorted -- the artifact contradicted itself, and `cmd_compare` printed a
    *different* payload than it wrote.  `mode` and the caveat now travel together,
    set in `properties` (`SORTED_MODE_CAVEAT`), and the test that was supposed to
    catch it only read the default; it now exercises `mode="sorted"`.
  * `credits_aligned` still had the `min()` clamp that `credits_from_trace` had
    already been fixed for; both now share `_validated_head_count`.
  * `SCHEMA_VERSION` 5 spanned three commits that added fields *and* changed
    semantics, so `warn_if_stale` said "fine" for artifacts that predate them.
    Bumped to 6; the summarizer now emits 22 warnings on the committed `ds-results/`,
    which is the honest signal.  The README caveat now lists *every* missing field
    (truncation counters, `control_exhausted`, `random_distinct`, the QA/CoT
    conditions, `aligned_top_heads[].n`, `distinct_subsets`, `code_sha256`).
  * The control subsets are drawn without repetition now -- which is **not** an
    additive change: the committed hybrid K=1 point has a duplicate
    (`[L3H2, L3H2, L11H3]`), so its `random_std` came from two distinct
    interventions.  The README said "no number changes"; it now says exactly which
    arm a re-run would change.  The same drawing rule now also applies to
    `qa_ablation`/`cot_ablation`, via one shared helper.
  * `--random-trials 0`, `--max-new-tokens 0` (for `mask`/`qa`/`cot`) and an explicit
    empty `--lengths` are rejected instead of producing all-zero metrics or `null`
    spreads; `detect` already refused the zero budget, so the commands now agree.
  * Smaller: `plot_attention_distribution`'s `top_n` is applied instead of ignored;
    the exact-match legend no longer promises bars it does not draw; `cmd_compare`
    prints `finite_json` (stdout used to be invalid JSON with `NaN` while the file
    was fine); `HeadRef` is re-exported from `utils` rather than from `models` (which
    merely imported it); README's "18 of 48 heads" is 15, and "within ~1%" is stated
    for the means (per-instance the worst case is ~2%).
  * Documented, not hidden: the `prompt` argmax domain is the rendered prompt
    (haystack + question + template), not the paper's haystack-only `x`, so extra
    positions can only withhold credit -- a haystack-only variant is not implemented;
    and `token_mixer_ablation`'s dense branch is reachable from the API but not from
    the CLI.
  * Not done: per-sample std for the QA retrieval arm (so those bars stay absent),
    and the 2-3-needle ablation grid (GPU run).
* **Review round (the third report), verified against the code:**
  * **The pairing invariant was wrong on the EOS path** (the common one).  A step is
    recorded after the EOS check, so the last *recorded* decode step always predicted
    a token that was never fed; the re-scoping was conditioned on `not
    stopped_on_eos`, so only the truncation path was fixed.  Rows are now scoped by
    what actually happened: the last decode row is `same_step`-only, and the prefill
    row is `next_step` only if something was generated at all (an immediate EOS means
    nothing was).  The test now asserts `len(stream) == len(generated)` for both
    exits.
  * `require_matching_scores` compared only layer/head geometry, which a base model
    and its chat variant share -- the Sec. 4.3 case.  It now also checks
    `hidden_size`, `model_class` and `num_kv_heads`, and warns (not fails) on a name
    mismatch, since a job's paths legitimately differ.
  * `credits_from_trace` silently dropped attention rows beyond the metadata's head
    count (`min`); it now refuses a mismatch in either direction.
  * The control pool is a *cap* on the retrieval arm: at the hybrid's last point the
    random arm drew the whole 15-head pool.  `control_exhausted` marks such points in
    the artifact and the summary table, and `random_distinct` records how many
    control subsets were actually distinct (they are now drawn without repetition).
  * `case_study.py` ignored every recorded condition except the argmax domain; it now
    takes `--scores <detect dir>` and reuses `chat_template`, `system_prompt`,
    `enable_thinking`, `capture_method` (plus `--dtype`, which was not even recorded).
  * `aligned_ranking` reports `n`/`n_missing`; the ablation records truncation; the
    dead `ids[-1] not in eos` clause is gone (`greedy_ids` never appends the stop
    token, so a full-length output *is* the budget signal); `[list(layers)] * n`
    aliased one list; the summarizer no longer loses the schema check silently in an
    environment without torch (it loads `provenance.py` by path and warns); the pie
    figure now labels both boundaries (the run's threshold and the paper's >0.5);
    and the docs no longer imply `figures` writes `retrieval_attention_dist.pdf`
    (that is `case_study.py`, which is in no stage).
  * Documented rather than hidden: `ds-results/` was written at `abbfc3c`, so it
    lacks the additive fields added since (truncation counters, `control_exhausted`,
    `random_distinct`) -- no number changes, and a `mask` re-run would add them.
  * Not done: a 2-3-needle ablation grid (GPU run) and splitting
    `test_regressions.py` by topic.
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
re-run of the current code (which now defaults to the `haystack` argmax domain, so a
re-run's detection numbers are expected to move). Details in §6.

Branch: **`Nikita-prog-art`**. `main` on the remote is untouched. See
`git log --oneline -1` for HEAD; the working tree holds the round-six changes.

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
finished offline is done, including round six (the `haystack` argmax domain and its
default, numeric answer comparison, three held-out eval needles, per-sample QA
spread, the raw-denominator matrices, the small guards, the aliased-`eager` patch,
the reproduce-script parity check, and the test-file split).  What is left needs a
GPU run, external data, or a judgement call.

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

* `set_attn_implementation` mutates the shared config object, so it is not
  thread-safe; fine for the batch CLI.
* `capture_method="output_attentions"` still trusts the order of the returned map
  tuple; the default `patch` keys by `layer_idx`, and a fast test on the tiny hybrid
  checks that both paths agree, so the public path is covered in CI.  (The *patch*
  path's own namespace problem -- `type(module).__module__` vs `forward.__globals__`
  vs an imported module object -- was fixed in round six.)
* `HeadMasker` clones the whole `o_proj` input per masked layer: an out-of-place
  slice assign allocates the same bytes, so there is no cheap win here, and the peak
  still grows with context length x masked layers.

Done in round six, kept here because the items used to be in this list:

* ~~A full `--lengths/--depths/--random-trials` parity check of the reproduce scripts
  against `SCALES`~~ -- written, and it found real drift (`reproduce_gpu.sh` `cot`
  ran at the 192-token default instead of `SCALES['paper']`'s 256).
* ~~`_patched_eager` misses an aliased `eager_attention_forward`~~ -- it now walks
  each method's `__globals__` and the module objects in them.
* ~~`build_needle_sample` cannot hit its 2% contract for small targets~~ -- it keeps
  the closest attempt and names the needle+question+template floor when the target is
  below it (64 tokens is unreachable by construction, and now says so).
* ~~Splitting `test_regressions.py` by topic~~ -- six topical files plus
  `tests/_helpers.py`, verified by an unchanged collected-test count.

One methodology item from the review is now *emitted* rather than documented:
the **raw per-token denominator** is a first-class artifact (`scores_*_raw.*`,
`sparsity_raw`), so the alternative reading of the paper's `|k|` needs no rescaling.
The unique-token denominator is still the primary one.

**Still a deliberate non-change: no conditioning on recitation.**
`score = |g_h ∩ k| / |unique(k)|` averages over all instances, so "the head did not
retrieve" and "the model did not recite the needle" are mixed.  `considered` is
stored per head in the JSONL and `sparsity_recited` / `scores_*_recited.*` carry the
recited-only view, so the conditioned variant is computable; it is not the headline
matrix.

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
5. **A run under the new `haystack` default.**  Everything that was "stored but not
   reported" is reported now: both pairings have per-head tables, `same_step` has its
   own `scores_same_step.*` / `summary_same_step.json`, `aligned_top_heads` is in the
   summary and printed by `summarize_results.py`, and the raw-denominator matrices are
   emitted.  What is pending is the *numbers* from a run that uses the current default
   (`haystack`), since the committed tree is `prompt`-domain -- i.e. the A100 job.
6. **`flash-linear-attention` is installed on the project disk; `causal-conv1d` is
   deliberately absent** (the job image ships only CUDA 11.8 against a cu128 torch, so
   the extension cannot be compiled there; see `run-in-datasphere.md` §3.3).  Speed
   only, not correctness.  Every artifact records `provenance.optional_kernels`, so a
   fused run is distinguishable from the committed one.
7. **bf16 vs fp32 sensitivity.** GPU numbers are bf16, CPU numbers fp32. A single
   same-instance comparison would say whether the argmax over attention is stable
   enough that this does not matter.  (`argmax_domain_shift` will not answer it: it
   compares domains, not dtypes.)
8. **`mixer_ablation` needs a fairer control.** Averaging over random layer subsets is
   implemented (`--mixer-trials`, default 3, recorded per trial), so layer *position*
   is no longer confounded with stack size; what remains unmatched is 6 full-attention
   layers against 18 linear ones, and one linear layer carries far more of the
   residual stream.  The docstring and the figure say which K is drawn.

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
.venv/bin/python -m pytest -q                     # 253 passed (236 fast + 17 integration)
.venv/bin/python -m retrieval_heads.cli describe --model qwen3.5-0.8b
# -> 6 scoreable layers [3,7,11,15,19,23], 48 scoreable heads, hybrid: True
.venv/bin/python -m retrieval_heads.cli describe --model qwen3-0.6b
# -> 28 scoreable layers, 448 scoreable heads, hybrid: False
```

If `describe` reports 24 scoreable layers for Qwen3.5, the architecture-aware head
discovery has regressed and every retrieval score after that is meaningless.
