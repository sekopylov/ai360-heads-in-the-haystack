# HANDOFF — state of the retrieval-heads reproduction

**Working document.** It is a snapshot for whoever picks this up next, not part of
the deliverable — and it is versioned with the code (the header used to say
"deliberately not committed", which stopped being true at the initial commit and has
been wrong in every round since). Delete it when it goes stale.

Read in this order: `README.md` (what this is, how to run it) →
`docs/datasphere-findings.md` (19 recorded divergences from the infra doc, each
tied to a job id) → this file (where things stand and what is left).

---

## 1. State at a glance

**Done and verified end to end.**

* `retrieval_heads/` — the paper's method, architecture-aware. 12 modules.
* 281 tests: 263 fast (`pytest -m "not integration"`, ~20 s), 18 integration against
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
* **Review round (the tenth report, pre-A100; the reviewer overran its budget twice and
  reported from a partial read, so this is a confirmation pass, not a full one):**
  * **The budget comment understated the dominant cost -- prefills, not just
    generations.**  Every generation in `mask`/`mixer` is preceded by its own prefill
    (`evaluate_samples` -> `greedy_generate` -> `prefill_cache` once per sample per
    configuration), so the unit is "prefill + decode": ~4.35k passes, of which ~3.87k
    are the 4-16K ablations and only 480 are detect's 1K-49K.  The config now says that,
    and names `--random-trials 5 -> 3` (~990 passes, ~23%) as the knob instead of the
    mixer ablation (810 at most).
  * **The hybrid's last K point is degenerate by construction, and the decision rule is
    now in the config.**  The random arm comes from the ≤0.1 pool (11-12 heads under
    `haystack` per the CPU probe), `--k-frac 0.33` resolves to K=16 on 48 heads, and
    `matched_k` caps both arms at the pool: that point is one deterministic intervention,
    not a curve point.  Read `retrieval_pool_by_domain` in the preflight; if the pool is
    under 12, drop `0.33` for that model or mark the `control_exhausted` points as one
    intervention.  (Already true of the committed tree -- the artifact records
    `k_effective`, `control_exhausted`, `random_distinct`.)
  * **The score-matrix memory numbers mixed two bases** (a bf16 figure inside a paragraph
    about the fp32 fallback).  Corrected to the fp32 fallback with Q-heads: 8192 x 49152
    x 4 B = 1.61 GB per head, so ~12.9 GB for the hybrid's 8 and ~25.8 GB for the dense
    model's 16 -- inside 80 GB either way, outside the 22 GiB L4 that OOM'd.  Same fix in
    `a100.yaml`, the `SCALES` comment and `run-in-datasphere.md`.
  * **`--limit` warned "debugging sample" even when it cut nothing** -- the preflights
    pass `--limit 60` for a grid of exactly 60, so the one cheap geometry measurement
    looked biased.  It warns only on an actual truncation now, with a test.
  * **Smaller:** `a100.yaml`'s footer still promised `qa,cot` in the resume tail (they
    are gone from both A100 configs); `plotting.py`'s module docstring still said
    case-study "is not part of the stage list"; the README now says the case-study PDF
    lands in `<model>/figures/` (the JSON would otherwise collide between models); and
    `a100-notemplate.yaml` got the V100 note it was missing, so the controlled pair
    differs only in the geometry and the prefix.
  * The reviewer's own §4 confirms the parts I had already verified: scoring against the
    paper's criteria, both pairings covering exactly the generated stream, the three
    domains as a reporting decision, the held-out eval needles, matched arms, the
    bias-free-`o_proj` caveat, the grid arithmetic (270/210, 45/point, 1665/1395/810),
    the driver's override wiring, dtype isolation, and artifact atomicity.
* **Review round (the eleventh report, pre-A100): one finding survived checking, four did
  not -- including its headline blocker.**  Recorded in this order because the refutations
  are the part that is easy to lose.
  * **The claimed `TypeError` in `case_study.py` does not exist.**  The report's blocker
    was `add_provenance(dtype=args.dtype, payload={...})` against
    `def add_provenance(payload, *, dtype=None, extra=None)`, on the theory that
    `payload` is positional-only.  It is positional-**or**-keyword: the `*` marks only
    `dtype`/`extra` as keyword-only, and a positional-only parameter would need a `/`.
    Verified by calling the real function with exactly that shape (returns the payload
    with `schema_version`/`provenance`) and by re-reading the AST (`args=['payload']`,
    `posonlyargs=[]`).  The test stub (`lambda payload, **k: payload`) is
    signature-compatible with the real call for the same reason, so the report's
    "the test accepts the wrong call" inference also fails.  No stage was ever dead.
  * **The README's masking table is right, and the report read the wrong field.**  It
    compared the table's per-trial **recall** cells against the top-level
    `random_trials`, which is the per-trial **f1** series by construction
    (`trials.append(m.f1)`; `metric: "f1"` is the primary series and `random_mean` is
    its mean).  Checked every cell against `per_sample[k]["random"][trial]`: dense K=9
    recall `95.7, 90.9, 100.0` and exact `6/10, 3/10, 10/10`, K=18 `93.9, 87.0, 0.0`,
    K=36 `85.7, 100.0, 98.7`, hybrid K=1 `39.1, 39.1, 49.6`, K=8 `48.3, 68.3, 71.7`,
    K=16 `0.0, 0.0, 0.0` -- all exact.  Its claim that `48.7` matches nothing also
    failed: `baseline_recall` is 48.695; 46.4 is the *f1*, and the report swapped them.
    The finding is real as a *trap*, though, so it is now closed twice: the
    `random_trials` field carries a comment saying what it is and where per-trial
    recall lives, the README table says which field each column comes from, and
    `test_the_readme_masking_table_matches_the_committed_artifacts` re-derives every
    cell from `ds-results/` so a stale table fails in CI.
  * **The A100 grid arithmetic is right (210 + 270 = 480); the report's 240 was off by
    one length.**  `--lengths` has nine values, `within_context_limit` drops those
    above `40960 - 64 = 40896`, and *both* 40960 and 49152 go -- leaving seven, not
    eight: `7 x 10 x 3 = 210` for the dense model and `9 x 10 x 3 = 270` for the hybrid.
    The one real inaccuracy in the region was wording: "all 270 prompts are built" in
    the preflight bullet is the *hybrid's* count, so `a100.yaml` and the `SCALES`
    comment now say 270/210 per model.
  * **`prefill_impl`/`capture_impl` are recorded -- in `meta.config`, not in
    `provenance()`.**  `provenance()` describes the *environment* (versions,
    `deterministic`, `optional_kernels`, `dtype`, `code_sha256`); the run's conditions
    live in `scores.meta["config"]`, and the committed schema-5 tree already carries
    `prefill_impl: sdpa`, `capture_impl: eager`, `capture_method: patch` there.  So
    `sdpa`-prefill is distinguishable from `eager`-prefill from the artifact, and the
    asymmetry the report saw is the intended split.
  * **The one finding that survived: `mask --pairing same_step` mislabelled its own
    provenance.**  `masking_curve.json` carried `pairing: scores.pairing` (correct,
    from `masking.py`) *next to* `detection_pairing: scores.meta.config.pairing`, and
    the config block records the detect *run*'s primary pairing -- so a `same_step`
    ablation wrote `detection_pairing: next_step` while the heads came from the
    `same_step` matrix (the two top-10 sets share 0/10 heads on both models).  Fixed:
    `DetectionSettings.pairing` is now the loaded matrix's own pairing,
    `resolve_detection_settings` logs when it differs from the run's primary and warns
    if the file does not hold the pairing `--pairing` asked for, and `cmd_mask` writes
    `settings.pairing`.  `test_ablation_labels_the_pairing_of_the_matrix_it_loaded`
    pins all three behaviours.
  * **`--verify-hashes`: the one hardening the report asked for that is worth its
    cost.**  `verify_weights` checked presence and non-zero size only, while
    `configs/models.json` pins a SHA-256 for every file.  Hashing ~3.3 GB is a few
    seconds against 542.88 RUB/h (measured: 4.0 s for both checkpoints on this machine,
    and the real tree passes), so the driver gained the flag, `verify_weights(...,
    hashes=True)` streams each digest and refuses a mismatch (with the missing-file
    path unchanged), and all five A100 configs -- the two preflights included, so a
    corrupt shard fails in the *cheap* run -- pass it.  The L4 scales keep the
    presence-only check, which is what `download_models.sh` already verified before
    upload.  Two tests: the driver's behaviour on a tampered shard, and the flag's
    presence on every `a100*` config.
  * **Declined, with the reason: `inspect.signature` on the patched
    `eager_attention_forward`.**  The pin is `transformers>=5.18,<5.19` and the wrapper
    calls the original positionally with the six documented arguments, so a renamed
    parameter cannot break it and a reordered one would not be caught by a name check
    either.  The failure modes that matter are already loud (head-count mismatch is
    refused at both capture paths; "captured nothing" names the kernel).  Also refuted
    while checking: `_patch_targets` caches by `id(namespace)` but the dict *values*
    hold the namespace objects, so the ids cannot be recycled while the cache lives.
  * Process note: this round was run with the reviewer's report already in hand rather
    than by a subagent, per the instruction to wind the subagents down; no second
    reviewer was launched, so this is a confirmation pass on one report, not a new
    adversarial read.
* **Review round (the ninth report, pre-A100), verified against the code:**
  * **The launch geometry was the real finding, and it is now a controlled pair.**
    `a100.yaml` pinned `--argmax-domain haystack` *with* the chat template, i.e. the
    geometry the project itself calls inflated.  The reviewer is right, and the fix is
    not to pick silently: `configs/datasphere/a100-notemplate.yaml` is the same grid
    with `--no-chat-template` (the paper's geometry), the two configs carry the measured
    probe table and say what each is comparable with, and the README's launch advice is
    to run both or to state which and why.  `a100.yaml` now also says it is the run
    comparable with the committed tree.
  * **The flash-backend probe runs in every job now.**  It lived only in the bootstrap
    job (an L4), while the chunked-prefill memory argument depends on flash being
    available on the card that actually runs; `report_environment()` prints it for the
    card it got, before the grid.  The reviewer's other half -- that the preflight's
    1024/4096 lengths never touch the 49K/8192 regime -- is what step 2 of the launch
    sequence is for (the preflight is a decision aid, not a memory soak test).
  * **`qa`/`cot` are out of the A100 stage list.**  Two reviewers flagged 8+8
    hand-written items on 542.88 RUB/hour: a pipeline check the L4 already exercises,
    and one item is 12.5 points so it cannot support Sec. 5.3 either way.  A real
    measurement needs `--data file.jsonl`; the config comments say so and
    `a100-resume.yaml` mirrors the new tail (`compare,figures,case-study`).
  * **The budget is stated as arithmetic, not adjectives.**  `a100.yaml` now carries the
    generation counts (detect 270/model; mask 1665 dense / 1395 hybrid; mixer 810
    hybrid ~ 3.9k generations of 4-16K plus 480 prefills) and names `--random-trials 5`
    as the dominant multiplier and the mixer ablation as the first thing to cut.  The
    reviewer's own 2475 omitted the hybrid's mask arm; the counts are now checkable in
    the config rather than estimated in prose.
  * **`run-in-datasphere.md` was three rounds stale** -- "3 needles x 5 depths = 15
    samples" (it is 3 lengths x 5 x 3 = 45), "the CLI default is 3x5x1 = 15" (it is
    1x5x1 = 5), and the pre-round-eight truncation numbers (0.684/0.793 -> 0.671/0.755).
    Fixed, plus a new checklist step: preflight both geometries *before* paying for the
    grid.
  * **Smaller:** `summarize_results.py` prints `retrieval_pool_by_domain` (the number
    the preflight configs point at); the `case-study` stage runs at `--length 4096
    --max-new-tokens 96` instead of the script's 1024/32 defaults, so Fig. 1 uses the
    run's generation regime; the README states that `figures`/`compare` draw the
    primary domain only (the per-domain comparison is in the summary and the notes) and
    that the truncation counters mean "hit the budget", not "wanted to continue".
  * **Process note, and my mistake:** the reviewer saw the repo move under it
    (`5caa54f` -> `34646cd` plus an uncommitted test file) because I kept working while
    it read.  Future rounds: commit and stop touching the tree while a reviewer runs.
* **Review round (the eighth report, pre-A100), verified against the code:**
  * **`mask` on the A100 ran at 32 tokens while `detect` ran at 96 -- fixed.**  The
    reviewer is right that the stage carrying the causal claim was generating with the
    CLI default, *tighter* than the 48 the committed t4 tree used, and the hybrid's
    answers were being cut mid-sentence (`masking_curve.json`'s `per_sample[...]
    ["retrieval"]["generated_texts"]`).  `SCALES['a100']['mask']` now pins
    `--max-new-tokens 96`; `qa` deliberately keeps 24 (its metric is token-F1 over the
    whole completion, so narration costs precision rather than buying recall) and `cot`
    stays 256.  Three budgets in one tree is a measurement condition, not a bug, and
    every artifact records `max_new_tokens` -- now said in the README's caveats, since
    the A100 numbers are not comparable with the committed 48-token tree even in the
    same argmax domain.
  * **The domain bound IS a theorem, and my own test asserted the opposite.**  The
    reviewer argued a bf16 tie could lose credit when moving `prompt` -> `haystack`.
    That cannot happen: the haystack span is a subset of the prompt and contains the
    needle, so if the prompt argmax is a needle position it is also the *first* maximum
    inside the span (an earlier tied position inside the span would have been the prompt
    argmax instead).  So credit can only be gained, ties included, and the reverse fails
    (a template token can win the prompt argmax).  But the report did find a real defect:
    `test_detect_writes_a_matrix_per_argmax_domain` asserted the *wrong* direction
    (`haystack <= prompt`) and passed only because the tiny model earns no credit at all
    -- verified by printing the matrices, all zero.  The vacuous assertion is replaced by
    an explicit all-zero guard, the theorem now has a property test on quantised rows
    (ties included) in `test_scoring.py`, and the integration test's docstring carries
    the argument so it is not re-raised as an empirical claim.
  * **`prefill_cache` now passes `logits_to_keep=1`.**  The reviewer is right that the
    chunk-memory arithmetic in `a100.yaml` described the fp32-fallback attention matrix,
    which the bf16 run never materialises: the binding chunk-sized allocation was
    `lm_head` over every position of every chunk (~4.1 GB at 8192 x 248320 for the
    hybrid, ~2.6 GB for the dense model) plus a matmul of the same order as the whole
    chunk's transformer work, per chunk.  The returned value is unchanged
    (`out.logits[:, -1, :]`), and the chunked-prefill equivalence tests still pass.  The
    config comment now names both peaks.
  * **`--no-chat-template` is a driver flag, and there is a template-free preflight.**
    The sink geometry is a property of the prompt, so it cannot be derived from a
    template-on run, and choosing it used to mean editing `a100.yaml` -- exactly what
    `docs/datasphere-findings.md` section 16 warns against.
    `configs/datasphere/a100-preflight-notemplate.yaml` is the twin of the template-on
    preflight, and the flag reaches only the stages that render a prompt.
  * **The cited truncation numbers did not reproduce -- replaced.**  `a100.yaml` quoted
    "0.684 against 0.793 on the top head"; recomputed from
    `ds-results/qwen3.5-0.8b/instances_next_step.jsonl` split on `meta.truncated` it is
    **0.671** (11 truncated) against **0.755** (64), and the old pair appears in none of
    the three committed trees.  The comment now cites the artifact it came from.
  * **Smaller, all verified:** `NeedleSample.haystack_tokens` (the *prompt* length, next
    to `n_haystack_tokens`, the *context span*) is now `prompt_tokens`, which is what the
    artifact already called it; two `--k-frac` values that collapse into one K are
    logged and both the request and the resolution are recorded
    (`k_frac_args`/`k_args`); the pie figure's title takes its threshold label from the
    panels instead of from whichever model came last; a domain missing on *some*
    instances (a `prompt`-domain run where a prompt is not verbatim) is aggregated over
    the subset that has it, with `n_instances_without_domain` recorded, instead of a
    `KeyError` mid-grid; `summary_*.json` now carries `argmax_domains_captured`;
    `aggregate_scores(domain=primary)` raises a message naming the complement instead of
    a bare `KeyError`; and `docs/datasphere-findings.md`'s pointer to the deleted
    `test_regressions.py` now points at `test_masking_regressions.py`.
  * Left as the reviewer's *decision*, not a code change: the hybrid's random-arm pool
    shrinks as the `>0.1` share grows (33/48 heads above 0.1 under `prompt` leaves 15),
    so under `haystack` the `mask` curve may degenerate.  That is what the two preflights
    are for; the artifact already records `control_exhausted`,
    `random_control_contaminated`, `k_effective` and `random_retrieval_overlap`.  Then
    *measured* it offline instead of leaving it to the preflight (3 instances per model,
    512 tokens, 96-token budget, CPU fp32, `detect --preflight` into `.tmp/`, with the
    chat template and without it):

    | model | template | `haystack` >0.1 | `prompt` >0.1 | sink in `x` | recall | pool |
    |---|---|---|---|---|---|---|
    | Qwen3-0.6B | yes | 170/448 (38%) | 34/448 (8%) | no | 1.00 | 278 |
    | Qwen3-0.6B | no | 26/448 (6%) | 18/448 (4%) | yes | 0.55 | 422 |
    | Qwen3.5-0.8B | yes | 37/48 (77%) | 35/48 (73%) | no | 1.00 | 11 |
    | Qwen3.5-0.8B | no | 36/48 (75%) | 35/48 (73%) | yes | 1.00 | 12 |

    So: the dense model's 38% is a *template artifact* (6% in the paper's geometry,
    inside its 3-6% band -- but there the same model recovers only 0.55 of the needle,
    so that number is measured on a partly-failing model); the hybrid is
    template-insensitive at 75-77% with recall 1.00 either way, i.e. its non-sparsity
    is architectural; the pool shrinks (13 -> 11) but does not vanish, and everything
    from K=11 is the whole pool.  The honest launch plan is therefore **both**
    geometries (`a100-preflight.yaml` + `a100-preflight-notemplate.yaml`), with the
    paper-comparable share and the task-success drop both reported.  The ordering
    `haystack` ⊇ `prompt` ⊇ `full` is pinned as a theorem in
    `test_scoring.py::test_the_three_domains_are_ordered_by_credit` (40 quantised rows,
    ties included), and `summary_*.json` carries `retrieval_pool_by_domain` so the
    preflight answers the pool question without arithmetic.
  * Also left deliberately: `qa`/`cot` stay on the A100 (minutes against hours, and they
    complete the tree), and the dense model's two longest lengths stay dropped (its
    40960-token window is a model property; changing the grid would break "same grid as
    `paper`").
* **Review round (the seventh report: the pre-A100 review), verified against the code:**
  * **B1 -- the argmax domain is now a *reporting* dimension, not a run-level choice.**
    The reviewer's main point survived checking: `a100.yaml` pinned `haystack`, which
    had never been measured at any scale, and the run's headline ("a few percent of
    heads") is a property of the position set as much as of the model.  Rather than
    gamble the grid, `decode_with_attention` now captures **every** domain from the
    same row (`StepTrace.argmax_by_domain`; no extra forward pass), `credits_from_trace`
    takes `domain=`, and `score_instance` scores all of them: `scores` stays the
    run's own domain (historical filenames, and what the ablations rank heads by) while
    `scores_by_domain` holds the complement, `run_detection` aggregates each into
    `scores_<pairing>_<domain>.npz|.json`, and the summary carries
    `sparsity_by_domain`, `top_heads_by_domain` and `domain_ranking_overlap` (how many
    of the primary domain's top-10 survive under each other domain -- i.e. how much a
    domain change would change what `mask` masks).  `SCHEMA_VERSION` is 8.
    The reviewer's cost estimate was wrong in mechanism and size: the JSONL does not
    carry per-step argmax positions at all, so the growth is one extra per-head score
    vector per alternative domain (~+30 KB/instance), not "20-40 MB per model" from
    duplicating an argmax map.
  * **It does not remove the *template* axis, and the config now says so.**  Whether
    position 0 is inside `x` is decided by the prompt, so `haystack`-with-template and
    `haystack`-without are different measurements, not two readings of one.  The
    preflight below is what prices that axis.
  * **B2 -- the ablation sample count was wrong by 3x in three places.**  The count is
    `lengths x depths x needles`; the README, the `a100` scale comment and a test all
    multiplied only the last two, so the A100's 3x5x3 = **45** samples per point was
    called "15" (and the t4 run's 2x5x1 = 10 was called 5).  Fixed in all three, the
    assertion now computes the product from the flag list, and the artifact records
    `lengths` / `n_samples_per_point` (`mixer_ablation.json` too) -- the two trees were
    previously indistinguishable on the axis that dominates the stage's cost.
  * **B3 -- `task_qa.json` / `task_cot.json` now record `data_path` (and
    `dataset: builtin|file`).**  Without it a real-dataset artifact differed from the
    built-in stand-ins only in `n_samples`.  The reviewer's other suggestion there
    (don't spend A100 time on 8 hand-written items) is a judgement call: the stage is
    minutes against hours for `mask`, and it completes the tree, so it stays.
  * **B5 -- the config's own preflight advice is now executable.**  The driver gained
    `--lengths` (replaces each stage's values) and `--limit` (detect only), so the
    "start at a short length" line in `a100.yaml` is a command line instead of an edit
    to `SCALES` that must be reverted; `configs/datasphere/a100-preflight.yaml` runs
    `describe,detect` at 1024/4096 with `--limit 60`, and `a100-resume.yaml` is the
    A100 counterpart of `t4-resume.yaml` (staging tree `ds-a100-resume/`, gitignored)
    so a late-stage failure does not re-pay for the 45-sample `mask` stage.
  * **B5, second half -- `detect --preflight` builds every prompt before the first
    forward pass.**  `score_instance` refuses a sample whose `haystack` span is
    missing (a prompt that does not contain the context verbatim), and without the
    flag that refusal aborts `detect` mid-grid, after the GPU time already spent.  The
    flag builds all planned prompts (CPU only), validates them, and keeps them so the
    loop does not rebuild; `SCALES['a100']['detect']` turns it on and the L4 scales
    leave it off (there a mid-grid failure costs minutes).  Two tests pin the
    ordering: with the flag the order is build, build, score, score; without it,
    build, score, build, score.
  * **B4 -- Fig. 1 is a stage now.**  `case-study` is the one driver stage that runs a
    standalone script: `run_cli` dispatches it to `scripts/case_study.py` via `runpy`
    **in-process**, so it reuses the resident model (the script imports
    `retrieval_heads.cli._load`, which keeps one model) and costs one 1024-token
    instance rather than a second GPU session.  Its `--scores <model dir>` reuses the
    run's recorded conditions, `--out <model dir>` keeps two models' JSON apart, and
    the script's "no copy step found" exit (1) only warns, because aborting the last
    stage would discard a finished run.  `paper.yaml`, `t4-cached.yaml` and `a100.yaml`
    list the stage.
  * **Minor 1 -- `argmax_domain_shift` now counts the rows the score counts.**
    It takes `sample`/`pairing` and applies the same `applies_to` + criterion-(1)
    filter as `credits_from_trace`, so the recorded share describes the scored rows
    (the artifact says `restricted_to`); called without them it still counts every
    captured row, which is the capture-level property.  It is recorded **per pairing**
    (the two pairings score different token sets), and `summary_same_step.json` /
    `scores_same_step_raw.json` carry their own sum instead of a copy of the primary
    pairing's.
  * **Minor 6 -- the hybrid pie caption states both head bases**
    (`48 scoreable heads / 18 linear layers (336 heads in all)`), so the README's
    "2.7% of 336" and "68.8% of 48" cannot be read against the wrong denominator.
  * **Minor 7 -- `mean_sink_rate` is documented as a share of scored *(head, step)*
    pairs**, not of steps (`considered` is identical across a layer's heads, so the two
    readings differ whenever some heads point elsewhere).
  * **Minor 3 did NOT survive checking: `HeadMasker`'s clones do not accumulate.**
    The reviewer claimed the masking peak grows as `seq x hidden x #masked_layers`
    (~1 GB at K=148/16K); the clone is created inside the `o_proj` pre-hook, consumed
    by that layer's `o_proj` call and released before the next layer runs, so the peak
    overhead is one `(seq, hidden)` tensor.  Measured (one process per configuration,
    `ru_maxrss` of one forward on a synthetic 8-layer stack at 8192 tokens): 405.6 MB
    with one masked layer against 405.8 MB with eight at `hidden=4096`, and 106.8 vs
    107.1 MB at `hidden=1024`.  The README carried the same wrong sentence and is
    fixed; what grows is the ablation's *compute*, not its peak.
  * **Minor 2 (bf16 argmax ties) is now stated** in the README's limitations: the
    argmax runs over probabilities in the model's dtype, so near-ties can move
    criterion (2) without the model changing, and `argmax_domain_shift` compares
    domains rather than dtypes (that comparison still needs one GPU run).
  * **Minor 4 (the dense control cannot see 40K/49K) is left as-is, deliberately.**
    Adding 40000 would give it one more long point but would also break the property
    the config rests on ("the A100 grid is `paper`'s grid"), and the dense model's
    40960-token window is a model property, already documented.  The reviewer's
    suggestion is a real limitation, not a defect to patch quietly.
  * Left as judgement calls for the launch: whether to run `--no-chat-template` (the
    paper's geometry, worse needle recall on the probed instance), and whether to trim
    the A100 stage list to skip the 8-item `qa`/`cot` pipeline check.
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
re-run of the current code (which now defaults to the `haystack` argmax domain and
writes a matrix per captured domain, so a re-run's detection numbers are expected to
move and its tree will carry both readings). Details in §6.

Branch: **`Nikita-prog-art`**. `main` on the remote is untouched. See
`git log --oneline -1` for HEAD; the working tree holds the round-seven changes
(committed, not pushed).

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

### Launch sequence for the A100 (the one thing that needs a decision)

1. **Preflight, both geometries** (minutes each, `describe,detect` at 1024/4096 with
   `--limit 60`): `configs/datasphere/a100-preflight.yaml` (chat template) and
   `a100-preflight-notemplate.yaml` (paper geometry).  Launch them as two jobs.  Both
   pass `--verify-hashes`, so a corrupt checkpoint on the project disk fails here in
   minutes instead of in the full grid.
2. **Read four numbers per model** from each `summary_next_step.json`:
   `sparsity_by_domain`, `retrieval_pool_by_domain`, `sink_in_haystack`,
   `n_instances_recited`/`mean_needle_recall`.  The CPU probe already predicts them
   (README's sink-geometry bullet: dense 38% with the template vs 6% without, hybrid
   75-77% either way, pool 11-12 for the hybrid) -- the preflight's job is to confirm
   them at 1024/4096 tokens, where the context is longer than the probe's 512.  Its log
   now also prints the SDPA flash-backend probe for the card it actually got.
3. **Decide the geometry.**  Both are defensible and they answer different questions:
   with the template the models solve the task (recall 1.00) and the dense model's
   share is inflated by the sink; without it the dense share lands in the paper's
   3-6% band but the dense model recovers only ~0.55 of the needle.  Reporting both
   is the honest option; if only one is run, say which and why in the results doc.
4. **Launch the full grid**: `a100.yaml` (chat template, comparable with the committed
   `ds-results/` tree) or `a100-notemplate.yaml` (the paper's geometry) -- identical
   grids, so they are a controlled pair.  Both pin `--argmax-domain haystack`,
   `--max-new-tokens 96` for detect *and* mask, `--preflight`, `--verify-hashes`, and run
   `describe,detect,mask,compare,figures,case-study`; `qa`/`cot` are deliberately
   *not* in the A100 list (8+8 hand-written items are a pipeline check, not A100
   time -- a real measurement needs `--data file.jsonl`, and the cheap grid already
   exercises the stage).
5. **If a late stage fails**, stage the downloaded tree as `ds-a100-resume/` and launch
   `a100-resume.yaml` -- it re-runs only `compare,figures,case-study`, because `mask`
   is the stage that costs real money (45 samples per point x 6 K x 6 arms ~ 1.4-1.7k
   generations per model, plus the hybrid's 810 mixer generations).

Each item below says what to do, not just what is missing.  Everything that could be
finished offline is done, including round six (the `haystack` argmax domain and its
default, numeric answer comparison, three held-out eval needles, per-sample QA
spread, the raw-denominator matrices, the small guards, the aliased-`eager` patch,
the reproduce-script parity check, and the test-file split), round seven (the
argmax domain as a reporting dimension with a matrix per domain, the ablation
sample-count corrections and the `lengths` provenance, `data_path` on the QA/CoT
artifacts, the driver's `--lengths`/`--limit` preflight overrides plus the
`a100-preflight`/`a100-resume` configs, `case-study` as a stage, and the
`argmax_domain_shift`/pie-caption/sink-wording/bf16 notes) and round eight (the mask
generation budget, `logits_to_keep`, the `--no-chat-template` driver flag and the
template-free preflight, the recomputed truncation numbers, the subset-domain
aggregation, the K-collapse warning, `prompt_tokens`, the pie threshold label, and
the domain-ordering theorem with its property test).  Rounds nine through eleven are
summarised as their own entries above: the controlled geometry pair and the flash
probe, the A100 budget arithmetic and the degenerate-pool rule, the
`detection_pairing` provenance fix, `--verify-hashes`, and the README/artifact
tripwire.  What is left needs a GPU run, external data, or a judgement call.

Known measurement limits, in the artifacts themselves rather than hidden:
`mask` now keeps per-sample F1 and reports the retrieval arm's spread, and its
default is 5 samples per point -- still not a statistically powered curve, which is
what the paper-scale grid is for.  K in the retrieval arm is capped by the control
pool size; the artifact records both requested `k_values` and realized
`k_effective`, and `summarize_results.py` prints "K →k_eff".  `scripts/case_study.py`
(Fig. 1) is the `case-study` driver stage now (run in-process, so it reuses the
resident model) and can still be run by hand.
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
  slice assign allocates the same bytes, so there is no cheap win here.  The clone
  dies with its own layer's `o_proj` call, so the *peak* does **not** grow with the
  number of masked layers -- it is one `(seq, hidden)` tensor (measured: 405.6 MB at
  one masked layer against 405.8 MB at eight, synthetic 8-layer stack, 8192 tokens,
  `hidden=4096`).  What grows is the ablation's compute (contexts x K x trials).

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
   `configs/datasphere/a100.yaml` is the ready-made run, `a100-preflight.yaml` is the
   cheap first step, and the driver now takes `--lengths`/`--limit` so neither needs a
   `SCALES` edit. Lengths past a model's `max_position_embeddings` are dropped with a
   warning.
3. **Real datasets.** `--data file.jsonl` on `qa`/`cot`:
   `{"context","question","answer"}` and `{"question","answer"}`. The built-ins are
   calibrated stand-ins, not benchmarks; the artifact now records `data_path`/`dataset`,
   so a real-dataset run is distinguishable from them.
4. **Paul Graham haystack.** The canonical NIAH haystack corpus is at
   `source/PaulGrahamEssays/*.txt`; `--corpus <file>` uses it on `detect` and now
   also on `mask` (so detection and the causal experiment share a haystack, and the
   artifact records which one). **Do not read the rest of `source/`** — the user
   explicitly said the original authors' code is not to be used (§7).
5. **A run under the new `haystack` default.**  Everything that was "stored but not
   reported" is reported now: both pairings have per-head tables, `same_step` has its
   own `scores_same_step.*` / `summary_same_step.json`, `aligned_top_heads` is in the
   summary and printed by `summarize_results.py`, the raw-denominator matrices are
   emitted, and **every captured argmax domain is scored and written**
   (`scores_<pairing>_<domain>.*` + `sparsity_by_domain`/`domain_ranking_overlap`), so
   the run answers the domain question instead of pinning it.  What is pending is the
   *numbers* from a run that uses the current default (`haystack`), since the committed
   tree is `prompt`-domain -- i.e. the A100 job.  The one thing the domain matrices do
   **not** settle is the chat template: whether position 0 is inside `x` is a property
   of the prompt, so `--no-chat-template` is a separate measurement (the preflight
   config is where to price it).
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
.venv/bin/python -m pytest -q                     # 281 passed (263 fast + 18 integration)
.venv/bin/python -m retrieval_heads.cli describe --model qwen3.5-0.8b
# -> 6 scoreable layers [3,7,11,15,19,23], 48 scoreable heads, hybrid: True
.venv/bin/python -m retrieval_heads.cli describe --model qwen3-0.6b
# -> 28 scoreable layers, 448 scoreable heads, hybrid: False
```

If `describe` reports 24 scoreable layers for Qwen3.5, the architecture-aware head
discovery has regressed and every retrieval score after that is meaningless.
