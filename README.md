# Retrieval Heads — re-implementation

Code for **“Retrieval Head Mechanistically Explains Long-Context Factuality”**
(Wu, Wang, Xiao, Peng, Fu) — a from-scratch implementation of the paper's method,
written to run on **Qwen3.5-0.8B** and verified against it.

```bibtex
@article{wu2024retrieval,
  title  = {Retrieval Head Mechanistically Explains Long-Context Factuality},
  author = {Wu, Wenhao and Wang, Yizhong and Xiao, Guangxuan and Peng, Hao and Fu, Yao},
  year   = {2024}
}
```

---

## The one thing to know before reading further

**Qwen3.5-0.8B is not a plain transformer.** Its `config.json` describes a
*hybrid* stack:

| | count | what it is | has an attention map? |
|---|---|---|---|
| Gated DeltaNet | 18 layers | linear attention (recurrent state) | **no** |
| Gated Attention | 6 layers | full softmax attention | **yes** |

The full-attention layers are `3, 7, 11, 15, 19, 23`; each has 8 query heads,
2 KV heads and `head_dim = 256`. The paper's retrieval score is
`|g_h ∩ k| / |k|` where `g_h` is built from `argmax` over a head's attention
distribution — a quantity that **does not exist** for a Gated DeltaNet layer.
So exactly **6 × 8 = 48 heads** are scoreable, out of 24 layers.

This is not a workaround; it is the honest consequence of the architecture, and
it is interesting precisely because the paper's own discussion (Sec. 5) claims
that *full attention is a must for long-context retrieval*. Qwen3.5 passes
262K-context needle tests with only six full-attention layers, which puts that
claim to a real test.

The code discovers this by introspection rather than hard-coding it
(`retrieval_heads/models.py`), and **Qwen3-0.6B** is included as a dense control
from the same lab — 28 layers × 16 heads = 448 scoreable heads — so the hybrid
numbers can be read against a normal transformer.

---

## Install

```bash
# torch (CPU wheel; drop the index-url for a CUDA build)
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'transformers>=5.18,<5.19' numpy matplotlib tqdm
uv pip install --python .venv/bin/python pytest          # for the test suite

# ~1.7 GB + ~1.5 GB of weights, checksum-verified
./scripts/download_models.sh
```

Requires `transformers >= 5.18,<5.19` — Qwen3.5 (`model_type: qwen3_5`) is not in
older releases, and the code reaches into v5 internals (patched
`eager_attention_forward`, `_attn_implementation`, the `dtype=` kwarg), so a
major bump is not assumed to work. The DataSphere lock pins `5.18.0`.

> **Note on the weight filenames.** Qwen3.5-0.8B's index points at a single shard
> literally named `model.safetensors-00001-of-00001.safetensors`; there is no
> `model.safetensors`. Fetching the obvious name returns a 15-byte
> `Entry not found` body that loads as a corrupt checkpoint. `download_models.sh`
> takes the real shard names from the `shards` block of `configs/models.json` (the
> registry is the single source of truth; the index file is not parsed), and every
> file is pinned by SHA-256.

---

## Quick start

```bash
.venv/bin/python -m retrieval_heads.cli describe --model qwen3.5-0.8b
.venv/bin/python -m retrieval_heads.cli detect   --model qwen3.5-0.8b --profile laptop  # writes the .npz
.venv/bin/python -m retrieval_heads.cli mask     --model qwen3.5-0.8b --k-frac 0.02 0.04 0.08 0.17 0.33
.venv/bin/python -m retrieval_heads.cli figures  --runs results/qwen3.5-0.8b
```

Or the whole thing:

```bash
./scripts/reproduce_laptop.sh              # CPU, well under an hour
./scripts/reproduce_gpu.sh qwen3.5-0.8b    # paper-scale grid
pytest -m "not integration"                # fast unit suite
pytest                                     # + real-checkpoint tests
```

**CI coverage.** `.github/workflows/tests.yml` runs only the fast suite: the
`integration` tests load 3.2 GB of checkpoints, which is not viable on every push.
They are the ones that actually exercise architecture discovery, attention
capture (both paths), chunked-prefill equivalence and the head-mask identity, so
run `pytest` locally after touching `models.py`, `attention.py` or `scoring.py`.

### On a Yandex DataSphere GPU job

```bash
export PATH="$HOME/yandex-cloud/bin:$PATH"
source scripts/datasphere_auth.sh                 # -> YC_IAM_TOKEN
CLI=.venv-datasphere/bin/datasphere
PROJECT=bt1u5v72b71eesdhp9k5

# once: build a persistent venv on the project disk and validate every stage
$CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-bootstrap.yaml

# then, with ~40 s startup instead of ~9 min.  Run it blocking *in the background*:
# that is the only mode that streams progress, and the job itself lives on the
# service, so losing the laptop connection does not stop it.
mkdir -p logs
nohup $CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-cached.yaml \
    > "logs/ds_$(date +%m%d_%H%M).log" 2>&1 &
tail -f logs/ds_*.log
```

`job attach` and the job page show stdout only after completion, so they are not a
way to watch progress; `--async` is for runs where no live stream is needed.
Artifacts and logs come back with
`$CLI project job download-files --id <job_id> --with-logs`.

`configs/datasphere/` also holds `t4.yaml` (cacheless fallback), `t4-resume.yaml`
(only `qa,cot,compare,figures`, reusing a finished run's detect/mask artifacts via
`local-paths`), `laptop.yaml`, `paper.yaml` and `smoke.yaml`.  Everything non-obvious about this path — the pip
crash that shapes the requirements file, the `cmd` grammar, why the cached venv
cannot be the entry point, what the "T4" slot actually hands out — is written up
in [`docs/datasphere-findings.md`](docs/datasphere-findings.md).

---

## Method

### Retrieval score (paper Sec. 3)

During greedy decoding, let `w` be the token being generated and `a` a head's
attention probabilities over the input. The head **copy-pastes** `w` when

1. `w ∈ k` — the token belongs to the needle, and
2. `x_j = w`, `j = argmax(a)`, `j ∈ i_q` — the head's most-attended input
   position holds that very token **and** lies inside the needle.

With `g_h` the set of tokens copied by head `h`:

```
retrieval_score(h) = |g_h ∩ k| / |k|
```

A head qualifies as a *retrieval head* above a threshold of `0.1` (the paper's
choice).

### Two places the paper is underspecified, and what this code does

**The denominator.** `g_h` is a *set*, so `|g_h ∩ k| ≤ |unique(k)|`. Taking `|k|`
to be the raw needle length would cap the score below 1.0 for any needle with a
repeated token. We use the number of **unique** needle tokens — the only
self-consistent reading.

**Which attention row.** “The attention scores of a head” at the step where `w`
is generated can be paired with `w` in two ways. Both come out of the same greedy
decoding pass, so the choice is a reporting decision rather than a compute
decision. (`next_step` does need one extra single-token forward: the row that
*produces* the first generated token belongs to the prefill, not to a decode
step.)

| `pairing` | the row | mechanism |
|---|---|---|
| `next_step` *(default)* | the row whose query **produces** `w` — the head points at the source position it is about to paste | CopyNet-style paste |
| `same_step` | the row at `w`'s **own** query position — the head looks back at the source of the token it just emitted | induction-head-like |

The default is the literal reading of “the current token being generated as `w`”
with the attention at that same step. **The two are not interchangeable** — on
Qwen3.5-0.8B a single instance already ranks different heads top (see Results),
so both are stored in every run and the choice is a reporting decision, never
silently baked in.

Both pairings cover **exactly the generated token stream, first token included**.
That needs one extra forward pass: the last prompt token is fed on its own under
the capture kernel, because its attention row is the one that *produces* the first
generated token, and `next_step` used to be unable to credit that token while
`same_step` always could.  That row is stored with `applies_to=("next_step",)` so
it cannot leak into `same_step`, and it costs `(heads, kv_len)` like any decode
step.  Symmetrically, when decoding stops on `max_new_tokens` the final row
predicted a token that was never emitted, so it is scoped to `same_step` and
`next_step` cannot credit a hypothetical token
(`retrieval_heads/scoring.py`).

### Detection grid

The paper's full recipe is 3 needle sets × 20 lengths in 1K–50K × 10 insertion
depths ≈ 600 instances per model. `DetectionConfig` exposes every axis, with
named profiles:

| profile | grid | use |
|---|---|---|
| `smoke` | 1 × 1 × 1 | CI / sanity |
| `laptop` | 2 × 3 (1K, 2K, 4K) × 3 = 18 | `scripts/reproduce_laptop.sh` (CPU) |
| `paper` | 3 × 7 × 10 | GPU |

The masking and downstream grids used for the reported run are somewhat wider than
the profile table (`t4` in `scripts/datasphere_job.py`: 3 × 5 × 5 = 75 detection
instances per model).

Filler text is generated from a seeded word list (offline, deterministic);
`--corpus FILE` swaps in a real corpus such as the Paul Graham essays.

---

## Code map

| file | paper section | what it does |
|---|---|---|
| `retrieval_heads/models.py` | — | **architecture-aware head discovery**: separates softmax attention from linear/recurrent mixers |
| `retrieval_heads/haystack.py` | 3 | needle insertion at a given depth; token span recovered from character offsets |
| `retrieval_heads/attention.py` | 3, 4.1 | attention capture (`patch` by default, keyed by `layer_idx`; public `output_attentions` as the alternative) and head / token-mixer ablation hooks |
| `retrieval_heads/scoring.py` | 3 | the retrieval score: two criteria, both pairings, dense layer × head matrices |
| `retrieval_heads/detection.py` | 3 | the detection driver and its configurable grid |
| `retrieval_heads/properties.py` | 4 | sparsity buckets, activation-frequency gap, Pearson correlation, head-set overlap |
| `retrieval_heads/masking.py` | 4.1, 5 | top-K vs random-K masking curves; full-attention vs linear layer ablation |
| `retrieval_heads/downstream.py` | 5 | extractive QA and CoT reasoning with and without masking |
| `retrieval_heads/plotting.py` | all figures | `ring_graph`, `score_distribution`, `heat_map`, `corr_map`, `masking_heads`, `task_qa`, `task_cot` |
| `retrieval_heads/cli.py` | — | `describe / detect / mask / qa / cot / compare / figures` |

### Implementation notes worth knowing

* **Prefill cheap, decode precise.** Attention maps are only needed at decoding
  steps, where `q_len = 1`. The prefill therefore runs on `sdpa` and only the
  decode steps on `eager`. A captured row costs `(heads, kv_len)` instead of
  `(heads, seq, seq)`, so *capture* memory stays flat in context length (the
  masking/ablation stage does not: each masked layer clones its `o_proj` input, so
  its peak grows with context length times the number of masked layers) — the difference
  between running at 4K and at 50K on one machine.
* **Masking a head = zeroing its `o_proj` input slice.** A head's attention
  output only ever touches `[h·d : (h+1)·d]` of what enters `o_proj`, so zeroing
  that slice is *exactly* equivalent to zeroing the head's attention row, with no
  surgery on the attention kernel. Hooks are installed and removed strictly
  (`HeadMasker` as a context manager), and the tests assert the logits return to
  baseline bit-for-bit afterwards.
* **Both capture paths are tested for numerical agreement**, so the monkeypatch
  fallback can never quietly disagree with the public API.

---

## Results

**Scope.** What is tested here is the *mechanism* (which heads matter, and what
happens when they are removed), not the paper's quantitative replication: the grid
is 75 instances per model (3 needles x 5 depths x 5 lengths) against ~600, the
models are 0.6B/0.8B against 6-34B across four families, the context is up to 16K
against 1K-50K, and the downstream stage uses 8 hand-written QA items and 8
arithmetic items against MMLU/MuSiQue/GSM8K.  Read the tables as "the mechanism
reproduces", not "the numbers reproduce".

The numbers below come from one GPU run on an NVIDIA L4, in two jobs: detection
and masking in
[job `bt1ethsb4jlpds7m6i32`](https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5/job/bt1ethsb4jlpds7m6i32)
(~31 min, which then failed on `qa`), and the remaining stages in
[job `bt18bmvbuggt68cnel9p`](https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5/job/bt18bmvbuggt68cnel9p)
(~17 min, reusing the first job's artifacts through `configs/datasphere/t4-resume.yaml`).
**75 instances per model** — 3 needles x 5 depths x 5 lengths, 1K-16K.  Full
tables: [`docs/results-gpu.md`](docs/results-gpu.md); regenerate any time with

```bash
.venv/bin/python scripts/summarize_results.py ds-results
```

These artifacts match the committed code (`schema_version` 5): the held-out eval
needle, the prompt-only argmax domain, the needle-text gold tokens and the
corrected filler sizing are all in effect.  Realized prompt lengths land within
~1% of the request (e.g. 1024 -> 1022, 16384 -> 16467 on average; per-instance
values are in `instances_*.jsonl`), against the ~2% tolerance the code enforces.

### Detection

| model | recited | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (dense) | 75/75 | 0.963 | `L16H14` | 0.89 | 28/448 (**6.2%**) | 5/448 (1.1%) |
| Qwen3.5-0.8B (hybrid) | 75/75 | 0.925 | `L11H1` | 0.74 | 33/48 (**68.8%**) | 9/48 (18.8%) |

**The dense model now matches the paper's Fig. 2 shape; the hybrid does not.**
Qwen3-0.6B has 66.1% of its heads *zeroed* and 27.7% weak, both inside the paper's
quoted 45-73% and 25-52% ranges, and 6.2% clear the 0.1 threshold -- marginally
above the paper's 3-6% band.  Qwen3.5-0.8B has 68.8% of its 48 scoreable heads
above 0.1, so retrieval there is not sparse at all.  Two readings, both worth
stating:

* over the heads that *can* retrieve (its 6 full-attention layers), retrieval in
  this architecture is not sparse;
* the grid's extreme depths (needle first / needle last) are included, and with
  only 5 depths they carry 40% of the weight -- against the paper's 10 interior
  points.  `iter_depths` is endpoints-inclusive by design, so depth-robustness and
  this weighting should not be confused;
* over *all* its token-mixer heads (48 attention + 288 Gated DeltaNet = 336 — the
  count is `ModelInfo.n_all_heads`, read from the mixer modules, not typed into
  prose), the 9 strongly-retrieving heads are 2.7% — back inside the paper's
  range, but only by counting objects the retrieval score is not defined for.

The honest statement: the paper's "a few percent" is a property of a dense stack in
which most layers do no retrieval, and it does not transfer to a hybrid stack where
18 of 24 layers cannot retrieve at all and the remaining 6 must carry it.

All 75 instances per model count as recited under the LCS recall, so the
recited-only view (`sparsity_recited`) equals the unconditional one here; the
threshold is deliberately soft (0.3 of the needle's words, in order).  The hybrid
does hit `max_new_tokens` on 11 of its 75 instances (detection runs a 48-token
budget against a ~26-token needle), which is why the run reports
`n_instances_truncated`.

### The paper's definition is ambiguous, and the ambiguity decides the answer

The retrieval score pairs a head's attention row with "the token being generated",
which can mean the row that *produces* the token (`next_step`) or the row at the
token's *own* position (`same_step`).  On **both** models the top-10 heads under the
two readings overlap **0/10**:

| model | `next_step` top heads | `same_step` top heads |
|---|---|---|
| Qwen3.5-0.8B | L11H1, L15H7, L19H5, L23H1, L23H5 | L7H7, L11H3, L3H7, L23H6, L23H2 |
| Qwen3-0.6B | L16H14, L21H8, L20H14, L6H11, L18H5 | L6H6, L11H2, L6H7, L2H10, L1H15 |

The pattern is systematic, not noise: `next_step` selects **late** layers (where the
copy is emitted into the residual stream), `same_step` selects **early** layers (the
induction-head position, where the retrieved token is staged for later use).  Both
come out of the same decoding pass, so this costs nothing to report — and no
reproduction should quote one without the other.

### Masking: the causal claim holds from ~8% of heads, not at 2%

Needle-in-a-Haystack, retrieval heads vs random heads.  The per-trial numbers below
are the ones that matter; a mean over three trials hides how erratic the control is.

| model | baseline | K (share of heads) | retrieval | random trials |
|---|---|---|---|---|
| Qwen3-0.6B | 94.3 f1 / 90% exact / 98.7 recall | 9 (2%) | 93.0 / 50% / 96.1 | 95.7, 90.9, **100.0** recall; 6/10, 3/10, **10/10** exact |
| | | 18 (4%) | 83.2 / **0%** / 88.3 | 93.9, 87.0, **0.0** recall |
| | | 36 (8%) | 71.0 / 0% / 59.6 | 85.7, 100.0, 98.7 recall |
| Qwen3.5-0.8B | 46.4 f1 / 0% exact / 48.7 recall | 1 (2%) | 42.1 / 0% / 43.9 | 39.1, 39.1, 49.6 recall |
| | | 8 (17%) | 35.0 / 0% / 27.8 | 48.3, 68.3, 71.7 recall |
| | | 15 (31%) | 23.0 / 0% / **21.7** | **0.0, 0.0, 0.0** recall (degenerate text) |

**On the dense model the honest statement is "the effect is clear from ~8% of heads;
at 2-4% this measurement does not separate the arms".**  At K=9 (2%) the retrieval arm
is *not* worse than the control — one random trial scores **10/10 exact, above the
9/10 baseline** — and at K=18 the random mean (59.8 f1) is produced by a single trial
collapsing to recall 0.0, while retrieval's recall (88.3) sits inside the random
range (93.9 / 87.0 / 0.0).  Only from K=36 does retrieval's recall (59.6) fall below
every random trial (85.7 / 100.0 / 98.7).  `random_std` is ±42.3 at K=18 — that is
the shape of the control, not noise around a clean separation.

**On the hybrid the causal claim is not just weak, it inverts at the last point.**
Its non-retrieval pool is only 15 heads, so K=16 is capped to 15 and the "random"
arm becomes the *entire* sub-threshold pool — a deterministic intervention, not a
sample.  It drives recall to **0.0 on all ten samples** with degenerate output
(`"1. 1. 1. 1. ..."`), while masking the top-15 *by score* leaves fluent but wrong
answers (`harp`, `sundial`) at recall 21.7.  So on this model the lowest-scoring
heads are collectively *more* critical for NIAH than the highest-scoring ones, and
the 0.1 threshold does not isolate a removable subset.  The exact-match axis is
uninformative there for a separate reason: the baseline is already 0%, because
`exact_match` requires the whole needle while the question asks for a sub-span (the
per-sample `generated_texts` in `masking_curve.json` show the answers are often
correct — `masking_recall.pdf` plots the LCS recall instead).

At the large-K end of the curve the "retrieval" arm necessarily reaches below the
0.1 threshold (there are only 28 such heads on the dense model while the curve goes
to K=148), so points there are "top-K by score", not "retrieval heads"; the artifact
records `k_effective` and the per-trial overlap so the transition is auditable.  At
large K everything collapses (both models fall to ~0), which is expected and is why
the curve is reported rather than a single point.

**How the two arms are matched (current code).**  The retrieval arm is the top K
heads *by score* — not "every head above the 0.1 threshold, capped at K", which
made K unrepresentative above the threshold count.  The random arm is drawn from
`non_retrieval_pool`, i.e. heads at or below the threshold; drawing from all
scoreable heads would put retrieval heads in the control most of the time on the
hybrid (33 of its 48 heads clear 0.1).  Both arms therefore remove exactly the
same number of heads at every point, and the realized counts are stored in
`masking_curve.json` (`k_effective`, `retrieval_masked`, `random_masked_mean`).
The pool size is also why the two models' K columns are not comparable: 31% of the
hybrid's scoreable heads is its whole control pool, 2% of the dense model's is nine
heads out of 420.

### Downstream: CoT vs extractive QA under masking — a pipeline check, not a measurement

Read this section as "the stage runs end to end and its artifacts are auditable", not
as evidence about the paper's Sec. 5.  The sets are 8 hand-written items each, so one
item is 12.5 points and the differences below are 1-4 items.

Chain-of-thought, 8 items, after recalibration (baseline off the floor):

| model | variant | baseline | retrieval masked | random masked |
|---|---|---|---|---|
| Qwen3-0.6B | answer-only | 75.0 | 75.0 | 25.0 |
| Qwen3-0.6B | **CoT** | **100.0** | 50.0 | 81.2 |
| Qwen3.5-0.8B | answer-only | 75.0 | 87.5 | 62.5 |
| Qwen3.5-0.8B | **CoT** | **100.0** | 75.0 | 31.2 |

With CoT the model needs the question text across steps, and masking retrieval heads
costs 25-50 points while random masking costs 19-69.  That is the paper's Sec. 5.3
direction, but the answer-only half does **not** replicate: the paper reports no
effect there, and we see the retrieval arm at or *above* baseline (75.0 vs 75.0 on
the dense model, 87.5 vs 75.0 on the hybrid).  With 8 hand-written arithmetic items
one item is 12.5 points, so these are weak measurements either way; they are
reported as measured.

Extractive QA (8 synthetic document/answer pairs).  Qwen3-0.6B baseline 67.5 F1:
masking 18/36/76 retrieval heads gives 19.6/10.5/0.0, versus 58.8/28.8/0.0 for
random — retrieval masking hurts much more at every K here.  On the hybrid the
baseline is 100.0 and the arms are 75.0/62.5/46.9 (retrieval) against
88.1/66.7/56.9 (random), i.e. a consistent but small gap.  The random arms are
erratic enough that the 8-sample measurement cannot separate the two cleanly.

### Cross-model correlation: read the mode

`compare` reports Pearson correlation between retrieval-score matrices.  Across
these two families it gives **0.93** in `sorted` mode — but that mode compares the
*sorted score vectors*, i.e. whether the two models have similarly shaped score
distributions, not whether they retrieve with the *same heads*.  Two models that each
have a few high-scoring heads and many low ones will correlate highly by
construction.  The paper's "different families correlate below 0.1" is a statement
about per-head correspondence, which needs a shared layer x head grid (i.e. models
of the same architecture).  Do not read the 0.93 as "these models use the same
heads"; `--mode grid` is the flag for the question the paper asked, and it is only
meaningful within a family.

## Limitations

* **Scope.** The reported run is 75 instances/model on a GPU (`t4` grid; the CPU
  `laptop` profile is 18 instances/model). Either is enough to locate retrieval
  heads and to show the masking effect, but the paper's per-head numbers come from
  ~600 instances and are smoother than what you get here. Raise `--profile paper`
  on a GPU for that grid.
* **The argmax domain is the input, and that is a choice.** Criterion (2) is read as
  the paper writes it -- "the *input* token that receives the most attention" -- so
  the argmax runs over the prompt positions only (`--argmax-domain prompt`, the
  default). With `full` the already-generated positions compete too, which makes a
  head's credit depend on how much the model happened to generate; the two domains
  give different head sets, and every artifact records which one was used.
* **The random control is drawn from the non-retrieval pool, as the paper does.**
  `tex-src` says "masking out random *non-retrieval* heads" (intro and Sec. 4), and
  `control_pool` implements exactly that: everything above the threshold is
  excluded, and the per-trial overlap with the retrieval arm is recorded so the
  exclusion is auditable. An earlier note here claimed the paper drew uniformly
  over all heads and that this biased the contrast; that was wrong about the paper.
  The one thing to keep in mind is the hybrid: its pool is only 18 of 48 heads, so
  at large K the control is nearly the whole pool and `random_std` collapses.
* **The masking curve measures F1/EM against the whole needle, while the question
  asks for a sub-span**, so a correct short answer is penalised by the metric
  itself. The curve also records the LCS needle recall (`retrieval_recall`,
  `random_recall_mean`, plottable with `metric="recall"`), which does not have that
  confound -- read the two together before attributing a drop to lost retrieval.
* **The pairing choice changes which heads are "retrieval heads".** `next_step` is
  the default because the row that *produces* `w` is the literal "attention scores
  at the step where `w` is generated"; `same_step` (the row at `w`'s own position,
  what a naive `output_attentions` + `generated_ids` pairing gives) is stored beside
  it. The two top-10 sets overlap 0/10 here, so every number in the tables is
  pairing-specific -- read `summary_same_step.json` before quoting one.
* **Thresholds depend on the denominator convention.** The score divides by the
  number of *unique* needle tokens (so `g_h` being a set is self-consistent), not by
  the raw needle length as the paper's "9 of 10" example reads. The artifact records
  both counts, `denominator_inflation`, and the per-head numerator
  (`copied_tokens`), so the raw-denominator variant is recomputable; the >0.1 shares
  in the tables are unique-token shares. Note the consequence for the threshold
  itself: because the scale is inflated by `denominator_inflation` (**1.0405** on
  these needles: 25.67 needle tokens against 24.67 unique), "score > 0.1" is
  *weaker* than "copied 10% of the needle tokens"; the equivalent raw-denominator
  threshold is `0.1 / denominator_inflation` ~ **0.096**. The counts in the tables
  are therefore not directly comparable to the paper's.
* **The length grid is geometric, not uniform.** The paper samples 20 lengths
  uniformly over 1K-50K, so its long contexts carry far more weight; `paper` here is
  7 geometric lengths (210 instances) and `t4` is 5 lengths up to 16K. "The
  mechanism reproduces" is a claim about this grid.
* **Fig. 3's "activation frequency" is `P(score > 0)`** -- which is what the paper
  defines it as ("the head activated on at least one token"), not a mean token
  count.  So the "gap" in that figure compares a mean with a thresholded indicator
  derived from the same scores; it is a real quantity but not an independent axis.
  The paper's other reading -- *how many* tokens a head copies -- is what
  `copied_tokens` records per instance, and no figure plots it yet.
* **Masking is inference-time ablation, not pruning.** Zeroing a head's `o_proj`
  slice is exactly equivalent to zeroing its attention row (proved by test), but it
  removes neither parameters nor KV entries, so the Sec. 5 KV-compression reading
  does not follow directly.
* **`mean_sink_rate` is "argmax at prompt position 0"**, which under a chat template
  is a template token rather than necessarily a BOS sink.  It is not a footnote:
  **0.759 on Qwen3-0.6B** (0.035 on the hybrid) means that in three of four steps
  where criterion (1) applies at all, the argmax sits on position 0, so criterion
  (2) can only fire in the remaining quarter.  Every absolute score, and the 0.1
  threshold with it, is conditioned on that -- which is another reason the shares
  are not comparable to the paper's.  `exact_match` in the artifacts is a
  normalised-contains check (NIAH convention), not character-exact equality.
* **Generation budget is now recorded on both sides.**  `detect` always had
  `n_instances_truncated` (11/75 on the hybrid at a 48-token budget); the ablations
  now carry `retrieval_truncated`/`random_truncated_mean` per K as well, because a
  drop in F1 cannot otherwise be told apart from a budget that ran out.
* **The paper's Sec. 4.3 "intrinsic" experiment is not tested here.**  It needs a
  base model and a derivative of it (the paper fine-tunes one); this registry has
  no such pair, and the 0.93 cross-model number is a `sorted`-mode correlation of
  score *distributions*, which the README says explicitly is not head
  correspondence.  So the claim "retrieval heads transfer to fine-tuned variants"
  is out of scope for this reproduction, not weakly confirmed by it.
* **Synthetic filler by default.** Every shipped profile generates the haystack
  from a 60-word template pool, so a 1K-50K context is highly repetitive text.
  Attention argmax is sensitive to that regularity, so the *absolute* scores are
  not transferable to the paper's natural documents; the head ranking and the
  masking effect are the parts that replicate. `--corpus <file>` (essay text) is
  wired through `detect`/`mask`, but no profile sets it yet.
* **The score's ceiling is set by the question, not only by the tokenizer.**
  `tokenization_attainable_score` records only whether the prompt exposes every
  needle token. The questions ask about a *sub-span* of the needle ("eat a
  sandwich in Dolores Park"), while `k` is the whole needle, so even a perfect
  copy head will not reach 1.0 on a long needle. `needle_recall` (the
  longest-common-subsequence ratio) and the per-instance `generated_text` are the
  handles for that gap.
* **`needle_recall` is a diagnostic, not the score.** It used to require the model
  to reproduce the needle from its *first* word, which scored a correct sub-span
  answer 0.0 and silently dropped it from the recited-only matrices; it is now an
  LCS ratio, and the old prefix measure is kept as `needle_prefix_recall`.
* **Hybrid models, one number short.** For Qwen3.5 the linear layers *cannot* be
  scored. Masking experiments (`token_mixer_ablation`) do cover them, so their
  contribution is measurable at layer granularity — just not as "retrieval
  heads", because the object does not exist there.  One result there deserves
  stating rather than leaving in the JSON: masking a *single* linear layer scores
  **above** the baseline (56.4 f1 against 46.4, all three sampled subsets), i.e. on
  this 8-sample measurement silencing one Gated DeltaNet layer does not hurt NIAH.
  With 18 linear layers and ±3.7 spread that is not "linear layers are harmful";
  it is a reminder that the hybrid's baseline sits low for a metric reason (see the
  masking section), so small positive deltas there are noise-level.
* **CPU dtype.** `float32` is used by default because it is the most numerically
  trustworthy for `argmax` over attention and fits in 15 GB; on GPU, `bfloat16`
  matches the paper's setting and is much faster.
* **CoT / QA datasets.** The paper uses GPT-4-generated news QA, MMLU, MuSiQue
  and GSM8K. Those are loaded from JSONL when supplied (`--data`); the built-in
  samples are small stand-ins that exercise the code path, not benchmarks.
  Calibration is not optional here: literal GSM8K items on a 0.8B model put the
  **baseline** at the floor (measured: 12.5% answer-only, 0% with CoT), and a
  baseline at the floor cannot show whether masking heads hurts. The built-in
  reasoning items are therefore small multi-step arithmetic a sub-1B model can
  actually solve -- chosen for headroom, not to be a benchmark.
* **Timings.** Qwen3.5's Gated DeltaNet layers fall back to pure-PyTorch kernels
  without `flash-linear-attention` / `causal-conv1d`, which dominates runtime on
  CPU. That affects speed only, not correctness.  On a job the driver runs a
  model's stages back to back in one process (`stage_plan`), so each checkpoint is
  loaded once rather than once per stage -- ~50 s saved per skipped load, and the
  resident-memory cost stays at one model.
* **The committed `ds-results/` matches the code; `results/` does not.**  The GPU
  tree was regenerated on the current pipeline (schema 5, held-out eval needle,
  prompt-only argmax, corrected filler sizing) and `docs/results-gpu.md` is
  generated from it.  The `results/` tree is an older CPU run kept for the quick
  start example, and is **not** refreshed: treat its numbers as historical.
* **`activation_freq` is `P(score > 0)`.**  A head's activation frequency is the
  fraction of instances in which it copied *any* needle token — i.e. a thresholded
  version of the same score.  The Fig. 3 "gap" therefore compares a mean with a
  positive rate, not two independent measurements.
* **`--k` vs `--k-frac`.**  Both default to unset; the command's default fractions
  apply only when neither is given, and passing both unions them.  Absolute K is
  still not comparable across a 48-head and a 448-head model — use `--k-frac` for
  cross-model statements.
