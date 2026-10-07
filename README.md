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
uv pip install --python .venv/bin/python transformers accelerate numpy matplotlib tqdm
uv pip install --python .venv/bin/python pytest          # for the test suite

# ~1.7 GB + ~1.5 GB of weights, checksum-verified
./scripts/download_models.sh
```

Requires `transformers >= 4.57` — Qwen3.5 (`model_type: qwen3_5`) is not in older
releases.

> **Note on the weight filenames.** Qwen3.5-0.8B's index points at a single shard
> literally named `model.safetensors-00001-of-00001.safetensors`; there is no
> `model.safetensors`. Fetching the obvious name returns a 15-byte
> `Entry not found` body that loads as a corrupt checkpoint. `download_models.sh`
> reads the real name out of `model.safetensors.index.json`, and both files are
> pinned by SHA-256.

---

## Quick start

```bash
.venv/bin/python -m retrieval_heads.cli describe --model qwen3.5-0.8b
.venv/bin/python -m retrieval_heads.cli detect   --model qwen3.5-0.8b --profile laptop
.venv/bin/python -m retrieval_heads.cli mask     --model qwen3.5-0.8b --k 1 2 4 8 16
.venv/bin/python -m retrieval_heads.cli figures  --runs results/qwen3.5-0.8b
```

Or the whole thing:

```bash
./scripts/reproduce_laptop.sh              # CPU, well under an hour
./scripts/reproduce_gpu.sh qwen3.5-0.8b    # paper-scale grid
pytest -m "not integration"                # fast unit suite
pytest                                     # + real-checkpoint tests
```

### On a Yandex DataSphere GPU job

```bash
export PATH="$HOME/yandex-cloud/bin:$PATH"
source scripts/datasphere_auth.sh                 # -> YC_IAM_TOKEN
CLI=.venv-datasphere/bin/datasphere
PROJECT=bt1u5v72b71eesdhp9k5

# once: build a persistent venv on the project disk and validate every stage
$CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-bootstrap.yaml

# then, with ~40 s startup instead of ~9 min:
$CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-cached.yaml --async
```

`configs/datasphere/` also holds `t4.yaml` (cacheless fallback), `laptop.yaml`,
`paper.yaml` and `smoke.yaml`.  Everything non-obvious about this path — the pip
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
is generated can be paired with `w` in two ways. Both are computed from a single
decoding pass (no extra forward passes):

| `pairing` | the row | mechanism |
|---|---|---|
| `next_step` *(default)* | the row whose query **produces** `w` — the head points at the source position it is about to paste | CopyNet-style paste |
| `same_step` | the row at `w`'s **own** query position — the head looks back at the source of the token it just emitted | induction-head-like |

The default is the literal reading of “the current token being generated as `w`”
with the attention at that same step. **The two are not interchangeable** — on
Qwen3.5-0.8B a single instance already ranks different heads top (see Results),
so both are stored in every run and the choice is a reporting decision, never
silently baked in.

### Detection grid

The paper's full recipe is 3 needle sets × 20 lengths in 1K–50K × 10 insertion
depths ≈ 600 instances per model. `DetectionConfig` exposes every axis, with
named profiles:

| profile | grid | use |
|---|---|---|
| `smoke` | 1 × 1 × 1 | CI / sanity |
| `laptop` | 2 × 3 (1K, 2K, 4K) × 3 = 18 | this repo's runs |
| `paper` | 3 × 7 × 10 | GPU |

Filler text is generated from a seeded word list (offline, deterministic);
`--corpus FILE` swaps in a real corpus such as the Paul Graham essays.

---

## Code map

| file | paper section | what it does |
|---|---|---|
| `retrieval_heads/models.py` | — | **architecture-aware head discovery**: separates softmax attention from linear/recurrent mixers |
| `retrieval_heads/haystack.py` | 3 | needle insertion at a given depth; token span recovered from character offsets |
| `retrieval_heads/attention.py` | 3, 4.1 | attention capture (public `output_attentions` **or** a monkeypatch fallback) and head / token-mixer ablation hooks |
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
  `(heads, seq, seq)`, so memory stays flat in context length — the difference
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

The numbers below come from one GPU run
([job `bt1hqv1b91ht36s2egdp`](https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1u5v72b71eesdhp9k5/job/bt1hqv1b91ht36s2egdp),
NVIDIA L4, ~39 min) over **75 instances per model** — 3 needles x 5 depths x 5
lengths.  Full tables: [`docs/results-gpu.md`](docs/results-gpu.md); regenerate any
time with

```bash
.venv/bin/python scripts/summarize_results.py ds-results
```

**A caveat on the length axis of that run.**  It used the pre-fix filler sizing, so
requested lengths 1K-16K came out as **1.6K-26K** (realized values are recorded per
instance in `instances_*.jsonl`).  The grid is still five geometrically spaced
contexts, so the conclusions hold, but the labels in the tables below are the
*realized* ones.  The committed code lands within a couple of percent of what it is
asked for.

### Detection

| model | recited | mean recall | top head | score | >0.1 | >0.5 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (dense) | 73/75 | 0.96 | `L16H14` | 0.86 | 22/448 (**4.9%**) | 5/448 (1.1%) |
| Qwen3.5-0.8B (hybrid) | 53/75 | 0.70 | `L11H1` | 0.74 | 30/48 (**62.5%**) | 8/48 (16.7%) |

**The dense control reproduces the paper's sparsity claim; the hybrid does not.**
4.9% of Qwen3-0.6B's heads clear the 0.1 threshold, inside the paper's quoted 3-6%,
and 1.1% clear 0.5, consistent with "less than 5%".  Qwen3.5-0.8B has 62.5% of its 48
scoreable heads above 0.1.  Two readings, both worth stating:

* over the heads that *can* retrieve (its 6 full-attention layers), retrieval in
  this architecture is not sparse at all;
* over *all* its token-mixer heads (48 attention + 288 Gated DeltaNet = 336), the 8
  strongly-retrieving heads are 2.4% — back inside the paper's range, but only by
  counting objects the retrieval score is not defined for.

The honest statement: the paper's "a few percent" is a property of a dense stack in
which most layers do no retrieval, and it does not transfer to a hybrid stack where
18 of 24 layers cannot retrieve at all and the remaining 6 must carry it.

The hybrid's 53/75 is also a real result, not noise: 15 of the 22 failures are at
the 26K contexts, i.e. a 0.8B hybrid model loses the needle at the long end of its
own grid.

### The paper's definition is ambiguous, and the ambiguity decides the answer

The retrieval score pairs a head's attention row with "the token being generated",
which can mean the row that *produces* the token (`next_step`) or the row at the
token's *own* position (`same_step`).  On **both** models the top-10 heads under the
two readings overlap **0/10**:

| model | `next_step` top heads | `same_step` top heads |
|---|---|---|
| Qwen3.5-0.8B | L11H1, L15H7, L19H5, L23H0, L23H5 | L7H7, L11H3, L3H7, L7H6, L7H2 |
| Qwen3-0.6B | L16H14, L21H8, L20H14, L18H5, L6H11 | L6H6, L2H10, L11H2, L1H15, L6H7 |

The pattern is systematic, not noise: `next_step` selects **late** layers (where the
copy is emitted into the residual stream), `same_step` selects **early** layers (the
induction-head position, where the retrieved token is staged for later use).  Both
come out of the same decoding pass, so this costs nothing to report — and no
reproduction should quote one without the other.

### Masking: the causal claim holds

Needle-in-a-Haystack, exact match, retrieval heads vs random heads:

| model | baseline | best retrieval-masked case | random |
|---|---|---|---|
| Qwen3-0.6B | 95.5 f1 / 100% exact | 66.5 / **0%** at K=9 (2% of heads) | 95.5 / 100% |
| Qwen3.5-0.8B | 86.7 f1 / 100% exact | 45.1 / **0%** at K=4 (8% of heads) | 69.5 / 67% |

Removing ~2-8% of heads *by retrieval score* destroys the exact answer, while
removing the same number at random leaves it intact.  At large K everything
collapses (both models fall to ~0), which is expected and is why the curve is
reported rather than a single point.

### Downstream: CoT depends on retrieval heads, extractive QA mostly does

Chain-of-thought, 8 items, after recalibration (baseline off the floor):

| model | variant | baseline | retrieval masked | random masked |
|---|---|---|---|---|
| Qwen3-0.6B | answer-only | 75.0 | 50.0 | 12.5 |
| Qwen3-0.6B | **CoT** | **100.0** | 75.0 | 81.2 |
| Qwen3.5-0.8B | answer-only | 75.0 | 25.0 | 68.8 |
| Qwen3.5-0.8B | **CoT** | **100.0** | 50.0 | 93.8 |

With CoT the model needs the question text across steps, and masking retrieval heads
costs 25-50 points while random masking costs 6-19.  That is the paper's Sec. 5.3
result.  The answer-only half does **not** replicate: the paper reports no effect
there, and we see 25-50 point drops too.  With 8 hand-written arithmetic items this
is a weak measurement either way, but it is reported as measured.

Extractive QA (8 synthetic document/answer pairs).  Qwen3-0.6B baseline 67.5 F1:
masking 18/36/76 retrieval heads gives 24.4/16.2/16.2, versus 55.1/20.3/4.2 for
random.  Retrieval masking hurts more at small K, which is the paper's direction,
but the random arms are erratic enough that the 8-sample measurement cannot separate
the two cleanly.

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

* **Scope.** The grid here is deliberately small (18 instances/model) so it fits
  on a laptop; it is enough to locate retrieval heads and to show the masking
  effect, but the paper's per-head numbers come from ~600 instances and are
  smoother than what you get here. Raise `--profile paper` on a GPU.
* **Hybrid models, one number short.** For Qwen3.5 the linear layers *cannot* be
  scored. Masking experiments (`token_mixer_ablation`) do cover them, so their
  contribution is measurable at layer granularity — just not as "retrieval
  heads", because the object does not exist there.
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
  CPU. That affects speed only, not correctness.
