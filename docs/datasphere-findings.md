# DataSphere findings

Findings from actually running this pipeline as DataSphere Jobs on the T4-class
GPU configuration.  `run-in-datasphere.md` is the instruction document; this file
records where reality differed from it, plus the workarounds now baked into the
repo.  Everything below was observed in job logs, not inferred.

Job ids are listed so each claim can be re-checked on the project's **DataSphere
Jobs** tab.

## 1. Job entry point path

`run-in-datasphere.md` §3.2 uses `cmd: python3 datasphere_job.py ...`.

`local-paths: [retrieval_heads, configs, scripts]` unpacks each entry **under its
own name** into `/job`, so the driver lands at `/job/scripts/datasphere_job.py`,
not `/job/datasphere_job.py`.  The job failed immediately with:

```
python3: can't open file '/job/datasphere_job.py': [Errno 2] No such file or directory
```

**Fixed:** all job configs now use `cmd: python3 scripts/datasphere_job.py ...`
(observed in `bt13ok6pqblk89kuno8j`).

## 2. Caching: weights yes, environment no

| Run | `inputs` upload | env build |
|---|---|---|
| `rh-smoke` (first) | `uploading 4 files (3.1GB)`, ~4 min | 146 packages resolved |
| `rh-t4-smoke2` | `uploading 1 files (43.7KB)` | 68 packages, from scratch |
| `rh-t4-smoke4` | `uploading 1 files (44.9KB)` | 65 real `Downloading` lines (~3.5 GB) |
| `rh-t4-smoke5` | `no files to upload` | rebuilt again (~7.5 min) |

So the **project caches `inputs`** (checkpoints upload once), but the **Python
environment is rebuilt on every job** — `smoke3` and `smoke4` had near-identical
requirement sets and `smoke4` still re-downloaded every wheel.  Budget ~8–9 min
and ~3.5 GB per job, and prefer one job with many stages over many small jobs.

**Workaround (option 2):** `t4-bootstrap.yaml` builds a venv on the project disk
(`flags: [attach-project-disk]`, `${DS_PROJECT_HOME}/rh-venv`); `t4-cached.yaml`
reuses it.  See §6 for the re-exec trick this needs.

## 3. pip 25.1.1 crashes building the environment

With any requirement set containing torch + transformers, the platform's pip
(25.1.1) fails after a *successful* resolve:

```
pip/_internal/resolution/resolvelib/resolver.py:276, in get_topological_weights
    assert len(weights) == expected_node_count
AssertionError
```

Cause: `huggingface-hub` is required twice (`transformers` and `tokenizers` both
depend on it).  The resolver's requirement table ends up with one more entry than
the dependency graph has nodes, so the assertion trips.  Shrinking the requirement
set does not help (`bt1semol1328oj66k6u2` failed with 5 pins).

Two things that did **not** fix it, so nobody wastes time on them:

* `torch==2.10.0` instead of 2.14.1.  (torch ≥ 2.12 additionally pulls
  `cuda-toolkit[cudart,cufft,...]` with extras, which is a nastier graph, but it is
  not the trigger — 2.10.0 fails identically.)
* Trimming the top-level requirements to five pins.

**Fixed by:** `scripts/requirements-datasphere.txt` is now a complete, exact lock
of the dependency closure (65 pins, extracted from a real resolve), and every job
config sets `pip.no-deps: 'true'`.  With no dependency edges the graph is
one node per requirement and the assertion cannot fire.  Observed working in
`bt13ok6pqblk89kuno8j` (`Successfully installed ... torch-2.10.0 ...`).

## 4. `requirements-file` must contain only requirement lines

The CLI runs `packaging.Requirement()` over **every** line, including comments and
blank-ish lines:

```
packaging.requirements.InvalidRequirement: Expected package name at the start of
dependency specifier
    # Environment for DataSphere Jobs (see run-in-datasphere.md).
```

So the file cannot carry a header comment.  The rationale for its contents lives
here and in `README.md` instead.

## 5. The CLI's `cmd` contract

* **`--seed` placement.** `scripts/datasphere_job.py` builds
  `detect --model ... --seed 0 ...`, but `--seed` was only declared on the *top
  level* parser, where argparse requires it *before* the subcommand.  The job died
  with `error: unrecognized arguments: --seed 0` (`bt1otb53j12aag4u4jdo`).
  `--seed` is now accepted per subcommand, with `default=argparse.SUPPRESS` so the
  subparser does not clobber the top-level value.
  Pinned by `tests/test_cli_argv.py`, which replays the driver's own `SCALES`
  table through the real parser — this class of mismatch otherwise costs a full
  job round trip to discover.

## 6. `${DS_PROJECT_HOME}/.../python` cannot be the first token of `cmd`

Putting the cached venv's interpreter at the start of `cmd`:

```
cmd: ${DS_PROJECT_HOME}/rh-venv/bin/python scripts/datasphere_job.py ...
```

fails in the CLI before the job is even created:

```
ValueError: file `${DS_PROJECT_HOME}/rh-venv/bin/python` was not found
```

`parse_python_main_module` accepts the first token only if it is a *common* python
name (`python3`) or an actually-executable interpreter (`is_python_interpreter`
runs it with `--version` on the laptop).  An unresolved `${...}` path is neither.

**Workaround:** the entry point stays `python3` (the platform venv, built from a
one-line `requirements-platform.txt`), and the driver re-execs itself into the
cached venv via `--use-venv ${DS_PROJECT_HOME}/rh-venv`.  `strip_flag` removes
`--use-venv`/`--bootstrap-venv` before `os.execv`, and `RH_VENV_ACTIVE` guards
against a re-exec loop.

## 7. GPU configuration: `gt4i.1` is an L4, not a T4

```
[entry] GPU: NVIDIA L4 sm_89 22.2 GiB | cuda 12.8 | 1 device(s)
```

The slot is T4-class in name, but the pool hands out an **L4** (sm_89, 22.2 GiB,
Ada generation).  That matters for planning:

* bfloat16 is natively supported on sm_89 (it is *not* on a real T4/sm_75).
* 22 GiB of VRAM is comfortable for a 0.8B model at long context.
* The GPU configuration was accepted by the community, and `gpu_stats.tsv` is
  produced.  The "GPU configurations and quotas unverified" caveat in the task
  document can now be closed: `bt107kjm3es8vung7130` ran **all seven stages** to
  `SUCCESS` on this configuration (see §17).

### Measured GPU speed

From that job, for a single 1024-token needle instance: **2.8 s** (Qwen3.5-0.8B)
and **2.5 s** (Qwen3-0.6B), against roughly 25 s per instance on the laptop CPU.
The full `t4` grid (75 instances per model, plus every ablation) is therefore a
matter of tens of minutes, not hours.  An earlier "several hours" estimate in this
project was an unjustified extrapolation from CPU timings and should be ignored.

## 8. Inputs must be placed on the model's device

First GPU run of `detect` (`bt1agsld6g8t9hl5fdub`) failed:

```
RuntimeError: Expected all tensors to be on the same device, but got index is on
cpu, different from other tensors on cuda:0
```

The model is moved to CUDA once at load time, but `input_ids` are built later from
tokenizer output, which is always on the CPU.  **Fixed** in
`retrieval_heads/models.model_device()` plus three call sites
(`decode_with_attention`, `greedy_generate`, `_generate_text`).  This never
surfaces on CPU, so it is exactly the class of bug the GPU smoke run exists to
catch.

## 9. Yandex PyPI mirror

The correct simple-index URL is:

```
https://mirror.yandex.ru/pypi/web/simple/
```

(not `/pypi/simple/`, which 404s — checked both with and without a trailing
slash).  It is a **partial** mirror:

| package | status |
|---|---|
| `numpy`, `matplotlib`, `tqdm`, `tokenizers`, `safetensors`, `sympy`, `networkx`, `pillow`, `regex`, `pyyaml`, `filelock`, `triton` | `200` |
| `torch`, `transformers`, `huggingface-hub` | `404` |
| `nvidia-*` (any) | `404` |

Since the ~3.4 GB of torch + nvidia wheels is exactly what is missing, the mirror
cannot meaningfully speed up the environment build.  It is wired in as
`pip.extra-index-urls` (PyPI stays primary) purely to shorten the small-package
downloads; it is not a substitute for caching the venv.

## 10. Cloud Registry as a caching PyPI proxy — evaluated, not adopted

`yc` 1.40.0 does support the remote-repository feature the docs describe:

```bash
yc cloud-registry v1 registry create <FOLDER-ID> \
  --name rh-pypi-proxy --registry-kind pypi --registry-type remote
```

`kind` accepts `pypi`, `type` accepts `remote`, and `create --example-yaml`
confirms a free-form `properties` map plus `pattern_filter`.  Folder for this
project: `b1gata76lvau0np2n4jd` (cloud `b1gt2raj9o4qsrko03a0`).
`yc cloud-registry v1 registry list` is currently **empty** — no registry exists.

It was not adopted, for two reasons:

1. **Auth.** pip would need
   `https://iam:<token>@registry.yandexcloud.net/pypi/<ID>/simple/`.  The job
   config would have to carry an IAM token (it is the only way to put it in
   `pip.index-url`), which the instruction doc explicitly forbids — tokens live
   ~12 h and would be stored in a config file.  Whether a DataSphere job can reach
   the registry under its own service account instead is unverified.
2. **It solves nothing we still need.**  The pip assertion bug (§3) is about the
   dependency graph, not the index, so the lock + `no-deps` workaround stays
   regardless.  And the project-disk venv (§12) already removes the rebuild cost.

Worth revisiting only if a *fully cacheless* setup is wanted (e.g. many different
images), where the proxy would make every fresh env build fast.

## 11. The image ships more than one Python

```
[entry]   base interpreter: /usr/bin/python3.10 (this job runs 3.10.12)
[entry]   venv python     : 3.10.12
```

`/usr/bin/python3` in the job image is **3.11**, while the platform's own venv is
**3.10.12**.  The naive "first non-venv python3" search therefore built a 3.11
venv, which cannot install the `cp310` wheels the lock file pins
(`torch-2.10.0-cp310-...`).  `find_stable_interpreter()` now matches
`sys.version_info[:2]` exactly and the bootstrap re-checks the version after
creating the venv, aborting rather than installing incompatible wheels.

## 12. The project disk is shared — treat it as hostile

`${DS_PROJECT_HOME}` is shared with other users' venvs, repositories and files.
Three rules are now enforced in code:

* **Namespaced path.**  The venv lives at
  `${DS_PROJECT_HOME}/ai360-heads-in-the-haystack/venv`, never at a bare
  `${DS_PROJECT_HOME}/rh-venv`.  (An earlier cancelled run did write to
  `project/rh-venv` before this was fixed; it was abandoned rather than deleted,
  because deleting a directory we cannot prove we own is the more dangerous act.)
* **Ownership marker.**  The venv directory carries `.rh-venv-owner` plus
  `.rh-venv-stamp` (requirements hash + python version).  If the target exists,
  is non-empty and has no marker, the job aborts with an explanatory message
  instead of writing into it.
* **Read-only survey first.**  `--project-home ${DS_PROJECT_HOME}` lists the disk
  root before anything is created, so the guarantee is visible in the log.
  Nothing outside our own leaf directory is ever written or removed.

### Audit: the stray `rh-venv` is ours, not someone else's

An earlier cancelled job created `${DS_PROJECT_HOME}/rh-venv` **before** the guard
existed, which raised the obvious question: did that name collide with somebody
else's directory?  `configs/datasphere/inspect-disk.yaml` (strictly read-only:
`--inspect-dir`, no weights, no outputs) settles it:

```
pyvenv.cfg:
  include-system-site-packages = false
  executable = /usr/bin/python3.11
  command    = /usr/bin/python3 -m venv --copies /job/project/rh-venv
site-packages: distutils-precedence.pth, pip (26.2.1), pkg_resources,
               setuptools (65.5.0)
```

`pyvenv.cfg` records *our* command line verbatim, the timestamp matches the
cancelled job, and site-packages holds nothing but pip/setuptools — that job died
right after `pip install --upgrade pip`, before installing any requirement.  No
third-party package of anyone else's was overwritten.  The directory is left
abandoned rather than deleted: removing it would be a delete on a shared disk, and
the driver deliberately has no delete capability.

The disk root as that job saw it:

```
.ipynb_checkpoints   =4.37.2   Justamouse   Nikita-prog-art
ai360-heads-in-the-haystack   getting-started_dedicated_{en,ru}.ipynb
logs   rh-venv   timon
```

`Justamouse`, `Nikita-prog-art` and `timon` line up with the remote branches
`origin/justamouse`, `origin/nastya`, `origin/timon` — a shared, multi-person
project, which is exactly why the namespacing matters.

## 13. The cached venv, measured

`job bt1k1hndn5lm6ld0lvjc` built it and then failed on an unrelated bug, but the
part that matters succeeded:

```
[entry]   python 3.10.12
[entry]   torch 2.10.0+cu128 | cuda True
[entry]   transformers 5.18.0
[entry] venv ready: 7.04 GiB, stamp .../venv/.rh-venv-stamp (1afe0e2c4cffa627)
```

* **7.04 GiB** on the project disk — almost all of it the CUDA wheels. Worth
  knowing before filling a shared disk with several such venvs; that is another
  reason the path is namespaced and the script never deletes anything it cannot
  prove it owns.
* The **platform** env build with the one-line `requirements-platform.txt` was
  fast: `system.log` is 2.7 KB, versus ~26 KB for the full requirement set.
* The venv is now **idempotent**: `.rh-venv-stamp` holds a hash of the
  requirements file, and a re-run with a matching stamp verifies the imports and
  skips pip entirely.

## 14. `local-paths` snapshots the code at job creation

The same job died on

```
NameError: name 'reexec_into_venv' is not defined
```

not because the code was wrong at the time of reading, but because the job had
uploaded `scripts/` at **creation** time, before the function was restored. Any
fix to code that lives in `local-paths` requires a **new job**; editing the
working copy does nothing to a job that is already created (and nothing to the
snapshot it took).  It also means a job's logs describe the code as of its own
creation, which is worth remembering when comparing two runs.

## 15. Reproduce

```bash
cd ~/projects/ai360-heads-in-the-haystack
export PATH="$HOME/yandex-cloud/bin:$PATH"
source scripts/datasphere_auth.sh
CLI=.venv-datasphere/bin/datasphere
PROJECT=bt1u5v72b71eesdhp9k5

# once: build the persistent venv and validate every stage on the GPU
$CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-bootstrap.yaml

# then, cheaply (~40 s startup).  Blocking mode is the only one that streams
# progress, so run it in the background with the output in a log file; the job
# itself runs on the service and survives a client disconnect.
mkdir -p logs
nohup $CLI project job execute -p "$PROJECT" -c configs/datasphere/t4-cached.yaml \
    > "logs/ds_$(date +%m%d_%H%M).log" 2>&1 &
tail -f logs/ds_*.log
```

`job attach` does not stream the job's stdout (it waits, printing only its own
keep-alive lines), and the job page shows it after completion -- so neither is a
progress view.  Once the run ends, logs and artifacts come back with
`$CLI project job download-files --id <job_id> --with-logs --output-dir <dir>`.

`t4.yaml` is the cacheless variant (full platform env build every time).  It
remains the reference path if the project disk is unavailable.

## 16. Edit discipline learned the hard way

Two of the bugs above were self-inflicted by careless string surgery on
`scripts/datasphere_job.py`:

* a slice-replacement silently deleted `strip_flag` and `reexec_into_venv`;
* an earlier refactor left **two** `def main` definitions, the first truncated —
  it was shadowed by the second, so the driver worked while half of `main` was
  dead code.

Both are now covered by `tests/test_cli_argv.py`, which replays the driver's own
`SCALES` table through the real CLI parser and asserts the source structure
(exactly one `main`, and that it calls `stage_argv` / `reexec_into_venv` /
`prepare_models`).  Run `pytest -m "not integration"` before every job.

## 17. What is now verified end to end

Job **`bt107kjm3es8vung7130`** (`rh-t4-smoke`, all seven stages, both models,
`smoke` scale) finished **`SUCCESS`** on the L4 with the cached venv, and every
artifact came back:

```
ds-smoke-results/
  correlation.json  overlap.json
  figures/{ring_graph,score_distribution,heat_map,layer_profile,
           corr_map,masking_heads,task_qa,task_cot,mixer_ablation}.{pdf,png}
  qwen3.5-0.8b/{model_info,scores_next_step,summary_next_step,masking_curve,
                mixer_ablation,task_qa,task_cot}.json + instances_next_step.jsonl
  qwen3-0.6b/  (same set, no mixer_ablation -- that model is not hybrid)
```

That closes the loop on the three things that were guesswork before:

| | before | now |
|---|---|---|
| GPU configuration allowed in the community | unverified | `gt4i.1` accepted, job ran |
| environment with torch + CUDA builds | unverified | builds; `cuda True`, torch 2.10.0+cu128 |
| cached venv actually reused | untested | `venv already matches ...; skipping install`, stages run from it |

### The smoke results also cross-check the CPU run

Same top head on GPU (1 instance) and CPU (18 instances) for both models:
`L15H7` for Qwen3.5-0.8B and `L16H14` for Qwen3-0.6B.  The dense model again lands
inside the paper's quoted 3-6% band (6.0% of heads above 0.1); the hybrid again
does not.

### A measurement trap found on the GPU

The same job exposed a second instance of the mistake described in §16's spirit:
the CoT baseline was **at the floor** -- 12.5% answer-only, 0% with chain-of-thought
-- because literal GSM8K items are beyond a 0.8B model and `max_new_tokens=128`
truncated the reasoning before the `####` answer.  A baseline at the floor cannot
show whether masking retrieval heads hurts, so that subsection was measuring
nothing.  `builtin_reasoning_samples()` now holds small multi-step arithmetic a
sub-1B model can solve; the real benchmarks still load via `--data`.

Both this and the earlier needle problem have the same shape: a test whose score
cannot move is not a test.  Check the *baseline* before trusting an ablation.

## 18. A 16K float32 prefill OOMs on a 22 GiB L4

The first full `t4` run (`bt1e0iq7jakl30kd3qjn`) died in `detect` after 7 minutes:

```
File ".../transformers/integrations/sdpa_attention.py", line 158, in sdpa_attention_forward
    attn_output = torch.nn.functional.scaled_dot_product_attention(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.61 GiB.
GPU 0 has a total capacity of 22.17 GiB of which 13.84 GiB is free.
```

The model is 3.2 GB in float32, so the weights are not the problem.  `sdpa` was
being used -- but on **float32** SDPA does not reach a flash kernel and can fall
back to the math backend, which materialises the full `(heads, seq, seq)` score
matrix.  At 16K tokens that is 8.6 GB for the scores alone, and with the softmax
copy and the dtype cast the allocator ends up asking for ~20 GB in one go.  The
needle tests at 1K and 2K were never going to surface this.

Two independent fixes, both now in place:

* **`--dtype bfloat16`.**  What the paper uses on GPU anyway, half the memory, and
  it lets SDPA pick a flash kernel on sm_89 instead of the math backend.
* **`--prefill-chunk 4096`.**  Feed the prompt through the KV cache in chunks, so
  peak prefill memory is `O(chunk^2)` no matter which kernel is chosen.  This is
  the fix that does not depend on dtype or hardware, so it is the one worth having
  in the code rather than only in a config.

Chunked prefill changes the execution path, so equivalence is asserted, not
assumed: `test_chunked_prefill_matches_single_shot` runs the same prompt one-shot
and at `chunk=16`, and requires an identical greedy continuation and matching
prefill logits.

`PYTORCH_ALLOC_CONF=expandable_segments:True` is set on the GPU jobs as well; the
error message itself suggests it, and fragmentation after a model load is real.

Note the failure mode to watch for in future runs: this was **not** a bug in the
retrieval code, and `detect` had already processed the short contexts correctly.
The run simply hit the largest context in the grid.

## 19. The context-length axis was 1.6x off label

Watching the fixed run work, the log lines gave the game away:

```
chunked prefill: 26295 tokens in chunks of 4096
```

`--lengths ... 16384` had produced a **26295**-token prompt.  `HaystackBuilder.text`
appended `n_tokens // 10` sentences per batch on the assumption of ~11 tokens per
filler sentence; the templates actually average **15.95**.  Measured overshoot was
a flat 1.58-1.60x at every length:

| requested | realized (before) | realized (after) |
|---|---|---|
| 1024 | 1614 | 1021 |
| 4096 | 6538 | 4115 |
| 16384 | 26156 | 16440 |
| 32768 | ~52k | 32859 |
| 49152 | ~78k | 49206 |

The grid is still a geometric spread of five lengths, so the *conclusions* are
unaffected -- but the labels were wrong, and "16K" was quietly outside the range
the run was sized for.

**Correction (code review).** The first fix was incomplete, and the "realized
(after)" column originally published here said 1024 → 1033.  That was wrong:
`text()` sizes only the *filler*, while the needle, the question and the chat
template are appended afterwards, so the rendered prompt carried a fixed ~40-token
overhead on top.  Measured with the real tokenizer, the first fix still landed at
1088 (Qwen3-0.6B) and 1091 (Qwen3.5) for a requested 1024 -- +6%, not "a couple of
percent".  The values in the table above are the measured ones after the second
fix:

* `text()` sizes the last batch from the tokens remaining (no `+1`, no `max(4, …)`),
  so it stops within one sentence (~16 tokens) instead of overshooting a batch;
* `build_needle_sample` renders the **whole prompt**, measures it, and re-budgets
  the filler up to four times, which absorbs the needle/question/template overhead.

With both in place the error is ~2% or better across 1K-49K (measured ≤0.5% at the
detection grid's own depths, and up to ~1.4% under other seeds, where the last
filler sentence is coarser than the tolerance; a run that cannot reach the 2%
tolerance says so in the log).  Pinned by
`tests/test_regressions.py::test_realized_context_length_tracks_the_request`.

Worth keeping in mind as a pattern: three of the four measurement problems found in
this project (needle answer shape, CoT baseline at the floor, and this) were
invisible in the code and only showed up as implausible *numbers* in a real run.
Print the realized values, not the requested ones.  A fourth instance of the same
pattern is that the first "after" column here was itself an unverified number.
