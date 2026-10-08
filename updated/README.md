# Retrieval heads on current Transformers

The code keeps the original Qwen3.5 checkpoint and separates four concerns:

```text
experiment/data     builds a context and inserts the needle
experiment/locator  finds the expected-answer span in prompt tokens
models/qwen_common  shared original-model loading and token-by-token decode
models/qwen35       Qwen3.5 hybrid full-attention layer selection
models/qwen3        dense Qwen3 and Thinking final-answer extraction
attention           observes or masks full-attention heads during decoding
```

`Qwen35Adapter` contains only model-specific work: loading Qwen, applying its
chat template, discovering full-attention layers, prefill and cached decoding.
It does not know about ROUGE, the 0.9 locator, experiment grids or JSON files.
It does not contain a rewritten Qwen: `self.model` is the original
`Qwen3_5ForCausalLM` loaded by Transformers.

The complete flow is:

```text
ExperimentCase
  -> ContextBuilder inserts the needle
  -> LegacyOverlapLocator finds its prompt-token span (> 0.9 overlap)
  -> Qwen35Adapter performs fast prefill
  -> one cached decode forward per generated token
  -> observable attention backend reports that token's attention
  -> collector scores heads and/or saves the full trace
  -> AnswerScorer calculates the source experiment's ROUGE-1 recall
```

The backend is selected only inside cached decoding and restored afterwards.
It uses Qwen's own projected Q/K/V tensors and returns the normal attention
output to Qwen; the rest of the model is never replaced.

## Environment

Python 3.11 or 3.12 and CUDA are recommended. Native Windows setup and all
experiment commands are documented in [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md).
Running the same experiment as a Yandex DataSphere Job is documented in
[DATASPHERE_GUIDE.md](DATASPHERE_GUIDE.md).

Install the appropriate PyTorch build first, then:

```bash
cd updated
python -m pip install -r requirements.txt
```

Use `--model` for a Hub ID or a direct checkpoint path, and `--adapter` for
the architecture implementation (`qwen35` or dense `qwen3`). Repeat
`--model-search-dir /path/to/models` to search roots in order for
`ROOT/Qwen/Qwen3.5-0.8B` before falling back to Hugging Face.
There is no implicit search directory. All job YAMLs explicitly pass
`--model-search-dir ${DS_PROJECT_HOME}/models` and attach the project disk.
See [DATASPHERE_GUIDE.md](DATASPHERE_GUIDE.md) for
the one-time download into project storage and reuse across jobs.
Local checkpoints are loaded offline; incomplete local folders raise an error.
The checkpoint is downloaded automatically by Transformers if not found locally.
SDPA is the cross-platform prefill default.

## Qwen3 Thinking

`--adapter qwen3 --model Qwen/Qwen3-4B-Thinking-2507` selects the dense Qwen3
adapter. The 4B checkpoint has 36 full-attention layers with 32 heads each.
Generation remains greedy. Attention observers receive only final-answer tokens
after `</think>`; thinking and the closing marker are excluded from scores and
saved traces. Masking still applies throughout decode. Unlike the legacy Qwen3.5 rule, newlines do not stop
Qwen3 decoding; EOS or `--max-new-tokens` does. The checkpoint's own chat template
is preserved, including its opening thinking tag.

ROUGE uses only the final answer after `</think>`. Results retain
`raw_model_response`, `generated_token_ids` and `finish_reason`; an unfinished
thinking segment yields an empty final answer, so quoting a needle in a thought
does not falsely count as a correct response. With unfinished thinking, no
attention steps reach the collectors. Trace step indices retain their positions
in the complete generated-token sequence; the first captured step need not be 0.

`datasphere/qwen3-smoke.yaml` loads the model strictly from project storage,
offline, on T4/float16: 3 detection cases, context 1000, depth 50, limit 2048 new
tokens. This finite smoke budget may truncate thinking. See the DataSphere guide
for the command and checkpoint path. The existing Qwen3.5 storage smoke is unchanged.

## Attention modes

Every generation receives an explicit `AttentionRequest`:

```python
AttentionRequest(capture="none")  # normal generation
AttentionRequest(capture="top1")  # one source position per layer/head/token
AttentionRequest(capture="full")  # all decode attention probabilities
AttentionRequest(  # source-compatible logit intervention
    blocked_heads=frozenset({(3, 2), (7, 5)}),
    mask_mode="legacy_uniform",
)
```

Full traces are moved to CPU without changing the model's probability dtype.
They contain only decode attention. A full prefill trace would be quadratic in
context length and was not collected by the source experiment either.

Qwen3.5 is hybrid. Only `full_attention` layers have ordinary token-to-token
attention matrices. Gated DeltaNet layers remain untouched.

## Detection

The input corpora live in `data/haystack_for_detect/` (detection) and
`data/PaulGrahamEssays/` (masking).

Small smoke run:

```bash
python retrieval_head_detection.py \
  --s 1000 --e 1000 --context-intervals 1 \
  --depths 50
```

The default `--capture top1` is sufficient for retrieval-head scoring.
To retain every attention probability for each analyzed token:

```bash
python retrieval_head_detection.py \
  --s 1000 --e 1000 --context-intervals 1 \
  --depths 50 --capture full
```

Outputs:

```text
detection/run.json         detection configuration and completion state
detection/results/         answer results, one per detection case
detection/contexts/        generated contexts, one per detection case
detection/head_scores.json aggregated head scores
detection/attention/*.pt   full traces, only with --capture full
```

The `.pt` trace is a plain dictionary loadable with `torch.load`. It contains
`prompt_token_ids` and one entry per analyzed token. Each step stores
`token_id` and `layers[layer_index]`, shaped `[heads, key_length]`. The case ID
is included in every result, context and trace filename so the three detection
cases do not overwrite one another.

The default retrieval metric is `needle_token_multiset_v1`. For each head, count token
IDs in the located needle span with a Counter. A generated token is credited
only when that head's attention argmax points to an identical token in the
needle. Credits for each token ID are capped at its frequency in the needle,
independently per head. Score = credited tokens / needle-span length, in [0, 1].
Repeated hits to the same position can use that token's remaining quota.
Detection still averages scores only over successful answers (ROUGE recall > 50).
Run manifests and detection result metadata record `retrieval_metric` and
`attention_scope`. Previously saved scores use the old uncapped formula;
rerun detection to obtain the new metric. ROUGE and masking modes are unchanged.
To select the original uncapped formula, pass `--retrieval-metric legacy` to
`retrieval_head_detection.py` or `datasphere/job.py` (add it to YAML `cmd` for Jobs).
Legacy adds `1 / needle-span length` for every matching hit, including repeats;
it can exceed 1. The Qwen3 answer-only filter applies with either metric.
Implementations are separate classes in `experiment/scoring.py`:
`LegacyRetrievalScoreCollector` and `MultisetRetrievalScoreCollector`.
Detection selects one through `create_retrieval_collector(metric, ...)`;
the abstract `RetrievalScoreCollector` shares only hit detection and the interface.
Add a new implementation to `_RETRIEVAL_COLLECTORS` to expose another metric.

```python
import torch

trace = torch.load("detection/attention/detect-1_....pt")
first_generated_token = trace["steps"][0]["token_id"]
layer_3_attention = trace["steps"][0]["layers"][3]
```

## Masking

Run detection first, then run all three paired conditions:

```bash
python needle_in_haystack_with_mask.py --mask-topk 0 --lengths 1000,2000
python needle_in_haystack_with_mask.py --mask-topk 8 --lengths 1000,2000
python needle_in_haystack_with_mask.py --mask-topk -8 --seed 42 --lengths 1000,2000
```

The default `legacy_uniform` mode exactly preserves the source intervention:
it sets selected heads' logits to zero before softmax, producing uniform
attention. `zero_output` is available only as an explicit additional ablation.
As in the source experiment, a new random control set is drawn for every case;
the exact heads are stored in each result JSON.

## Adding another model

Implement `models/base.py::ModelAdapter`, add the model to
`models/registry.py`, and reuse the experiment, locator, scoring and storage
code unchanged. Models that support the Transformers `AttentionInterface` can
reuse the common observable eager backend.
