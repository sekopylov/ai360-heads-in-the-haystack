# Retrieval heads on current Transformers

The code keeps the original Qwen3.5 checkpoint and separates four concerns:

```text
experiment/data     builds a context and inserts the needle
experiment/locator  finds the expected-answer span in prompt tokens
models/qwen35       runs the original Qwen model token by token
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

Install the appropriate PyTorch build first, then:

```bash
cd updated
python -m pip install -r requirements.txt
```

The checkpoint is downloaded automatically by Transformers on first use.
SDPA is the cross-platform prefill default. FlashAttention is an optional
Linux-only optimization and can be installed from `requirements-flash.txt`.

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

Small smoke run:

```bash
python retrieval_head_detection.py \
  --s 1000 --e 1000 --context-intervals 1 \
  --depths 50
```

The default `--capture top1` is sufficient for the original retrieval-head
metric. To retain every attention probability for every generated token:

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
`prompt_token_ids` and one entry per generated token. Each step stores
`token_id` and `layers[layer_index]`, shaped `[heads, key_length]`. The case ID
is included in every result, context and trace filename so the three detection
cases do not overwrite one another.

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
python compare_masking_results.py --topk 8
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
