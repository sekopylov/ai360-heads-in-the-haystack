# Query-focused retrieval score

Metric key: `query_retrieval_score_v1`.
Source: [QRHead, EMNLP2025](https://aclanthology.org/2025.emnlp-main.1214.pdf),
§3.1–3.2, Eq.3–4. For each eligible Q head:

`score = mean_{question tokens} sum_{gold tokens} softmax(QK * scale + causal/mask)`.

In NIAH the gold document is the **entire inserted needle** (`case.needle`),
not necessarily `expected_answer`, which remains the target for ROUGE and
legacy decode metrics. Question tokens exclude template instructions and
the generated answer/thinking. Token spans include tokens overlapping the
question/gold character boundaries (e.g. a merged leading space).
The full softmax denominator includes all causally visible prompt keys.
There is no division by gold length, hit test or answer success filter.
Null-query calibration is a QRRetriever document-ranking technique (§4.1),
not the head-detection QRScore, and is not applied here.

## Enable

Add the key to the existing detection command, for example:

```bash
python3 retrieval_head_detection.py \
  --adapter qwen3_8b_no_thinking --model /content/models/Qwen3-8B \
  --dtype float16 --device-map cuda:0 --prefill-attention sdpa_flash \
  --lengths 1000 --depths 50 --max-new-tokens 64 --context-seed 42 \
  --retrieval-metrics query_retrieval_score_v1,needle_token_multiset_v1,needle_attention_mass_v1 \
  --output-root /content/drive/MyDrive/retrieval-heads-colab/qwen3-qrscore-smoke

python3 aggregate_head_scores.py \
  --run /content/drive/MyDrive/retrieval-heads-colab/qwen3-qrscore-smoke --all-cases
```

Run from `updated/`. QR-only selection also works:
`--retrieval-metric query_retrieval_score_v1`. The current detection CLI still
generates an answer for its normal result/ROUGE schema; that generation does
not contribute to QRScore. Aggregation requires `--all-cases` to implement
Eq.4 without silently restricting the sample to successful answers. It writes
`detection/aggregation/head_scores_query_retrieval_score_v1.json`, usable by
the existing ranking/masking pipeline. Mixed aggregation with `--all-cases`
also averages the other metrics over all cases; their score formulas do not change.

## Implementation and limitations

- Enabled QRScore runs one additional prompt-only forward (`use_cache=False`),
  delegates attention output to the original kernel, and restores the backend
  in a finally block. No edits to the cached decode engine or historical prompts.
- FP32 scoring recomputes only question rows in chunks of16. It never requests
  full `N×N` attention. Causal/additive/boolean masks and GQA are handled.
- Supports the project's `sdpa`, `sdpa_flash`, `sdpa_memory_efficient` prefill
  backends, single unpadded text prompts, eval mode, all eligible full-attention
  layers. Other backends, dropout and logit soft-capping fail explicitly.
- Exact decoded question uses its last occurrence; gold needle must occur
  exactly once before it. Stable prefix decoding is required for slow
  tokenizers; unsupported decoding fails instead of guessing evidence spans.
- Save query/gold token spans in each result and the definition in run.json.
  Neither `answer_only` nor `all_decode_tokens` changes this prefill metric.
- Old runs cannot acquire QRScore from stored decode summaries. GPU execution
  has not yet been tested for this new metric; CPU synthetic/tiny-model tests
  cover the formula, masks, GQA, integration, restoration and generation parity.
