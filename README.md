# Heads in the Haystack

An AI360 student project studying how language models retrieve information from long contexts.

**Needle-in-a-Haystack (NIAH)** places a fact—the *needle*—inside a long distracting text—the *haystack*—and asks the model to recover it. We inspect attention heads and selectively disable them to investigate which components contribute to retrieval.

## Research directions

1. **Reproduce the foundational study.** Reproduce the head-detection and ablation experiments from [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574), the starting point for our project.

2. **Improve head detection.** Develop and compare methods for identifying heads important for NIAH, building on the original paper and related approaches such as [query-focused retrieval heads](https://aclanthology.org/2025.emnlp-main.1214/).

3. **Compare attention architectures.** Study NIAH in models with standard attention, linear attention, and hybrid architectures, including retrieval mechanisms in [Mamba-2](https://arxiv.org/abs/2405.21060).

4. **Make retrieval harder.** Add noise and distractors, reformulate questions, and test whether heads identified on simple NIAH remain important. Related benchmarks include [RULER](https://arxiv.org/abs/2404.06654) and [NoLiMa](https://arxiv.org/abs/2502.05167).

5. **Explore context extension.** Study how [YaRN](https://arxiv.org/abs/2309.00071) and other context-extension methods affect NIAH beyond a model’s native context window, including their impact on shorter contexts.
