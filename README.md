# Heads in the Haystack

An AI360 student project studying how language models retrieve information from long contexts.

**Needle-in-a-Haystack (NIAH)** places a fact—the *needle*—inside a long distracting text—the *haystack*—and asks the model to recover it. We inspect attention heads and selectively disable them to investigate which components contribute to retrieval.

**Poster:** [Team 14 — final A0 poster (PDF)](poster/heads-in-the-haystack.pdf).

## Research directions

1. **Reproduce the foundational study.** Reproduce the head-detection and ablation method from [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) on Qwen1.5-14B-Chat and Qwen3-8B.

2. **Compare head-detection methods.** Compare the paper’s retrieval score, attention mass, and [query-focused scores (QRScore)](https://aclanthology.org/2025.emnlp-main.1214/). Test which rankings identify heads whose masking most harms retrieval in GLM-4.6V-Flash and Ministral 3 8B.

3. **Make retrieval harder.** Add noise and distractors, reformulate questions, and test whether heads identified on simple NIAH remain important. Distinguish loss of key facts from changes in answer wording. Related benchmarks include [RULER](https://arxiv.org/abs/2404.06654) and [NoLiMa](https://arxiv.org/abs/2502.05167).

4. **Compare attention architectures.** Study NIAH with standard, linear, and hybrid attention. Compare full and linear attention within Qwen3.5-0.8B and investigate retrieval in [Mamba-2](https://arxiv.org/abs/2405.21060) 2.7B.

5. **Explore context extension.** Compare native RoPE with [YaRN](https://arxiv.org/abs/2309.00071) on Qwen2.5-7B-Instruct, within and beyond its native context window.
