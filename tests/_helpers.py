"""Shared helpers for the regression suites.

Moved verbatim out of `test_regressions.py` when it was split by topic; these four
were the only module-level objects several of the new files needed from each other.
"""

from __future__ import annotations

import torch

from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import RetrievalScores


def attention_info(num_layers: int, heads: int, name: str = "toy") -> ModelInfo:
    return ModelInfo(
        name=name, path=name, model_type="toy", num_layers=num_layers,
        layer_types=["full_attention"] * num_layers,
        num_heads={i: heads for i in range(num_layers)},
        num_kv_heads={i: heads for i in range(num_layers)},
        head_dim=4, hidden_size=8, max_position_embeddings=128,
        scoreable_layers_=list(range(num_layers)),
    )


class _Args:
    def __init__(self, k=None, k_frac=None):
        self.k = k
        self.k_frac = k_frac


def scores_with(values: list[float], *, threshold: float = 0.1) -> RetrievalScores:
    """A single-layer score matrix, one value per head."""
    info = attention_info(1, len(values))
    score = torch.tensor([values], dtype=torch.float32)
    return RetrievalScores(
        info=info, score=score, activation_freq=torch.zeros_like(score),
        n_instances=1, threshold=threshold,
    )


def _well_formed_curve() -> dict:
    return {"m": {"k_values": [1, 2], "k_effective": [1, 2], "retrieval": [90.0, 80.0],
                  "retrieval_std": [3.0, 4.0], "random_mean": [95.0, 94.0],
                  "random_std": [1.0, 1.5], "baseline": 97.0,
                  "retrieval_exact_match": [80.0, 70.0], "retrieval_exact_std": [5.0, 6.0],
                  "random_exact_match_mean": [90.0, 89.0], "retrieval_recall": [70.0, 60.0],
                  "random_recall_mean": [88.0, 87.0], "baseline_recall": 92.0}}
