"""Retrieval heads: a clean re-implementation of

    "Retrieval Head Mechanistically Explains Long-Context Factuality"
    Wu, Wang, Xiao, Peng, Fu -- https://github.com/nightdessert/Retrieval_Head

The package is deliberately architecture-agnostic.  The paper's central object is
the *retrieval score* of an attention head, which is only defined for heads that
materialise a distribution over the context (a softmax attention map).  We call
those **scoreable** heads and discover them by introspection, so the same code
runs on a plain dense transformer and on a hybrid linear/full-attention model
such as Qwen3.5 (see :mod:`retrieval_heads.models`).
"""

from retrieval_heads.models import (
    HeadRef,
    ModelInfo,
    load_model,
    describe_model,
)
from retrieval_heads.haystack import (
    NeedleSample,
    HaystackBuilder,
    build_needle_sample,
)
from retrieval_heads.attention import (
    AttentionRecorder,
    HeadMasker,
    TokenMixerMasker,
    masked_heads,
    masked_token_mixers,
)
from retrieval_heads.scoring import (
    RetrievalScores,
    score_instance,
    aggregate_scores,
)
from retrieval_heads.detection import (
    DetectionConfig,
    run_detection,
)

__all__ = [
    "HeadRef",
    "ModelInfo",
    "load_model",
    "describe_model",
    "NeedleSample",
    "HaystackBuilder",
    "build_needle_sample",
    "AttentionRecorder",
    "HeadMasker",
    "TokenMixerMasker",
    "masked_heads",
    "masked_token_mixers",
    "RetrievalScores",
    "score_instance",
    "aggregate_scores",
    "DetectionConfig",
    "run_detection",
]

__version__ = "0.1.0"
