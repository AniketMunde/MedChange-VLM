from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TemporalEmbeddingFeatures:
    prior: np.ndarray
    current: np.ndarray
    delta: np.ndarray
    absolute_delta: np.ndarray
    cosine_similarity: float


def _to_numpy(
    embedding: torch.Tensor,
) -> np.ndarray:

    return (
        embedding
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )


def build_temporal_embedding_features(
    prior_embedding: torch.Tensor,
    current_embedding: torch.Tensor,
) -> TemporalEmbeddingFeatures:

    prior = _to_numpy(
        prior_embedding
    )

    current = _to_numpy(
        current_embedding
    )

    if (
        prior.shape
        != current.shape
    ):
        raise ValueError(
            "Prior and current embeddings "
            "must have identical shapes."
        )

    delta = (
        current
        - prior
    )

    absolute_delta = (
        np.abs(
            delta
        )
    )

    denominator = (
        np.linalg.norm(
            prior
        )
        * np.linalg.norm(
            current
        )
    )

    cosine_similarity = (
        float(
            np.dot(
                prior,
                current,
            )
            / denominator
        )
        if denominator > 0
        else 0.0
    )

    return TemporalEmbeddingFeatures(
        prior=prior,
        current=current,
        delta=delta,
        absolute_delta=(
            absolute_delta
        ),
        cosine_similarity=(
            cosine_similarity
        ),
    )


def build_current_only_vector(
    features: TemporalEmbeddingFeatures,
) -> np.ndarray:

    return (
        features.current
        .astype(
            np.float32
        )
    )


def build_longitudinal_vector(
    features: TemporalEmbeddingFeatures,
) -> np.ndarray:

    cosine = np.asarray(
        [
            features.cosine_similarity
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            features.prior,
            features.current,
            features.delta,
            features.absolute_delta,
            cosine,
        ]
    ).astype(
        np.float32
    )