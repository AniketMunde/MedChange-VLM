from __future__ import annotations

import numpy as np

from medchange.models.temporal.features import (
    TemporalEmbeddingFeatures,
)


ABLATION_FEATURE_SETS = [
    "current",
    "prior_current",
    "delta",
    "abs_delta",
    "current_delta",
    "prior_current_delta",
    "full",
]


def build_ablation_vector(
    features: TemporalEmbeddingFeatures,
    feature_set: str,
) -> np.ndarray:
    """
    Construct one ablation feature representation.
    """

    if feature_set not in ABLATION_FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    prior = features.prior.astype(
        np.float32
    )

    current = features.current.astype(
        np.float32
    )

    delta = features.delta.astype(
        np.float32
    )

    abs_delta = (
        features.absolute_delta.astype(
            np.float32
        )
    )

    cosine = np.asarray(
        [
            features.cosine_similarity
        ],
        dtype=np.float32,
    )

    if feature_set == "current":
        return current

    if feature_set == "prior_current":
        return np.concatenate(
            [
                prior,
                current,
            ]
        )

    if feature_set == "delta":
        return delta

    if feature_set == "abs_delta":
        return abs_delta

    if feature_set == "current_delta":
        return np.concatenate(
            [
                current,
                delta,
            ]
        )

    if feature_set == "prior_current_delta":
        return np.concatenate(
            [
                prior,
                current,
                delta,
            ]
        )

    return np.concatenate(
        [
            prior,
            current,
            delta,
            abs_delta,
            cosine,
        ]
    )