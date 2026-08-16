import numpy as np
import pytest
import torch

from medchange.models.temporal.features import (
    build_current_only_vector,
    build_longitudinal_vector,
    build_temporal_embedding_features,
)


def test_temporal_features():
    prior = torch.tensor(
        [
            [
                1.0,
                0.0,
                0.5,
            ]
        ]
    )

    current = torch.tensor(
        [
            [
                0.5,
                1.0,
                0.5,
            ]
        ]
    )

    features = (
        build_temporal_embedding_features(
            prior,
            current,
        )
    )

    assert np.allclose(
        features.delta,
        [
            -0.5,
            1.0,
            0.0,
        ],
    )

    assert np.allclose(
        features.absolute_delta,
        [
            0.5,
            1.0,
            0.0,
        ],
    )

    current_vector = (
        build_current_only_vector(
            features
        )
    )

    longitudinal = (
        build_longitudinal_vector(
            features
        )
    )

    assert (
        current_vector.shape[0]
        == 3
    )

    assert (
        longitudinal.shape[0]
        == 13
    )


def test_embedding_shape_mismatch():
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        build_temporal_embedding_features(
            torch.randn(
                1,
                4,
            ),
            torch.randn(
                1,
                5,
            ),
        )