import numpy as np

from medchange.models.temporal.ablation_features import (
    ABLATION_FEATURE_SETS,
    build_ablation_vector,
)
from medchange.models.temporal.features import (
    TemporalEmbeddingFeatures,
)


def build_features():
    return TemporalEmbeddingFeatures(
        prior=np.asarray(
            [
                1.0,
                2.0,
            ],
            dtype=np.float32,
        ),

        current=np.asarray(
            [
                3.0,
                5.0,
            ],
            dtype=np.float32,
        ),

        delta=np.asarray(
            [
                2.0,
                3.0,
            ],
            dtype=np.float32,
        ),

        absolute_delta=np.asarray(
            [
                2.0,
                3.0,
            ],
            dtype=np.float32,
        ),

        cosine_similarity=0.8,
    )


def test_all_ablation_feature_sets():
    features = build_features()

    expected_dimensions = {
        "current": 2,
        "prior_current": 4,
        "delta": 2,
        "abs_delta": 2,
        "current_delta": 4,
        "prior_current_delta": 6,
        "full": 9,
    }

    for feature_set in (
        ABLATION_FEATURE_SETS
    ):
        vector = (
            build_ablation_vector(
                features,
                feature_set,
            )
        )

        assert (
            vector.shape[0]
            == expected_dimensions[
                feature_set
            ]
        )


def test_delta_vector():
    vector = (
        build_ablation_vector(
            build_features(),
            "delta",
        )
    )

    assert np.allclose(
        vector,
        [
            2.0,
            3.0,
        ],
    )


def test_current_delta_vector():
    vector = (
        build_ablation_vector(
            build_features(),
            "current_delta",
        )
    )

    assert np.allclose(
        vector,
        [
            3.0,
            5.0,
            2.0,
            3.0,
        ],
    )