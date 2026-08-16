import numpy as np

from medchange.fusion.encoding import (
    build_qwen_feature_vector,
    encode_qwen_state,
)


def test_qwen_state_encoding():

    vector = encode_qwen_state(
        "resolved"
    )

    assert (
        vector.shape[0]
        == 5
    )

    assert (
        vector.sum()
        == 1.0
    )


def test_qwen_unknown_state():

    vector = encode_qwen_state(
        "something_invalid"
    )

    # uncertain is final position
    assert (
        vector[-1]
        == 1.0
    )


def test_qwen_feature_vector():

    vector = (
        build_qwen_feature_vector(
            state="new",
            confidence=0.82,
        )
    )

    assert (
        vector.shape[0]
        == 6
    )

    assert np.isclose(
        vector[-1],
        0.82,
    )