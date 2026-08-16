from __future__ import annotations

import numpy as np


QWEN_TEMPORAL_STATES = [
    "absent",
    "new",
    "persistent",
    "resolved",
    "uncertain",
]


def encode_qwen_state(
    state: str,
) -> np.ndarray:
    """
    One-hot encode Qwen temporal state.
    """

    normalized = str(
        state
    ).strip().lower()

    vector = np.zeros(
        len(
            QWEN_TEMPORAL_STATES
        ),
        dtype=np.float32,
    )

    if normalized not in (
        QWEN_TEMPORAL_STATES
    ):
        normalized = "uncertain"

    index = (
        QWEN_TEMPORAL_STATES
        .index(
            normalized
        )
    )

    vector[index] = 1.0

    return vector


def build_qwen_feature_vector(
    state: str,
    confidence: float,
) -> np.ndarray:
    """
    Qwen feature vector:

    5-dimensional state one-hot
    +
    confidence

    total = 6 dimensions.
    """

    state_vector = (
        encode_qwen_state(
            state
        )
    )

    confidence_vector = np.asarray(
        [
            float(
                confidence
            )
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            state_vector,
            confidence_vector,
        ]
    )