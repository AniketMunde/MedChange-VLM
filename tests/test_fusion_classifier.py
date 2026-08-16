import numpy as np

from medchange.fusion.classifier import (
    FusionClassifier,
)


def test_fusion_classifier():

    x = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            "absent",
            "absent",
            "new",
            "new",
        ]
    )

    classifier = (
        FusionClassifier(
            seed=42
        )
    )

    classifier.fit(
        x,
        y,
    )

    predictions = (
        classifier.predict(
            x
        )
    )

    assert (
        len(
            predictions
        )
        == 4
    )