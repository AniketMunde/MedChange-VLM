import numpy as np
import pytest

from medchange.evaluation.labels import (
    LabelState,
    classify_label,
    prepare_binary_labels,
)


def test_positive_label():
    assert (
        classify_label(1)
        == LabelState.POSITIVE
    )


def test_negative_label():
    assert (
        classify_label(0)
        == LabelState.NEGATIVE
    )


def test_uncertain_label():
    assert (
        classify_label(-1)
        == LabelState.UNCERTAIN
    )


def test_missing_label():
    assert (
        classify_label(np.nan)
        == LabelState.MISSING
    )


def test_invalid_label():
    with pytest.raises(
        ValueError,
    ):
        classify_label(2)


def test_prepare_binary_labels():
    labels, mask = (
        prepare_binary_labels(
            [
                1,
                0,
                -1,
                np.nan,
                1,
            ]
        )
    )

    assert labels.tolist() == [
        1,
        0,
        1,
    ]

    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]