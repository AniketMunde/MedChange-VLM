import pytest

from medchange.data.nih.streaming import (
    iter_nih_examples,
)


def test_invalid_view_fails_before_stream():
    generator = iter_nih_examples(
        split="train",
        max_samples=1,
        view="LATERAL",
    )

    with pytest.raises(
        ValueError,
        match="view must be",
    ):
        next(generator)


def test_invalid_max_samples_fails_before_stream():
    generator = iter_nih_examples(
        split="train",
        max_samples=0,
    )

    with pytest.raises(
        ValueError,
        match="max_samples must be positive",
    ):
        next(generator)