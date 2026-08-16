import pandas as pd
import pytest

from medchange.data.nih.temporal_sampling import (
    sample_temporal_evaluation_subset,
)


def test_balanced_temporal_subset():
    dataframe = pd.DataFrame(
        {
            "pair_id": [
                f"pair-{index}"
                for index in range(
                    20
                )
            ],

            "has_temporal_change": (
                [True] * 10
                + [False] * 10
            ),
        }
    )

    subset = (
        sample_temporal_evaluation_subset(
            dataframe=dataframe,
            total_pairs=10,
            changed_fraction=0.5,
            seed=42,
        )
    )

    assert len(
        subset
    ) == 10

    assert int(
        subset[
            "has_temporal_change"
        ].sum()
    ) == 5


def test_temporal_sampling_reproducible():
    dataframe = pd.DataFrame(
        {
            "pair_id": [
                f"pair-{index}"
                for index in range(
                    40
                )
            ],

            "has_temporal_change": (
                [True] * 20
                + [False] * 20
            ),
        }
    )

    first = (
        sample_temporal_evaluation_subset(
            dataframe,
            total_pairs=20,
            seed=42,
        )
    )

    second = (
        sample_temporal_evaluation_subset(
            dataframe,
            total_pairs=20,
            seed=42,
        )
    )

    assert (
        first[
            "pair_id"
        ].tolist()
        == second[
            "pair_id"
        ].tolist()
    )


def test_invalid_temporal_sample_size():
    dataframe = pd.DataFrame(
        {
            "has_temporal_change": [
                True,
                False,
            ]
        }
    )

    with pytest.raises(
        ValueError
    ):
        sample_temporal_evaluation_subset(
            dataframe,
            total_pairs=100,
        )