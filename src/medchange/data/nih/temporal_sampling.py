from __future__ import annotations

import numpy as np
import pandas as pd


def sample_temporal_evaluation_subset(
    dataframe: pd.DataFrame,
    total_pairs: int = 1000,
    changed_fraction: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a reproducible pair-level evaluation subset.

    This balances:
        pairs with at least one NEW/RESOLVED finding
        pairs without NEW/RESOLVED findings

    It does not alter the per-pathology prevalence.
    """

    if dataframe.empty:
        raise ValueError(
            "Temporal dataframe is empty."
        )

    if total_pairs <= 0:
        raise ValueError(
            "total_pairs must be positive."
        )

    if not (
        0.0
        < changed_fraction
        < 1.0
    ):
        raise ValueError(
            "changed_fraction must "
            "be between 0 and 1."
        )

    changed = dataframe[
        dataframe[
            "has_temporal_change"
        ]
    ]

    unchanged = dataframe[
        ~dataframe[
            "has_temporal_change"
        ]
    ]

    target_changed = int(
        round(
            total_pairs
            * changed_fraction
        )
    )

    target_unchanged = (
        total_pairs
        - target_changed
    )

    if (
        len(changed)
        < target_changed
    ):
        raise ValueError(
            "Not enough changed pairs "
            "for requested subset."
        )

    if (
        len(unchanged)
        < target_unchanged
    ):
        raise ValueError(
            "Not enough unchanged pairs "
            "for requested subset."
        )

    changed_sample = (
        changed.sample(
            n=target_changed,
            random_state=seed,
            replace=False,
        )
    )

    unchanged_sample = (
        unchanged.sample(
            n=target_unchanged,
            random_state=(
                seed + 1
            ),
            replace=False,
        )
    )

    subset = pd.concat(
        [
            changed_sample,
            unchanged_sample,
        ],
        ignore_index=True,
    )

    rng = np.random.default_rng(
        seed
    )

    order = rng.permutation(
        len(
            subset
        )
    )

    subset = (
        subset.iloc[
            order
        ]
        .reset_index(
            drop=True
        )
    )

    return subset