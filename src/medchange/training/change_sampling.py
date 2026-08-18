from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pandas as pd


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


PAIR_CATEGORIES = [
    "persistent",
    "new",
    "resolved",
    "stable",
]


def temporal_states_for_row(
    row: pd.Series,
) -> list[str]:

    return [
        str(
            row[
                f"{finding}_temporal"
            ]
        )
        .strip()
        .lower()

        for finding
        in TARGET_FINDINGS
    ]


def categorize_pair(
    row: pd.Series,
) -> str:
    """
    Assign one category per pair.

    Rarest/high-value temporal states are given
    priority for change-aware sampling.
    """

    states = set(
        temporal_states_for_row(
            row
        )
    )

    if "persistent" in states:
        return "persistent"

    if "new" in states:
        return "new"

    if "resolved" in states:
        return "resolved"

    return "stable"


def add_pair_category(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    output = (
        dataframe
        .copy()
    )

    output[
        "sampling_category"
    ] = output.apply(
        categorize_pair,
        axis=1,
    )

    return output


def sample_change_aware_pairs(
    dataframe: pd.DataFrame,
    *,
    max_samples: int,
    seed: int = 42,
    target_fractions: dict[
        str,
        float
    ] | None = None,
) -> pd.DataFrame:
    """
    Pair-level stratified sampling.

    Defaults:
      stable     35%
      new        25%
      resolved   20%
      persistent 20%

    Sampling may use replacement for rare change
    categories. Patient membership is NOT altered;
    this function should only be applied AFTER
    patient-disjoint splitting.
    """

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be > 0."
        )

    if target_fractions is None:
        target_fractions = {
            "stable": 0.35,
            "new": 0.25,
            "resolved": 0.20,
            "persistent": 0.20,
        }

    total_fraction = sum(
        target_fractions.values()
    )

    if abs(
        total_fraction
        - 1.0
    ) > 1e-6:
        raise ValueError(
            "target_fractions must sum to 1."
        )

    categorized = (
        add_pair_category(
            dataframe
        )
    )

    sampled_frames = []

    for index, category in enumerate(
        PAIR_CATEGORIES
    ):
        fraction = (
            target_fractions[
                category
            ]
        )

        requested = int(
            round(
                max_samples
                * fraction
            )
        )

        subset = (
            categorized[
                categorized[
                    "sampling_category"
                ]
                == category
            ]
        )

        if subset.empty:
            print(
                f"WARNING: no "
                f"{category} pairs available."
            )

            continue

        replace = (
            len(
                subset
            )
            < requested
        )

        sampled = subset.sample(
            n=requested,
            replace=replace,
            random_state=(
                seed
                + index
            ),
        )

        sampled_frames.append(
            sampled
        )

    if not sampled_frames:
        raise ValueError(
            "No samples available."
        )

    output = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    output = output.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    return output


def state_counts(
    dataframe: pd.DataFrame,
) -> dict[str, int]:

    counts = Counter()

    for _, row in (
        dataframe
        .iterrows()
    ):
        counts.update(
            temporal_states_for_row(
                row
            )
        )

    return {
        state: int(
            counts.get(
                state,
                0,
            )
        )

        for state in [
            "absent",
            "new",
            "persistent",
            "resolved",
        ]
    }


def category_counts(
    dataframe: pd.DataFrame,
) -> dict[str, int]:

    categorized = (
        add_pair_category(
            dataframe
        )
    )

    values = (
        categorized[
            "sampling_category"
        ]
        .value_counts()
        .to_dict()
    )

    return {
        category: int(
            values.get(
                category,
                0,
            )
        )

        for category
        in PAIR_CATEGORIES
    }