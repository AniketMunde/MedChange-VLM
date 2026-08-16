from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "finding",
    "feature_set",
    "macro_f1",
    "balanced_accuracy",
}


def load_seed_summary(
    path: str | Path,
    seed: int,
) -> pd.DataFrame:
    """
    Load one temporal-ablation summary and attach its seed.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Temporal ablation summary not found: {path}"
        )

    dataframe = pd.read_csv(path)

    missing = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Temporal ablation summary missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()
    dataframe["seed"] = int(seed)

    return dataframe


def load_all_seed_summaries(
    seed_paths: dict[int, str | Path],
) -> pd.DataFrame:
    """
    Combine all per-seed temporal ablation summaries.
    """

    if not seed_paths:
        raise ValueError(
            "At least one seed summary is required."
        )

    frames = []

    for seed, path in sorted(
        seed_paths.items()
    ):
        frames.append(
            load_seed_summary(
                path=path,
                seed=seed,
            )
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def aggregate_feature_performance(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate each finding/feature representation
    across patient-aware seeds.
    """

    grouped = (
        combined
        .groupby(
            [
                "finding",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            num_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    grouped[
        "macro_f1_std"
    ] = grouped[
        "macro_f1_std"
    ].fillna(0.0)

    grouped[
        "balanced_accuracy_std"
    ] = grouped[
        "balanced_accuracy_std"
    ].fillna(0.0)

    return grouped


def add_current_baseline_comparison(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add mean Macro-F1 difference against current-only features.
    """

    current = aggregate[
        aggregate[
            "feature_set"
        ]
        == "current"
    ][
        [
            "finding",
            "macro_f1_mean",
            "balanced_accuracy_mean",
        ]
    ].rename(
        columns={
            "macro_f1_mean":
                "current_macro_f1_mean",

            "balanced_accuracy_mean":
                "current_balanced_accuracy_mean",
        }
    )

    if current.empty:
        raise ValueError(
            "Current-only baseline is missing."
        )

    result = aggregate.merge(
        current,
        on="finding",
        how="left",
        validate="many_to_one",
    )

    result[
        "delta_macro_f1_vs_current"
    ] = (
        result[
            "macro_f1_mean"
        ]
        - result[
            "current_macro_f1_mean"
        ]
    )

    result[
        "delta_balanced_accuracy_vs_current"
    ] = (
        result[
            "balanced_accuracy_mean"
        ]
        - result[
            "current_balanced_accuracy_mean"
        ]
    )

    return result


def compute_seed_win_counts(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count how often each temporal feature set beats current-only
    Macro F1 for the same finding and seed.
    """

    current = combined[
        combined[
            "feature_set"
        ]
        == "current"
    ][
        [
            "finding",
            "seed",
            "macro_f1",
        ]
    ].rename(
        columns={
            "macro_f1":
                "current_macro_f1",
        }
    )

    comparison = combined.merge(
        current,
        on=[
            "finding",
            "seed",
        ],
        how="left",
        validate="many_to_one",
    )

    comparison[
        "beats_current"
    ] = (
        comparison[
            "macro_f1"
        ]
        > comparison[
            "current_macro_f1"
        ]
    )

    wins = (
        comparison
        .groupby(
            [
                "finding",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            wins_vs_current=(
                "beats_current",
                "sum",
            ),
            num_seed_comparisons=(
                "seed",
                "nunique",
            ),
        )
    )

    wins[
        "wins_vs_current"
    ] = wins[
        "wins_vs_current"
    ].astype(int)

    return wins


def build_final_temporal_summary(
    combined: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build:
    1. full feature-level aggregate
    2. one-row-per-finding final comparison
    """

    aggregate = (
        aggregate_feature_performance(
            combined
        )
    )

    aggregate = (
        add_current_baseline_comparison(
            aggregate
        )
    )

    wins = compute_seed_win_counts(
        combined
    )

    aggregate = aggregate.merge(
        wins,
        on=[
            "finding",
            "feature_set",
        ],
        how="left",
        validate="one_to_one",
    )

    final_rows = []

    for finding in sorted(
        aggregate[
            "finding"
        ].unique()
    ):
        subset = aggregate[
            aggregate[
                "finding"
            ]
            == finding
        ].copy()

        current = subset[
            subset[
                "feature_set"
            ]
            == "current"
        ]

        if current.empty:
            raise ValueError(
                f"Current baseline missing for {finding}."
            )

        temporal = subset[
            subset[
                "feature_set"
            ]
            != "current"
        ]

        if temporal.empty:
            raise ValueError(
                f"No temporal feature sets found for {finding}."
            )

        best_index = temporal[
            "macro_f1_mean"
        ].idxmax()

        best = temporal.loc[
            best_index
        ]

        current_row = current.iloc[
            0
        ]

        final_rows.append(
            {
                "finding": finding,

                "current_macro_f1_mean":
                    float(
                        current_row[
                            "macro_f1_mean"
                        ]
                    ),

                "current_macro_f1_std":
                    float(
                        current_row[
                            "macro_f1_std"
                        ]
                    ),

                "best_temporal_feature":
                    str(
                        best[
                            "feature_set"
                        ]
                    ),

                "best_temporal_macro_f1_mean":
                    float(
                        best[
                            "macro_f1_mean"
                        ]
                    ),

                "best_temporal_macro_f1_std":
                    float(
                        best[
                            "macro_f1_std"
                        ]
                    ),

                "delta_macro_f1":
                    float(
                        best[
                            "delta_macro_f1_vs_current"
                        ]
                    ),

                "temporal_wins_vs_current":
                    int(
                        best[
                            "wins_vs_current"
                        ]
                    ),

                "num_seeds":
                    int(
                        best[
                            "num_seed_comparisons"
                        ]
                    ),

                "current_balanced_accuracy_mean":
                    float(
                        current_row[
                            "balanced_accuracy_mean"
                        ]
                    ),

                "best_temporal_balanced_accuracy_mean":
                    float(
                        best[
                            "balanced_accuracy_mean"
                        ]
                    ),
            }
        )

    final_summary = pd.DataFrame(
        final_rows
    )

    return (
        aggregate,
        final_summary,
    )