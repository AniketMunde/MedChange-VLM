import pandas as pd
import pytest

from medchange.evaluation.temporal_aggregate import (
    aggregate_feature_performance,
    build_final_temporal_summary,
    compute_seed_win_counts,
)


def build_test_dataframe():
    return pd.DataFrame(
        {
            "finding": [
                "atelectasis",
                "atelectasis",
                "atelectasis",
                "atelectasis",
                "atelectasis",
                "atelectasis",
            ],

            "feature_set": [
                "current",
                "current_delta",
                "full",
                "current",
                "current_delta",
                "full",
            ],

            "macro_f1": [
                0.20,
                0.30,
                0.25,
                0.22,
                0.32,
                0.24,
            ],

            "balanced_accuracy": [
                0.30,
                0.40,
                0.35,
                0.32,
                0.42,
                0.34,
            ],

            "seed": [
                11,
                11,
                11,
                21,
                21,
                21,
            ],
        }
    )


def test_aggregate_feature_performance():
    dataframe = (
        build_test_dataframe()
    )

    result = (
        aggregate_feature_performance(
            dataframe
        )
    )

    current = result[
        result[
            "feature_set"
        ]
        == "current"
    ].iloc[0]

    assert (
        current[
            "macro_f1_mean"
        ]
        == pytest.approx(
            0.21
        )
    )

    assert (
        current[
            "num_seeds"
        ]
        == 2
    )


def test_seed_win_counts():
    dataframe = (
        build_test_dataframe()
    )

    wins = compute_seed_win_counts(
        dataframe
    )

    current_delta = wins[
        wins[
            "feature_set"
        ]
        == "current_delta"
    ].iloc[0]

    assert (
        current_delta[
            "wins_vs_current"
        ]
        == 2
    )


def test_final_temporal_summary():
    dataframe = (
        build_test_dataframe()
    )

    (
        _,
        summary,
    ) = build_final_temporal_summary(
        dataframe
    )

    row = summary.iloc[0]

    assert (
        row[
            "best_temporal_feature"
        ]
        == "current_delta"
    )

    assert (
        row[
            "current_macro_f1_mean"
        ]
        == pytest.approx(
            0.21
        )
    )

    assert (
        row[
            "best_temporal_macro_f1_mean"
        ]
        == pytest.approx(
            0.31
        )
    )

    assert (
        row[
            "delta_macro_f1"
        ]
        == pytest.approx(
            0.10
        )
    )

    assert (
        row[
            "temporal_wins_vs_current"
        ]
        == 2
    )