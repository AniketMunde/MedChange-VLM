from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from medchange.evaluation.selective_metrics import (
    compute_selective_metrics,
)
from medchange.safety.policies import (
    apply_tuned_policy,
)


POLICIES = [
    "strict",
    "confidence_margin",
    "change_sensitive",
    "low_confidence_only",
]


THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


def evaluate_policy_configuration(
    dataframe: pd.DataFrame,
    *,
    policy: str,
    threshold: float,
) -> dict[str, Any]:

    predictions = []
    reviews = []

    for _, row in (
        dataframe.iterrows()
    ):
        decision = (
            apply_tuned_policy(
                policy=policy,

                biomedclip_state=(
                    row[
                        "biomedclip_state"
                    ]
                ),

                qwen_state=(
                    row[
                        "qwen_state"
                    ]
                ),

                biomedclip_confidence=(
                    row[
                        "biomedclip_confidence"
                    ]
                ),

                threshold=threshold,
            )
        )

        predictions.append(
            decision.final_state
        )

        reviews.append(
            decision.requires_review
        )

    metrics = (
        compute_selective_metrics(
            y_true=(
                dataframe[
                    "ground_truth"
                ]
                .astype(str)
                .to_numpy()
            ),

            y_pred=np.asarray(
                predictions,
                dtype=str,
            ),

            requires_review=np.asarray(
                reviews,
                dtype=bool,
            ),
        )
    )

    return metrics


def run_policy_grid(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for seed in sorted(
        predictions[
            "seed"
        ].unique()
    ):

        subset = predictions[
            predictions[
                "seed"
            ]
            == seed
        ]

        for policy in POLICIES:

            for threshold in (
                THRESHOLDS
            ):

                metrics = (
                    evaluate_policy_configuration(
                        subset,
                        policy=policy,
                        threshold=threshold,
                    )
                )

                rows.append(
                    {
                        "seed": (
                            int(
                                seed
                            )
                        ),

                        "policy": (
                            policy
                        ),

                        "threshold": (
                            threshold
                        ),

                        "coverage": (
                            metrics[
                                "coverage"
                            ]
                        ),

                        "abstention_rate": (
                            metrics[
                                "abstention_rate"
                            ]
                        ),

                        "selective_accuracy": (
                            metrics[
                                "selective_accuracy"
                            ]
                        ),

                        "selective_macro_f1": (
                            metrics[
                                "selective_macro_f1"
                            ]
                        ),

                        "error_rate_on_covered": (
                            metrics[
                                "error_rate_on_covered"
                            ]
                        ),

                        "review_rate": (
                            metrics[
                                "review_rate"
                            ]
                        ),

                        "absent_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "absent"
                            ]
                        ),

                        "new_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "new"
                            ]
                        ),

                        "persistent_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "persistent"
                            ]
                        ),

                        "resolved_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "resolved"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def aggregate_policy_grid(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            [
                "policy",
                "threshold",
            ],
            as_index=False,
        )
        .agg(
            coverage_mean=(
                "coverage",
                "mean",
            ),

            coverage_std=(
                "coverage",
                "std",
            ),

            selective_accuracy_mean=(
                "selective_accuracy",
                "mean",
            ),

            selective_accuracy_std=(
                "selective_accuracy",
                "std",
            ),

            selective_macro_f1_mean=(
                "selective_macro_f1",
                "mean",
            ),

            selective_macro_f1_std=(
                "selective_macro_f1",
                "std",
            ),

            error_rate_mean=(
                "error_rate_on_covered",
                "mean",
            ),

            error_rate_std=(
                "error_rate_on_covered",
                "std",
            ),

            review_rate_mean=(
                "review_rate",
                "mean",
            ),

            new_recall_mean=(
                "new_recall",
                "mean",
            ),

            persistent_recall_mean=(
                "persistent_recall",
                "mean",
            ),

            resolved_recall_mean=(
                "resolved_recall",
                "mean",
            ),

            absent_recall_mean=(
                "absent_recall",
                "mean",
            ),
        )
    )