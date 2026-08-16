from __future__ import annotations

import pandas as pd

from medchange.evaluation.evaluator import (
    evaluate_predictions,
)


def evaluate_by_view(
    dataframe: pd.DataFrame,
    labels: list[str],
    thresholds: dict[str, float]
    | None = None,
) -> pd.DataFrame:
    """
    Evaluate separately for AP and PA chest radiographs.
    """

    if (
        "view_position"
        not in dataframe.columns
    ):
        raise ValueError(
            "Missing view_position column."
        )

    results = []

    for view in [
        "AP",
        "PA",
    ]:
        subset = dataframe[
            dataframe[
                "view_position"
            ]
            == view
        ]

        if subset.empty:
            continue

        for label in labels:
            threshold = (
                thresholds.get(
                    label,
                    0.5,
                )
                if thresholds
                else 0.5
            )

            metrics = (
                evaluate_predictions(
                    dataframe=subset,
                    labels=[
                        label
                    ],
                    threshold=threshold,
                )
            )

            row = (
                metrics.iloc[
                    0
                ].to_dict()
            )

            row[
                "view_position"
            ] = view

            row[
                "n_view_samples"
            ] = len(
                subset
            )

            results.append(
                row
            )

    return pd.DataFrame(
        results
    )