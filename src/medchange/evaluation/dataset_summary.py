from __future__ import annotations

from typing import Any

import pandas as pd


def build_dataset_summary(
    dataframe: pd.DataFrame,
    findings: list[str],
) -> dict[str, Any]:
    """
    Summarize an evaluated CXR subset.
    """

    if dataframe.empty:
        raise ValueError(
            "Cannot summarize an empty dataframe."
        )

    summary: dict[str, Any] = {
        "num_images": int(
            len(dataframe)
        ),

        "num_unique_patients": int(
            dataframe[
                "patient_id"
            ].nunique()
        ),

        "views": {
            str(key): int(value)
            for key, value
            in dataframe[
                "view_position"
            ]
            .fillna("UNKNOWN")
            .value_counts()
            .items()
        },

        "findings": {},
    }

    for finding in findings:
        column = (
            f"{finding}_label"
        )

        positives = int(
            dataframe[
                column
            ].sum()
        )

        negatives = int(
            len(dataframe)
            - positives
        )

        summary[
            "findings"
        ][finding] = {
            "positive": positives,
            "negative": negatives,
            "prevalence": (
                positives
                / len(dataframe)
            ),
        }

    return summary