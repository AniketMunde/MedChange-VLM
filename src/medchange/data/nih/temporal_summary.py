from __future__ import annotations

from typing import Any

import pandas as pd

from medchange.data.nih.temporal_targets import (
    NIH_TEMPORAL_FINDINGS,
    TemporalTarget,
)


def summarize_temporal_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    if dataframe.empty:
        raise ValueError(
            "Temporal dataframe is empty."
        )

    summary: dict[
        str,
        Any,
    ] = {
        "num_pairs": int(
            len(
                dataframe
            )
        ),

        "num_patients": int(
            dataframe[
                "patient_id"
            ].nunique()
        ),

        "pairs_with_change": int(
            dataframe[
                "has_temporal_change"
            ].sum()
        ),

        "pairs_without_change": int(
            (
                ~dataframe[
                    "has_temporal_change"
                ]
            ).sum()
        ),

        "mean_changed_findings": float(
            dataframe[
                "num_changed_findings"
            ].mean()
        ),

        "findings": {},
    }

    for medchange_finding in (
        NIH_TEMPORAL_FINDINGS.values()
    ):

        column = (
            f"{medchange_finding}_temporal"
        )

        counts = (
            dataframe[
                column
            ]
            .value_counts()
            .to_dict()
        )

        summary[
            "findings"
        ][
            medchange_finding
        ] = {
            state.value: int(
                counts.get(
                    state.value,
                    0,
                )
            )
            for state
            in TemporalTarget
        }

    return summary