from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_macro_metrics(
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute macro averages across evaluated findings.
    """

    evaluated = metrics[
        metrics["status"]
        == "evaluated"
    ].copy()

    if evaluated.empty:
        raise ValueError(
            "No evaluated findings available."
        )

    result: dict[
        str,
        Any,
    ] = {
        "num_findings": int(
            len(evaluated)
        )
    }

    for metric in [
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
    ]:
        values = pd.to_numeric(
            evaluated[
                metric
            ],
            errors="coerce",
        )

        values = values[
            values.notna()
        ]

        result[
            f"macro_{metric}"
        ] = (
            float(
                np.mean(
                    values
                )
            )
            if len(values)
            else None
        )

    return result