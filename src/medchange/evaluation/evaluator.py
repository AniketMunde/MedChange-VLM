from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from medchange.evaluation.classification import (
    compute_binary_metrics,
)


def evaluate_predictions(
    dataframe: pd.DataFrame,
    labels: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate pathology scores stored in a dataframe.

    Labels with no valid binary ground-truth values after removing
    uncertain (-1) and missing labels are retained in the output
    with status='skipped' instead of causing the evaluation to fail.
    """

    results = []

    for label in labels:
        label_column = f"{label}_label"
        score_column = f"{label}_score"

        if label_column not in dataframe.columns:
            raise ValueError(
                f"Missing column: {label_column}"
            )

        if score_column not in dataframe.columns:
            raise ValueError(
                f"Missing column: {score_column}"
            )

        try:
            metrics = compute_binary_metrics(
                y_true=dataframe[label_column].values,
                y_score=dataframe[score_column].values,
                threshold=threshold,
            )

            results.append(
                {
                    "finding": label,
                    "status": "evaluated",
                    **asdict(metrics),
                }
            )

        except ValueError as exc:
            if (
                "No evaluable binary labels remain"
                not in str(exc)
            ):
                raise

            n_total = len(
                dataframe[label_column]
            )

            results.append(
                {
                    "finding": label,
                    "status": "skipped",
                    "auroc": None,
                    "auprc": None,
                    "f1": None,
                    "precision": None,
                    "recall": None,
                    "threshold": threshold,
                    "n_total": n_total,
                    "n_evaluated": 0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "n_excluded": n_total,
                }
            )

    return pd.DataFrame(results)