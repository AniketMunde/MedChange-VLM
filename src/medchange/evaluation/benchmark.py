from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from medchange.evaluation.bootstrap import (
    bootstrap_metric,
)
from medchange.evaluation.calibration import (
    find_best_f1_threshold,
)
from medchange.evaluation.evaluator import (
    evaluate_predictions,
)


def calibrate_thresholds(
    development: pd.DataFrame,
    findings: list[str],
) -> pd.DataFrame:
    """
    Calibrate one F1-optimal threshold per finding
    using development data only.
    """

    results = []

    for finding in findings:
        label_column = (
            f"{finding}_label"
        )

        score_column = (
            f"{finding}_score"
        )

        result = (
            find_best_f1_threshold(
                y_true=development[
                    label_column
                ].values,
                y_score=development[
                    score_column
                ].values,
            )
        )

        results.append(
            {
                "finding": finding,
                **asdict(
                    result
                ),
            }
        )

    return pd.DataFrame(
        results
    )


def evaluate_test_with_thresholds(
    test: pd.DataFrame,
    threshold_table: pd.DataFrame,
    findings: list[str],
) -> pd.DataFrame:
    """
    Evaluate test data using thresholds learned only
    from development data.
    """

    rows = []

    threshold_map = {
        row[
            "finding"
        ]: float(
            row[
                "threshold"
            ]
        )
        for _, row
        in threshold_table.iterrows()
    }

    for finding in findings:
        result = (
            evaluate_predictions(
                dataframe=test,
                labels=[
                    finding
                ],
                threshold=threshold_map[
                    finding
                ],
            )
        )

        rows.append(
            result.iloc[
                0
            ].to_dict()
        )

    return pd.DataFrame(
        rows
    )


def add_confidence_intervals(
    metrics: pd.DataFrame,
    test: pd.DataFrame,
    findings: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add AUROC and AUPRC bootstrap confidence intervals.
    """

    output = metrics.copy()

    output[
        "auroc_ci_lower"
    ] = None

    output[
        "auroc_ci_upper"
    ] = None

    output[
        "auprc_ci_lower"
    ] = None

    output[
        "auprc_ci_upper"
    ] = None

    for finding in findings:
        mask = (
            output[
                "finding"
            ]
            == finding
        )

        if not mask.any():
            continue

        y_true = test[
            f"{finding}_label"
        ].to_numpy(
            dtype=int
        )

        y_score = test[
            f"{finding}_score"
        ].to_numpy(
            dtype=float
        )

        if (
            len(set(
                y_true.tolist()
            ))
            < 2
        ):
            continue

        auroc_ci = (
            bootstrap_metric(
                y_true=y_true,
                y_score=y_score,
                metric_fn=(
                    roc_auc_score
                ),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

        auprc_ci = (
            bootstrap_metric(
                y_true=y_true,
                y_score=y_score,
                metric_fn=(
                    average_precision_score
                ),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

        output.loc[
            mask,
            "auroc_ci_lower",
        ] = auroc_ci.lower

        output.loc[
            mask,
            "auroc_ci_upper",
        ] = auroc_ci.upper

        output.loc[
            mask,
            "auprc_ci_lower",
        ] = auprc_ci.lower

        output.loc[
            mask,
            "auprc_ci_upper",
        ] = auprc_ci.upper

    return output