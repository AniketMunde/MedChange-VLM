from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.metrics import (
    f1_score,
    recall_score,
)


TEMPORAL_STATES = [
    "absent",
    "new",
    "persistent",
    "resolved",
]


def compute_selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    requires_review: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Evaluate predictions where "uncertain" represents abstention.

    Metrics:
    - coverage
    - abstention rate
    - selective accuracy
    - selective Macro F1
    - error rate among non-abstained predictions
    - review rate
    - per-state recall among covered predictions
    """

    y_true = np.asarray(
        y_true,
        dtype=str,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=str,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty prediction set."
        )

    abstained = (
        y_pred == "uncertain"
    )

    covered = ~abstained

    coverage = float(
        covered.mean()
    )

    abstention_rate = float(
        abstained.mean()
    )

    covered_count = int(
        covered.sum()
    )

    total_count = int(
        len(y_true)
    )

    if covered_count > 0:
        covered_true = (
            y_true[
                covered
            ]
        )

        covered_pred = (
            y_pred[
                covered
            ]
        )

        selective_accuracy = float(
            np.mean(
                covered_true
                == covered_pred
            )
        )

        error_rate = float(
            1.0
            - selective_accuracy
        )

        selective_macro_f1 = float(
            f1_score(
                covered_true,
                covered_pred,
                labels=TEMPORAL_STATES,
                average="macro",
                zero_division=0,
            )
        )

        state_recall = {}

        for state in TEMPORAL_STATES:
            positives = (
                covered_true
                == state
            )

            if positives.sum() == 0:
                state_recall[
                    state
                ] = None
                continue

            state_recall[
                state
            ] = float(
                recall_score(
                    covered_true,
                    covered_pred,
                    labels=[
                        state
                    ],
                    average="macro",
                    zero_division=0,
                )
            )

    else:
        selective_accuracy = None
        error_rate = None
        selective_macro_f1 = None

        state_recall = {
            state: None
            for state
            in TEMPORAL_STATES
        }

    if requires_review is None:
        review_rate = None

    else:
        review_array = np.asarray(
            requires_review,
            dtype=bool,
        )

        if (
            review_array.shape
            != y_true.shape
        ):
            raise ValueError(
                "requires_review must match "
                "the shape of y_true."
            )

        review_rate = float(
            review_array.mean()
        )

    return {
        "n_total": (
            total_count
        ),

        "n_covered": (
            covered_count
        ),

        "coverage": (
            coverage
        ),

        "abstention_rate": (
            abstention_rate
        ),

        "selective_accuracy": (
            selective_accuracy
        ),

        "error_rate_on_covered": (
            error_rate
        ),

        "selective_macro_f1": (
            selective_macro_f1
        ),

        "review_rate": (
            review_rate
        ),

        "state_recall": (
            state_recall
        ),
    }