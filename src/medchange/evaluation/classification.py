from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from medchange.evaluation.labels import (
    prepare_binary_labels,
)


@dataclass
class BinaryClassificationMetrics:
    auroc: float | None
    auprc: float | None
    f1: float
    precision: float
    recall: float
    threshold: float

    n_total: int
    n_evaluated: int
    n_positive: int
    n_negative: int
    n_excluded: int


def compute_binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """
    Compute binary metrics while excluding uncertain
    and missing labels.
    """

    raw_true = np.asarray(
        y_true,
        dtype=object,
    )

    score = np.asarray(
        y_score,
        dtype=float,
    )

    if raw_true.shape != score.shape:
        raise ValueError(
            "y_true and y_score must "
            "have identical shapes."
        )

    if raw_true.size == 0:
        raise ValueError(
            "Metric inputs cannot be empty."
        )

    true, valid_mask = (
        prepare_binary_labels(
            raw_true
        )
    )

    valid_scores = score[
        valid_mask
    ]

    if true.size == 0:
        raise ValueError(
            "No evaluable binary labels remain "
            "after removing uncertain and missing labels."
        )

    predicted = (
        valid_scores >= threshold
    ).astype(int)

    auroc = None
    auprc = None

    if np.unique(true).size == 2:
        auroc = float(
            roc_auc_score(
                true,
                valid_scores,
            )
        )

        auprc = float(
            average_precision_score(
                true,
                valid_scores,
            )
        )

    n_positive = int(
        np.sum(true == 1)
    )

    n_negative = int(
        np.sum(true == 0)
    )

    return BinaryClassificationMetrics(
        auroc=auroc,
        auprc=auprc,

        f1=float(
            f1_score(
                true,
                predicted,
                zero_division=0,
            )
        ),

        precision=float(
            precision_score(
                true,
                predicted,
                zero_division=0,
            )
        ),

        recall=float(
            recall_score(
                true,
                predicted,
                zero_division=0,
            )
        ),

        threshold=threshold,

        n_total=int(
            raw_true.size
        ),

        n_evaluated=int(
            true.size
        ),

        n_positive=n_positive,

        n_negative=n_negative,

        n_excluded=int(
            raw_true.size
            - true.size
        ),
    )