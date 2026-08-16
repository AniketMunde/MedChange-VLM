from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float


def find_best_f1_threshold(
    y_true,
    y_score,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> ThresholdResult:
    """
    Select the threshold maximizing F1.

    This function must be used on development data only.
    """

    true = np.asarray(
        y_true,
        dtype=int,
    )

    score = np.asarray(
        y_score,
        dtype=float,
    )

    if true.shape != score.shape:
        raise ValueError(
            "y_true and y_score must have identical shapes."
        )

    if true.size == 0:
        raise ValueError(
            "Calibration inputs cannot be empty."
        )

    thresholds = np.linspace(
        min_threshold,
        max_threshold,
        steps,
    )

    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        predicted = (
            score
            >= threshold
        ).astype(int)

        tp = int(
            np.sum(
                (predicted == 1)
                & (true == 1)
            )
        )

        fp = int(
            np.sum(
                (predicted == 1)
                & (true == 0)
            )
        )

        fn = int(
            np.sum(
                (predicted == 0)
                & (true == 1)
            )
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        current_f1 = float(
            f1_score(
                true,
                predicted,
                zero_division=0,
            )
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(
                threshold
            )
            best_precision = float(
                precision
            )
            best_recall = float(
                recall
            )

    return ThresholdResult(
        threshold=best_threshold,
        f1=best_f1,
        precision=best_precision,
        recall=best_recall,
    )