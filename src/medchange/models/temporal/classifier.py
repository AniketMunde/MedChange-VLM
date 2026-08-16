from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)



TEMPORAL_CLASSES = [
    "absent",
    "new",
    "persistent",
    "resolved",
]


@dataclass
class TemporalClassifierResult:
    macro_f1: float
    balanced_accuracy: float

    classification_report: dict
    confusion_matrix: list[
        list[int]
    ]


class TemporalLogisticClassifier:
    """
    Multiclass logistic regression temporal baseline.
    """

    def __init__(
        self,
        random_state: int = 42,
        class_weight: str | dict | None = (
            "balanced"
        ),
    ) -> None:

        self.model = (
            LogisticRegression(
                max_iter=2000,
                class_weight=(
                    class_weight
                ),
                random_state=(
                    random_state
                ),
                solver="lbfgs",
            )
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:

        self.model.fit(
            x_train,
            y_train,
        )

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        return self.model.predict(
            x
        )

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TemporalClassifierResult:

        predictions = (
            self.predict(
                x_test
            )
        )

        macro_f1 = float(
            f1_score(
                y_test,
                predictions,
                labels=TEMPORAL_CLASSES,
                average="macro",
                zero_division=0,
            )
        )

        present_classes = sorted(
            set(
                y_test.tolist()
            )
        )

        balanced_accuracy = float(
            recall_score(
                y_test,
                predictions,
                labels=present_classes,
                average="macro",
                zero_division=0,
            )
        )

        report = (
            classification_report(
                y_test,
                predictions,
                labels=TEMPORAL_CLASSES,
                output_dict=True,
                zero_division=0,
            )
        )

        matrix = (
            confusion_matrix(
                y_test,
                predictions,
                labels=TEMPORAL_CLASSES,
            )
            .astype(int)
            .tolist()
        )

        return TemporalClassifierResult(
            macro_f1=(
                macro_f1
            ),

            balanced_accuracy=(
                balanced_accuracy
            ),

            classification_report=(
                report
            ),

            confusion_matrix=(
                matrix
            ),
        )