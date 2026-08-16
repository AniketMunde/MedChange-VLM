from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    recall_score,
    classification_report,
    f1_score,
)


TEMPORAL_STATES = [
    "absent",
    "new",
    "persistent",
    "resolved",
]


def evaluate_fusion_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=TEMPORAL_STATES,
            average="macro",
            zero_division=0,
        )
    )

    present_classes = sorted(
        set(
            y_true.tolist()
        )
    )

    balanced_accuracy = float(
        recall_score(
            y_true,
            y_pred,
            labels = present_classes,
            average = "macro",
            zero_division=0,
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=TEMPORAL_STATES,
            output_dict=True,
            zero_division=0,
        )
    )

    return {
        "macro_f1": macro_f1,

        "balanced_accuracy": (
            balanced_accuracy
        ),

        "state_recall": {
            state: float(
                report[
                    state
                ][
                    "recall"
                ]
            )
            for state
            in TEMPORAL_STATES
        },

        "present_test_classes": (
            present_classes
        ),
    }