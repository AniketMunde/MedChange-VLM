from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
)


TEMPORAL_STATES = [
    "absent",
    "new",
    "persistent",
    "resolved",
]


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def normalize_finding_name(
    name: str,
) -> str:

    normalized = (
        str(
            name
        )
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    aliases = {
        "pleural_effusion":
            "pleural_effusion",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def prediction_to_state_map(
    prediction,
) -> dict[str, str]:

    state_map = {}

    for finding in (
        prediction.findings
    ):
        name = normalize_finding_name(
            finding.finding
        )

        if (
            name
            in TARGET_FINDINGS
        ):
            state_map[
                name
            ] = finding.change

    return state_map


def prediction_to_confidence_map(
    prediction,
) -> dict[str, float]:

    result = {}

    for finding in (
        prediction.findings
    ):
        name = normalize_finding_name(
            finding.finding
        )

        if (
            name
            in TARGET_FINDINGS
        ):
            result[
                name
            ] = float(
                finding.confidence
            )

    return result


def compute_qwen_temporal_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate flattened Qwen temporal predictions.

    Expected columns:
    finding
    ground_truth
    prediction
    confidence
    """

    if predictions.empty:
        raise ValueError(
            "Prediction dataframe is empty."
        )

    y_true = (
        predictions[
            "ground_truth"
        ]
        .astype(str)
        .to_numpy()
    )

    y_pred = (
        predictions[
            "prediction"
        ]
        .astype(str)
        .to_numpy()
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=TEMPORAL_STATES,
            average="macro",
            zero_division=0,
        )
    )

    state_report = (
        classification_report(
            y_true,
            y_pred,
            labels=TEMPORAL_STATES,
            output_dict=True,
            zero_division=0,
        )
    )

    state_recall = {
        state: float(
            state_report[
                state
            ][
                "recall"
            ]
        )
        for state
        in TEMPORAL_STATES
    }

    finding_metrics = {}

    for finding in (
        TARGET_FINDINGS
    ):
        subset = predictions[
            predictions[
                "finding"
            ]
            == finding
        ]

        if subset.empty:
            continue

        finding_metrics[
            finding
        ] = {
            "macro_f1": float(
                f1_score(
                    subset[
                        "ground_truth"
                    ],
                    subset[
                        "prediction"
                    ],
                    labels=TEMPORAL_STATES,
                    average="macro",
                    zero_division=0,
                )
            ),

            "n": int(
                len(
                    subset
                )
            ),
        }

    correct = predictions[
        predictions[
            "correct"
        ]
    ]

    incorrect = predictions[
        ~predictions[
            "correct"
        ]
    ]

    return {
        "macro_f1": macro_f1,

        "state_recall": (
            state_recall
        ),

        "finding_metrics": (
            finding_metrics
        ),

        "mean_confidence_correct": (
            float(
                correct[
                    "confidence"
                ].mean()
            )
            if not correct.empty
            else None
        ),

        "mean_confidence_incorrect": (
            float(
                incorrect[
                    "confidence"
                ].mean()
            )
            if not incorrect.empty
            else None
        ),
    }