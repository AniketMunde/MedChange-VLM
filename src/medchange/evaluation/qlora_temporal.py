from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


TEMPORAL_STATES = [
    "absent",
    "new",
    "persistent",
    "resolved",
]


def flatten_temporal_targets(
    records: list[dict[str, Any]],
) -> tuple[
    list[str],
    list[str],
]:
    pair_ids = []
    states = []

    for record in records:
        pair_id = str(
            record[
                "pair_id"
            ]
        )

        findings = (
            record[
                "target"
            ][
                "findings"
            ]
        )

        for finding in findings:
            pair_ids.append(
                pair_id
            )

            states.append(
                str(
                    finding[
                        "change"
                    ]
                )
            )

    return (
        pair_ids,
        states,
    )


def compute_temporal_metrics(
    *,
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=TEMPORAL_STATES,
                average="macro",
                zero_division=0,
            )
        ),

        "state_recall": {
            state: float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[
                        state
                    ],
                    average="macro",
                    zero_division=0,
                )
            )

            for state
            in TEMPORAL_STATES
        },
    }


def exact_pair_match_rate(
    *,
    pair_ids: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> float:
    grouped = defaultdict(
        lambda: {
            "true": [],
            "pred": [],
        }
    )

    for (
        pair_id,
        true_state,
        pred_state,
    ) in zip(
        pair_ids,
        y_true,
        y_pred,
    ):
        grouped[
            pair_id
        ][
            "true"
        ].append(
            true_state
        )

        grouped[
            pair_id
        ][
            "pred"
        ].append(
            pred_state
        )

    matches = [
        values[
            "true"
        ]
        == values[
            "pred"
        ]

        for values
        in grouped.values()
    ]

    if not matches:
        return 0.0

    return float(
        np.mean(
            matches
        )
    )