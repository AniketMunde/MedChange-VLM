from __future__ import annotations

from pathlib import Path

import pandas as pd

from medchange.data.nih.temporal_targets import (
    TemporalTarget,
    derive_all_temporal_targets,
)


REQUIRED_PAIR_COLUMNS = {
    "pair_id",
    "patient_id",
    "prior_image_index",
    "current_image_index",
    "prior_follow_up",
    "current_follow_up",
    "prior_view",
    "current_view",
    "same_view",
    "prior_labels",
    "current_labels",
}


def parse_label_string(
    value: object,
) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()

    text = str(
        value
    ).strip()

    if not text:
        return tuple()

    return tuple(
        label.strip()
        for label
        in text.split("|")
        if label.strip()
    )


def load_pair_manifest(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Pair manifest not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    missing = (
        REQUIRED_PAIR_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Pair manifest missing required columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Pair manifest is empty."
        )

    return dataframe


def build_temporal_dataframe(
    pair_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a longitudinal pair manifest into
    explicit per-pathology temporal targets.
    """

    missing = (
        REQUIRED_PAIR_COLUMNS
        - set(pair_manifest.columns)
    )

    if missing:
        raise ValueError(
            "Pair manifest missing required columns: "
            f"{sorted(missing)}"
        )

    rows: list[dict] = []

    for _, pair in pair_manifest.iterrows():

        prior_labels = parse_label_string(
            pair[
                "prior_labels"
            ]
        )

        current_labels = parse_label_string(
            pair[
                "current_labels"
            ]
        )

        targets = (
            derive_all_temporal_targets(
                prior_labels=prior_labels,
                current_labels=current_labels,
            )
        )

        row = {
            "pair_id": str(
                pair[
                    "pair_id"
                ]
            ),

            "patient_id": str(
                pair[
                    "patient_id"
                ]
            ),

            "prior_image_index": str(
                pair[
                    "prior_image_index"
                ]
            ),

            "current_image_index": str(
                pair[
                    "current_image_index"
                ]
            ),

            "prior_follow_up": int(
                pair[
                    "prior_follow_up"
                ]
            ),

            "current_follow_up": int(
                pair[
                    "current_follow_up"
                ]
            ),

            "prior_view": (
                pair[
                    "prior_view"
                ]
            ),

            "current_view": (
                pair[
                    "current_view"
                ]
            ),

            "same_view": bool(
                pair[
                    "same_view"
                ]
            ),

            "prior_labels": "|".join(
                prior_labels
            ),

            "current_labels": "|".join(
                current_labels
            ),
        }

        number_changed = 0

        for (
            finding,
            target,
        ) in targets.items():

            row[
                f"{finding}_temporal"
            ] = target.value

            if target in {
                TemporalTarget.NEW,
                TemporalTarget.RESOLVED,
            }:
                number_changed += 1

        row[
            "num_changed_findings"
        ] = number_changed

        row[
            "has_temporal_change"
        ] = (
            number_changed > 0
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )