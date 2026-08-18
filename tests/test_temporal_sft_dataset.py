import json

import pandas as pd

from medchange.training.temporal_sft_dataset import (
    TARGET_FINDINGS,
    build_sft_record,
    build_target_payload,
    derive_overall_change,
    patient_disjoint_split,
    validate_record,
)


def _row(
    patient_id: str = "1",
):
    values = {
        "pair_id": "1_0_1",
        "patient_id": (
            patient_id
        ),
    }

    for finding in (
        TARGET_FINDINGS
    ):
        values[
            f"{finding}_temporal"
        ] = "absent"

    return pd.Series(
        values
    )


def test_target_payload():
    row = _row()

    row[
        "atelectasis_temporal"
    ] = "new"

    payload = (
        build_target_payload(
            row
        )
    )

    assert (
        len(
            payload[
                "findings"
            ]
        )
        == 7
    )

    assert (
        payload[
            "overall_change"
        ]
        == "worsened"
    )


def test_mixed_change():
    assert (
        derive_overall_change(
            [
                "new",
                "resolved",
                "absent",
            ]
        )
        == "mixed"
    )


def test_build_record():
    record = (
        build_sft_record(
            row=_row(),

            prior_image_path=(
                "prior.png"
            ),

            current_image_path=(
                "current.png"
            ),
        )
    )

    assert len(
        record[
            "images"
        ]
    ) == 2

    assert len(
        record[
            "messages"
        ]
    ) == 3

    validate_record(
        record
    )


def test_assistant_target_is_json():
    record = (
        build_sft_record(
            row=_row(),

            prior_image_path=(
                "prior.png"
            ),

            current_image_path=(
                "current.png"
            ),
        )
    )

    text = (
        record[
            "messages"
        ][
            2
        ][
            "content"
        ][
            0
        ][
            "text"
        ]
    )

    payload = json.loads(
        text
    )

    assert (
        len(
            payload[
                "findings"
            ]
        )
        == 7
    )


def test_patient_disjoint_split():
    rows = []

    for patient in range(
        100
    ):
        values = {
            "pair_id":
                f"{patient}_0_1",

            "patient_id":
                str(
                    patient
                ),
        }

        for finding in (
            TARGET_FINDINGS
        ):
            values[
                f"{finding}_temporal"
            ] = "absent"

        rows.append(
            values
        )

    dataframe = (
        pd.DataFrame(
            rows
        )
    )

    splits = (
        patient_disjoint_split(
            dataframe,
            seed=42,
        )
    )

    train = set(
        splits[
            "train"
        ][
            "patient_id"
        ].astype(str)
    )

    validation = set(
        splits[
            "validation"
        ][
            "patient_id"
        ].astype(str)
    )

    test = set(
        splits[
            "test"
        ][
            "patient_id"
        ].astype(str)
    )

    assert not (
        train
        & validation
    )

    assert not (
        train
        & test
    )

    assert not (
        validation
        & test
    )