import pandas as pd

from medchange.data.nih.temporal_split import (
    patient_aware_temporal_split,
)


def test_patient_aware_temporal_split():
    dataframe = pd.DataFrame(
        {
            "patient_id": (
                ["1"] * 2
                + ["2"] * 2
                + ["3"] * 2
                + ["4"] * 2
                + ["5"] * 2
                + ["6"] * 2
            ),

            "pair_id": [
                f"p-{index}"
                for index in range(
                    12
                )
            ],
        }
    )

    split = (
        patient_aware_temporal_split(
            dataframe,
            train_fraction=0.5,
            validation_fraction=0.25,
            seed=42,
        )
    )

    train_patients = set(
        split.train[
            "patient_id"
        ]
    )

    validation_patients = set(
        split.validation[
            "patient_id"
        ]
    )

    test_patients = set(
        split.test[
            "patient_id"
        ]
    )

    assert (
        train_patients
        .isdisjoint(
            validation_patients
        )
    )

    assert (
        train_patients
        .isdisjoint(
            test_patients
        )
    )

    assert (
        validation_patients
        .isdisjoint(
            test_patients
        )
    )