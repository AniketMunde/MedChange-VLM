import pandas as pd

from medchange.data.nih.subset import (
    patient_aware_split,
)


def test_patient_aware_split():
    dataframe = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "1",
                "2",
                "3",
                "3",
                "4",
            ],

            "value": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
        }
    )

    split = patient_aware_split(
        dataframe=dataframe,
        development_fraction=0.5,
        seed=42,
    )

    development_patients = set(
        split.development[
            "patient_id"
        ]
    )

    test_patients = set(
        split.test[
            "patient_id"
        ]
    )

    assert (
        development_patients
        .isdisjoint(
            test_patients
        )
    )

    assert (
        len(
            split.development
        )
        + len(
            split.test
        )
        == len(
            dataframe
        )
    )