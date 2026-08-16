import pandas as pd

from medchange.evaluation.dataset_summary import (
    build_dataset_summary,
)


def test_dataset_summary():
    dataframe = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "1",
                "2",
            ],

            "view_position": [
                "PA",
                "PA",
                "AP",
            ],

            "atelectasis_label": [
                1,
                0,
                0,
            ],

            "pleural_effusion_label": [
                0,
                1,
                1,
            ],
        }
    )

    summary = (
        build_dataset_summary(
            dataframe=dataframe,
            findings=[
                "atelectasis",
                "pleural_effusion",
            ],
        )
    )

    assert (
        summary["num_images"]
        == 3
    )

    assert (
        summary[
            "num_unique_patients"
        ]
        == 2
    )

    assert (
        summary["views"]["PA"]
        == 2
    )

    assert (
        summary[
            "findings"
        ][
            "pleural_effusion"
        ][
            "positive"
        ]
        == 2
    )