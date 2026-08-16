import pandas as pd

from medchange.data.nih.longitudinal import (
    dataframe_to_longitudinal_studies,
)
from medchange.data.nih.longitudinal_audit import (
    audit_longitudinal_dataset,
)


def test_longitudinal_audit():
    dataframe = pd.DataFrame(
        {
            "Image Index": [
                "a.png",
                "b.png",
                "c.png",
            ],

            "Finding Labels": [
                "No Finding",
                "Effusion",
                "Atelectasis",
            ],

            "Patient ID": [
                1,
                1,
                2,
            ],

            "Follow-up #": [
                0,
                1,
                0,
            ],

            "View Position": [
                "PA",
                "PA",
                "AP",
            ],
        }
    )

    studies = (
        dataframe_to_longitudinal_studies(
            dataframe
        )
    )

    result = (
        audit_longitudinal_dataset(
            studies
        )
    )

    assert (
        result[
            "num_images"
        ]
        == 3
    )

    assert (
        result[
            "num_patients"
        ]
        == 2
    )

    assert (
        result[
            "num_repeated_patients"
        ]
        == 1
    )

    assert (
        result[
            "adjacent_pair_count"
        ]
        == 1
    )