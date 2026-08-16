import pandas as pd

from medchange.data.nih.temporal_summary import (
    summarize_temporal_dataset,
)


def test_temporal_summary():
    dataframe = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "2",
            ],

            "has_temporal_change": [
                True,
                False,
            ],

            "num_changed_findings": [
                2,
                0,
            ],

            "atelectasis_temporal": [
                "new",
                "absent",
            ],

            "cardiomegaly_temporal": [
                "persistent",
                "absent",
            ],

            "consolidation_temporal": [
                "absent",
                "absent",
            ],

            "edema_temporal": [
                "absent",
                "absent",
            ],

            "pleural_effusion_temporal": [
                "resolved",
                "absent",
            ],

            "pneumonia_temporal": [
                "absent",
                "absent",
            ],

            "pneumothorax_temporal": [
                "absent",
                "absent",
            ],
        }
    )

    summary = (
        summarize_temporal_dataset(
            dataframe
        )
    )

    assert (
        summary[
            "num_pairs"
        ]
        == 2
    )

    assert (
        summary[
            "pairs_with_change"
        ]
        == 1
    )

    assert (
        summary[
            "findings"
        ][
            "atelectasis"
        ][
            "new"
        ]
        == 1
    )