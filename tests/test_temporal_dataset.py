import pandas as pd

from medchange.data.nih.temporal_dataset import (
    build_temporal_dataframe,
    parse_label_string,
)


def test_parse_label_string():
    result = parse_label_string(
        "Cardiomegaly|Effusion"
    )

    assert result == (
        "Cardiomegaly",
        "Effusion",
    )


def test_build_temporal_dataframe():
    pair_manifest = pd.DataFrame(
        {
            "pair_id": [
                "1_0_1"
            ],

            "patient_id": [
                "1"
            ],

            "prior_image_index": [
                "prior.png"
            ],

            "current_image_index": [
                "current.png"
            ],

            "prior_follow_up": [
                0
            ],

            "current_follow_up": [
                1
            ],

            "prior_view": [
                "PA"
            ],

            "current_view": [
                "PA"
            ],

            "same_view": [
                True
            ],

            "prior_labels": [
                "Cardiomegaly|Effusion"
            ],

            "current_labels": [
                "Cardiomegaly|Atelectasis"
            ],
        }
    )

    result = (
        build_temporal_dataframe(
            pair_manifest
        )
    )

    row = result.iloc[0]

    assert (
        row[
            "atelectasis_temporal"
        ]
        == "new"
    )

    assert (
        row[
            "pleural_effusion_temporal"
        ]
        == "resolved"
    )

    assert (
        row[
            "cardiomegaly_temporal"
        ]
        == "persistent"
    )

    assert (
        row[
            "pneumothorax_temporal"
        ]
        == "absent"
    )

    assert (
        row[
            "num_changed_findings"
        ]
        == 2
    )

    assert bool(
        row[
            "has_temporal_change"
        ]
    )