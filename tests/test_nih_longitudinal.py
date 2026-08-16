import pandas as pd

from medchange.data.nih.longitudinal import (
    build_adjacent_pairs,
    dataframe_to_longitudinal_studies,
)


def build_test_dataframe():
    return pd.DataFrame(
        {
            "Image Index": [
                "a.png",
                "b.png",
                "c.png",
                "d.png",
            ],

            "Finding Labels": [
                "No Finding",
                "Effusion",
                "Effusion|Atelectasis",
                "Pneumothorax",
            ],

            "Patient ID": [
                1,
                1,
                1,
                2,
            ],

            "Follow-up #": [
                0,
                1,
                3,
                0,
            ],

            "View Position": [
                "PA",
                "PA",
                "AP",
                "PA",
            ],
        }
    )


def test_build_adjacent_pairs():
    dataframe = (
        build_test_dataframe()
    )

    studies = (
        dataframe_to_longitudinal_studies(
            dataframe
        )
    )

    pairs = (
        build_adjacent_pairs(
            studies
        )
    )

    assert len(
        pairs
    ) == 2

    assert (
        pairs[0].prior_image_index
        == "a.png"
    )

    assert (
        pairs[0].current_image_index
        == "b.png"
    )

    assert (
        pairs[1].prior_image_index
        == "b.png"
    )

    assert (
        pairs[1].current_image_index
        == "c.png"
    )

    assert (
        pairs[1].follow_up_delta
        == 2
    )


def test_same_view_filter():
    dataframe = (
        build_test_dataframe()
    )

    studies = (
        dataframe_to_longitudinal_studies(
            dataframe
        )
    )

    pairs = (
        build_adjacent_pairs(
            studies,
            same_view_only=True,
        )
    )

    assert len(
        pairs
    ) == 1

    assert (
        pairs[0].prior_view
        == "PA"
    )

    assert (
        pairs[0].current_view
        == "PA"
    )