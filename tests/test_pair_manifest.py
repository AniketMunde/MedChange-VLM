from medchange.data.nih.longitudinal import (
    NIHLongitudinalPair,
)
from medchange.data.nih.pair_manifest import (
    build_pair_manifest,
)


def test_build_pair_manifest():
    pair = NIHLongitudinalPair(
        patient_id="42",

        prior_image_index=(
            "prior.png"
        ),

        current_image_index=(
            "current.png"
        ),

        prior_follow_up=0,
        current_follow_up=1,

        prior_labels=(
            "Cardiomegaly",
        ),

        current_labels=(
            "Cardiomegaly",
            "Effusion",
        ),

        prior_view="PA",
        current_view="PA",

        follow_up_delta=1,
    )

    dataframe = (
        build_pair_manifest(
            [pair]
        )
    )

    assert len(
        dataframe
    ) == 1

    row = dataframe.iloc[0]

    assert (
        row[
            "pair_id"
        ]
        == "42_0_1"
    )

    assert bool(
        row[
            "same_view"
        ]
    )

    assert (
        row[
            "new_findings"
        ]
        == "Effusion"
    )

    assert (
        row[
            "resolved_findings"
        ]
        == ""
    )

    assert (
        row[
            "persistent_findings"
        ]
        == "Cardiomegaly"
    )