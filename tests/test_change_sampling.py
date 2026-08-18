import pandas as pd

from medchange.training.change_sampling import (
    categorize_pair,
    sample_change_aware_pairs,
    state_counts,
)


FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def make_row(
    *,
    patient_id: str,
    state: str,
):
    row = {
        "patient_id":
            patient_id,

        "pair_id":
            patient_id,
    }

    for finding in (
        FINDINGS
    ):
        row[
            f"{finding}_temporal"
        ] = "absent"

    row[
        "atelectasis_temporal"
    ] = state

    return row


def test_categories():

    stable = pd.Series(
        make_row(
            patient_id="1",
            state="absent",
        )
    )

    new = pd.Series(
        make_row(
            patient_id="2",
            state="new",
        )
    )

    resolved = pd.Series(
        make_row(
            patient_id="3",
            state="resolved",
        )
    )

    persistent = pd.Series(
        make_row(
            patient_id="4",
            state="persistent",
        )
    )

    assert (
        categorize_pair(
            stable
        )
        == "stable"
    )

    assert (
        categorize_pair(
            new
        )
        == "new"
    )

    assert (
        categorize_pair(
            resolved
        )
        == "resolved"
    )

    assert (
        categorize_pair(
            persistent
        )
        == "persistent"
    )


def test_change_aware_sampling():
    rows = []

    for index in range(
        100
    ):
        state = (
            "absent"
            if index < 70
            else
            (
                "new"
                if index < 80
                else
                (
                    "resolved"
                    if index < 90
                    else
                    "persistent"
                )
            )
        )

        rows.append(
            make_row(
                patient_id=str(
                    index
                ),
                state=state,
            )
        )

    dataframe = pd.DataFrame(
        rows
    )

    sampled = (
        sample_change_aware_pairs(
            dataframe,
            max_samples=100,
            seed=42,
        )
    )

    assert len(
        sampled
    ) == 100

    counts = (
        sampled[
            "sampling_category"
        ]
        .value_counts()
    )

    assert (
        counts[
            "stable"
        ]
        == 35
    )

    assert (
        counts[
            "new"
        ]
        == 25
    )

    assert (
        counts[
            "resolved"
        ]
        == 20
    )

    assert (
        counts[
            "persistent"
        ]
        == 20
    )