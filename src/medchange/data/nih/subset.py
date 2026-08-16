from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubsetSplit:
    development: pd.DataFrame
    test: pd.DataFrame


def patient_aware_split(
    dataframe: pd.DataFrame,
    development_fraction: float = 0.4,
    seed: int = 42,
) -> SubsetSplit:
    """
    Split an evaluated NIH subset by patient ID.

    A patient can appear in only one subset.
    """

    if dataframe.empty:
        raise ValueError(
            "Cannot split an empty dataframe."
        )

    if (
        "patient_id"
        not in dataframe.columns
    ):
        raise ValueError(
            "Missing patient_id column."
        )

    if not (
        0.0
        < development_fraction
        < 1.0
    ):
        raise ValueError(
            "development_fraction must "
            "be between 0 and 1."
        )

    patients = (
        dataframe[
            "patient_id"
        ]
        .astype(str)
        .drop_duplicates()
        .to_numpy(
            dtype=str,
            copy=True,
        )
    )


    rng = np.random.default_rng(
        seed
    )

    shuffled = patients.copy()

    rng.shuffle(
        shuffled
    )

    split_index = int(
        len(shuffled)
        * development_fraction
    )

    development_patients = set(
        shuffled[
            :split_index
        ]
    )

    test_patients = set(
        shuffled[
            split_index:
        ]
    )

    development = dataframe[
        dataframe[
            "patient_id"
        ]
        .astype(str)
        .isin(
            development_patients
        )
    ].copy()

    test = dataframe[
        dataframe[
            "patient_id"
        ]
        .astype(str)
        .isin(
            test_patients
        )
    ].copy()

    overlap = (
        set(
            development[
                "patient_id"
            ].astype(str)
        )
        & set(
            test[
                "patient_id"
            ].astype(str)
        )
    )

    if overlap:
        raise RuntimeError(
            "Patient leakage detected "
            "between development and test subsets."
        )

    return SubsetSplit(
        development=development,
        test=test,
    )