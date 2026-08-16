from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalDatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def patient_aware_temporal_split(
    dataframe: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> TemporalDatasetSplit:
    """
    Split temporal pairs by patient.

    No patient can occur in more than one subset.
    """

    if dataframe.empty:
        raise ValueError(
            "Temporal dataframe is empty."
        )

    if (
        "patient_id"
        not in dataframe.columns
    ):
        raise ValueError(
            "Missing patient_id column."
        )

    test_fraction = (
        1.0
        - train_fraction
        - validation_fraction
    )

    if (
        train_fraction <= 0
        or validation_fraction <= 0
        or test_fraction <= 0
    ):
        raise ValueError(
            "Train, validation and test "
            "fractions must all be positive."
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

    rng = (
        np.random.default_rng(
            seed
        )
    )

    rng.shuffle(
        patients
    )

    n_patients = len(
        patients
    )

    train_end = int(
        n_patients
        * train_fraction
    )

    validation_end = (
        train_end
        + int(
            n_patients
            * validation_fraction
        )
    )

    train_patients = set(
        patients[
            :train_end
        ]
    )

    validation_patients = set(
        patients[
            train_end:
            validation_end
        ]
    )

    test_patients = set(
        patients[
            validation_end:
        ]
    )

    patient_series = (
        dataframe[
            "patient_id"
        ].astype(str)
    )

    train = dataframe[
        patient_series.isin(
            train_patients
        )
    ].copy()

    validation = dataframe[
        patient_series.isin(
            validation_patients
        )
    ].copy()

    test = dataframe[
        patient_series.isin(
            test_patients
        )
    ].copy()

    return TemporalDatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )