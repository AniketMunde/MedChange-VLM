from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class LabelState(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    MISSING = "missing"


def classify_label(
    value: object,
) -> LabelState:
    """
    Interpret a MIMIC/CheXpert-style label.

    1.0  -> positive
    0.0  -> negative
    -1.0 -> uncertain
    NaN  -> missing
    """

    if pd.isna(value):
        return LabelState.MISSING

    numeric = float(value)

    if numeric == 1.0:
        return LabelState.POSITIVE

    if numeric == 0.0:
        return LabelState.NEGATIVE

    if numeric == -1.0:
        return LabelState.UNCERTAIN

    raise ValueError(
        f"Unsupported label value: {value}"
    )


def prepare_binary_labels(
    values: list[object]
    | np.ndarray
    | pd.Series,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Convert MIMIC-style labels into binary targets.

    For the initial benchmark:

        1   -> retain as positive
        0   -> retain as negative
        -1  -> exclude
        NaN -> exclude

    Returns
    -------
    labels:
        Binary labels containing only 0 and 1.

    valid_mask:
        Boolean mask selecting usable observations.
    """

    series = pd.Series(values)

    valid_mask = (
        series.isin(
            [0, 0.0, 1, 1.0]
        )
    ).to_numpy()

    labels = (
        series.loc[valid_mask]
        .astype(int)
        .to_numpy()
    )

    return (
        labels,
        valid_mask,
    )