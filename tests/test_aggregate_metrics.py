import pandas as pd
import pytest

from medchange.evaluation.aggregate import (
    compute_macro_metrics,
)


def test_macro_metrics():
    dataframe = pd.DataFrame(
        {
            "status": [
                "evaluated",
                "evaluated",
            ],

            "auroc": [
                0.8,
                0.6,
            ],

            "auprc": [
                0.4,
                0.2,
            ],

            "f1": [
                0.5,
                0.3,
            ],

            "precision": [
                0.6,
                0.4,
            ],

            "recall": [
                0.5,
                0.5,
            ],
        }
    )

    result = (
        compute_macro_metrics(
            dataframe
        )
    )

    assert (
        result[
            "macro_auroc"
        ]
        == pytest.approx(
            0.7
        )
    )

    assert (
        result[
            "macro_auprc"
        ]
        == pytest.approx(
            0.3
        )
    )